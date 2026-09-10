# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
Feature importance and model explanation utilities.

Provides SHAP-style feature contribution analysis using permutation-based
importance and per-prediction decomposition.  Gives adjusters and managers
human-readable explanations for why a claim was scored a certain way.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .feature_engineering import ClaimFeatureEngineer
from .models.severity_model import SeverityModel
from .models.reserve_model import ReserveModel
from .models.fraud_model import FraudModel


# ---------------------------------------------------------------------------
# Human-readable feature descriptions
# ---------------------------------------------------------------------------
_FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "loss_amount": "Claimed loss amount in dollars",
    "log_loss_amount": "Log-transformed loss amount",
    "reported_delay_days": "Days between loss event and claim filing",
    "fault_rating": "Fault attribution percentage (0-100)",
    "prior_claims_count": "Number of prior claims by this claimant",
    "policy_age_months": "Months since policy inception",
    "litigation_flag": "Whether the claim is in litigation",
    "litigation_filed_days": "Days from loss to litigation filing",
    "line_of_business_enc": "Insurance line of business",
    "loss_type_enc": "Type of loss event",
    "damage_type_enc": "Category of damage",
    "claimant_type_enc": "Relationship of claimant to policy",
    "coverage_type_enc": "Coverage type on the policy",
    "body_part_enc": "Injured body part (Workers Comp)",
    "fraud_flag_late_report": "Reported more than 30 days after loss",
    "fraud_flag_prior_claims": "Claimant has more than 3 prior claims",
    "fraud_flag_quick_litigation": "Litigation filed within 7 days of loss",
    "fraud_flag_new_policy": "Policy is less than 3 months old",
    "fraud_flag_round_amount": "Loss amount is a suspiciously round number",
    "fraud_indicator_sum": "Sum of fraud indicator flags",
    "delay_x_prior": "Interaction: reporting delay * prior claims",
    "loss_x_litigation": "Interaction: loss amount * litigation flag",
    "new_policy_x_prior": "Interaction: new policy flag * prior claims",
}


class ClaimExplainer:
    """Generate human-readable explanations for model predictions.

    Parameters
    ----------
    severity_model : SeverityModel
        Trained severity classifier.
    reserve_model : ReserveModel
        Trained reserve regressor.
    fraud_model : FraudModel
        Trained fraud detector.
    feature_engineer : ClaimFeatureEngineer
        Fitted feature transformer.
    """

    def __init__(
        self,
        severity_model: SeverityModel,
        reserve_model: ReserveModel,
        fraud_model: FraudModel,
        feature_engineer: ClaimFeatureEngineer,
    ) -> None:
        self.severity_model = severity_model
        self.reserve_model = reserve_model
        self.fraud_model = fraud_model
        self.feature_engineer = feature_engineer

    # ------------------------------------------------------------------
    # Global feature importance
    # ------------------------------------------------------------------

    def severity_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Return top-N features driving severity predictions.

        Returns
        -------
        pd.DataFrame
            Columns: feature, importance, description.
        """
        fi = self.severity_model.feature_importances().head(top_n)
        fi["description"] = fi["feature"].map(_FEATURE_DESCRIPTIONS).fillna("")
        return fi

    def reserve_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Return top-N features driving reserve estimates."""
        fi = self.reserve_model.feature_importances().head(top_n)
        fi["description"] = fi["feature"].map(_FEATURE_DESCRIPTIONS).fillna("")
        return fi

    # ------------------------------------------------------------------
    # Per-claim explanation
    # ------------------------------------------------------------------

    def explain_claim(
        self,
        claim: Dict[str, Any],
        X_background: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Produce a human-readable explanation for a single claim.

        Parameters
        ----------
        claim : dict
            Raw claim record.
        X_background : pd.DataFrame, optional
            Feature matrix to use as background for contrast.

        Returns
        -------
        dict
            Keys: severity_explanation, reserve_explanation,
            fraud_explanation, top_factors.
        """
        df = pd.DataFrame([claim])
        X = self.feature_engineer.transform(df)
        row = X.iloc[0]

        # Severity explanation
        severity = int(self.severity_model.predict(X)[0])
        severity_proba = self.severity_model.predict_proba(X)[0]
        classes = self.severity_model.model.classes_

        # Top factors via feature importance * feature value deviation
        top_factors = self._top_contributing_features(
            row, self.severity_model.feature_importances()
        )

        # Fraud explanation
        fraud_score = float(self.fraud_model.score(X)[0])
        fraud_flags = self._active_fraud_flags(row)

        # Reserve explanation
        reserve = float(self.reserve_model.predict(X)[0])

        explanation: Dict[str, Any] = {
            "severity": {
                "prediction": severity,
                "label": self.severity_model.severity_label(severity),
                "confidence": float(max(severity_proba)),
                "probabilities": {
                    int(c): round(float(p), 4) for c, p in zip(classes, severity_proba)
                },
            },
            "reserve": {
                "estimate": round(reserve, 2),
                "note": self._reserve_note(reserve, severity),
            },
            "fraud": {
                "score": round(fraud_score, 2),
                "risk_level": self._fraud_risk_level(fraud_score),
                "active_flags": fraud_flags,
            },
            "top_factors": top_factors,
        }
        return explanation

    # ------------------------------------------------------------------
    # Permutation importance (model-agnostic)
    # ------------------------------------------------------------------

    def permutation_importance_severity(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 5,
    ) -> pd.DataFrame:
        """Compute permutation importance for the severity model.

        Slower but model-agnostic; useful for validation.
        """
        result = permutation_importance(
            self.severity_model.model,
            X, y,
            n_repeats=n_repeats,
            random_state=42,
        )
        return (
            pd.DataFrame({
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _top_contributing_features(
        row: pd.Series, fi: pd.DataFrame, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify features that contributed most to this prediction."""
        factors = []
        for _, frow in fi.head(top_n).iterrows():
            feat = frow["feature"]
            imp = frow["importance"]
            val = row.get(feat, 0)
            desc = _FEATURE_DESCRIPTIONS.get(feat, feat)
            factors.append({
                "feature": feat,
                "value": float(val) if isinstance(val, (int, float, np.number)) else str(val),
                "importance": round(float(imp), 4),
                "description": desc,
            })
        return factors

    @staticmethod
    def _active_fraud_flags(row: pd.Series) -> List[str]:
        """Return human-readable names of active fraud flags for a claim."""
        flag_map = {
            "fraud_flag_late_report": "Late reporting (>30 days)",
            "fraud_flag_prior_claims": "Excessive prior claims (>3)",
            "fraud_flag_quick_litigation": "Quick litigation (<7 days)",
            "fraud_flag_new_policy": "New policy (<3 months)",
            "fraud_flag_round_amount": "Suspiciously round loss amount",
        }
        active = []
        for col, label in flag_map.items():
            if row.get(col, 0) == 1:
                active.append(label)
        return active

    @staticmethod
    def _reserve_note(reserve: float, severity: int) -> str:
        """Generate a context note for the reserve estimate."""
        if severity <= 2 and reserve > 50_000:
            return "Reserve appears high relative to low severity -- review recommended."
        if severity >= 4 and reserve < 50_000:
            return "Reserve appears low relative to high severity -- review recommended."
        return "Reserve estimate is within expected range for this severity level."

    @staticmethod
    def _fraud_risk_level(score: float) -> str:
        """Map numeric fraud score to a risk label."""
        if score >= 85:
            return "Critical"
        if score >= 65:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"
