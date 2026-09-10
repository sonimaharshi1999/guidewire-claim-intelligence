# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
Fraud risk scoring model using Isolation Forest.

Flags claims that exhibit anomalous patterns consistent with fraud.
Combines unsupervised anomaly detection with domain-driven fraud indicator
features (late reporting, excessive prior claims, quick litigation, etc.)
to produce a 0-100 fraud risk score.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..config import FRAUD_THRESHOLDS, FraudThresholds


class FraudModel:
    """Isolation Forest anomaly detector for fraud risk scoring.

    The model assigns a risk score from 0 (low risk) to 100 (high risk).
    Claims scoring above a configurable threshold are flagged for SIU
    referral.

    Parameters
    ----------
    thresholds : FraudThresholds, optional
        Contamination rate and fraud indicator cut-offs.
    siu_referral_threshold : float
        Minimum score (0-100) to flag a claim for SIU.  Default 65.
    """

    def __init__(
        self,
        thresholds: Optional[FraudThresholds] = None,
        siu_referral_threshold: float = 65.0,
    ) -> None:
        self.thresholds = thresholds or FRAUD_THRESHOLDS
        self.siu_referral_threshold = siu_referral_threshold
        self.model: Optional[IsolationForest] = None
        self.is_trained: bool = False
        self.feature_names: List[str] = []
        self.metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Fraud-relevant feature subset
    # ------------------------------------------------------------------

    @staticmethod
    def _fraud_feature_cols() -> List[str]:
        """Columns most relevant to fraud detection."""
        return [
            "loss_amount",
            "log_loss_amount",
            "reported_delay_days",
            "prior_claims_count",
            "policy_age_months",
            "litigation_flag",
            "litigation_filed_days",
            "fault_rating",
            "fraud_flag_late_report",
            "fraud_flag_prior_claims",
            "fraud_flag_quick_litigation",
            "fraud_flag_new_policy",
            "fraud_flag_round_amount",
            "fraud_indicator_sum",
            "delay_x_prior",
            "new_policy_x_prior",
        ]

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select only fraud-relevant columns, tolerating missing ones."""
        cols = [c for c in self._fraud_feature_cols() if c in X.columns]
        return X[cols]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y_fraud: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Fit the Isolation Forest on *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (full or fraud-subset).
        y_fraud : pd.Series, optional
            Ground-truth fraud labels for evaluation (not used in fitting).

        Returns
        -------
        dict
            Training summary and, if *y_fraud* is provided, detection stats.
        """
        X_fraud = self._select_features(X)
        self.feature_names = list(X_fraud.columns)

        self.model = IsolationForest(
            n_estimators=self.thresholds.n_estimators,
            contamination=self.thresholds.contamination,
            random_state=self.thresholds.random_state,
        )
        self.model.fit(X_fraud)
        self.is_trained = True

        scores = self.score(X)
        self.metrics = {
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "flagged_pct": float(np.mean(scores >= self.siu_referral_threshold) * 100),
            "n_samples": len(X),
        }

        if y_fraud is not None:
            flagged = scores >= self.siu_referral_threshold
            actual_fraud = y_fraud.astype(bool)
            tp = int((flagged & actual_fraud).sum())
            fp = int((flagged & ~actual_fraud).sum())
            fn = int((~flagged & actual_fraud).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            self.metrics.update({
                "precision": precision,
                "recall": recall,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            })

        return self.metrics

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud risk scores (0-100) for each row in *X*.

        Higher scores indicate higher fraud risk.
        """
        self._check_trained()
        X_fraud = self._select_features(X)

        # Isolation Forest decision_function: lower (more negative) = more anomalous
        raw = self.model.decision_function(X_fraud)  # type: ignore[union-attr]

        # Normalize to 0-100 where 100 = most anomalous
        min_val, max_val = raw.min(), raw.max()
        if max_val - min_val < 1e-10:
            return np.full(len(raw), 50.0)
        normalized = (max_val - raw) / (max_val - min_val) * 100
        return np.clip(normalized, 0, 100)

    def flag_for_siu(self, X: pd.DataFrame) -> np.ndarray:
        """Return boolean array: True where a claim should be referred to SIU."""
        scores = self.score(X)
        return scores >= self.siu_referral_threshold

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the trained model to *path*."""
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.model, "feature_names": self.feature_names,
                 "metrics": self.metrics, "thresholds": self.thresholds,
                 "siu_referral_threshold": self.siu_referral_threshold},
                f,
            )

    def load(self, path: str | Path) -> None:
        """Load a previously saved model from *path*."""
        with open(Path(path), "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.metrics = data["metrics"]
        self.thresholds = data["thresholds"]
        self.siu_referral_threshold = data["siu_referral_threshold"]
        self.is_trained = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")
