# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
Unified scoring pipeline for claim intelligence.

Orchestrates the full prediction flow: feature engineering, severity
prediction, reserve estimation, fraud scoring, and adjuster recommendation.
Supports both single-claim and batch modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import SEVERITY_LEVELS
from .data_generator import generate_claim_data
from .feature_engineering import ClaimFeatureEngineer, build_features
from .models.adjuster_recommender import AdjusterRecommender, AdjusterRecommendation
from .models.fraud_model import FraudModel
from .models.reserve_model import ReserveModel
from .models.severity_model import SeverityModel


@dataclass
class ClaimScore:
    """Complete scoring result for a single claim."""

    claim_number: str
    severity: int
    severity_label: str
    severity_probabilities: Dict[int, float]
    reserve_estimate: float
    fraud_score: float
    siu_referral: bool
    adjuster_recommendation: str
    secondary_adjuster: Optional[str]
    adjuster_confidence: float
    adjuster_reasoning: str


@dataclass
class BatchScoreResult:
    """Result container for batch scoring."""

    scored_df: pd.DataFrame
    summary: Dict[str, Any]
    n_claims: int
    n_siu_referrals: int
    severity_distribution: Dict[int, int]


class ClaimScoringPipeline:
    """End-to-end claim scoring pipeline.

    Trains all models on synthetic data (or loads pre-trained ones),
    then provides single-claim and batch scoring methods.

    Parameters
    ----------
    n_training_samples : int
        Number of synthetic claims to generate for training.
    random_state : int
        Seed for reproducibility across data generation and models.
    """

    def __init__(
        self,
        n_training_samples: int = 5000,
        random_state: int = 42,
    ) -> None:
        self.n_training_samples = n_training_samples
        self.random_state = random_state

        self.feature_engineer = ClaimFeatureEngineer()
        self.severity_model = SeverityModel()
        self.reserve_model = ReserveModel()
        self.fraud_model = FraudModel()
        self.adjuster_recommender = AdjusterRecommender()

        self.is_ready: bool = False
        self.training_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Generate training data, engineer features, and train all models.

        Returns
        -------
        dict
            Metrics from each model.
        """
        # Generate synthetic claim data
        df = generate_claim_data(
            n_samples=self.n_training_samples,
            random_state=self.random_state,
        )

        # Feature engineering
        X, y_severity, self.feature_engineer = build_features(
            df, target_col="severity"
        )
        y_reserve = df["loss_amount"]
        y_fraud = df["is_fraud"]

        # Train models
        severity_metrics = self.severity_model.train(X, y_severity)
        reserve_metrics = self.reserve_model.train(X, y_reserve)
        fraud_metrics = self.fraud_model.train(X, y_fraud)

        self.is_ready = True
        self.training_metrics = {
            "severity": severity_metrics,
            "reserve": reserve_metrics,
            "fraud": fraud_metrics,
            "n_training_samples": len(df),
        }
        return self.training_metrics

    # ------------------------------------------------------------------
    # Single-claim scoring
    # ------------------------------------------------------------------

    def score_claim(self, claim: Dict[str, Any]) -> ClaimScore:
        """Score a single claim dictionary.

        Parameters
        ----------
        claim : dict
            A claim record with the same keys as the data generator output.

        Returns
        -------
        ClaimScore
            Full scoring result.
        """
        self._check_ready()

        df = pd.DataFrame([claim])
        X = self.feature_engineer.transform(df)

        # Severity
        severity = int(self.severity_model.predict(X)[0])
        proba = self.severity_model.predict_proba(X)[0]
        classes = self.severity_model.model.classes_
        severity_probs = {int(c): float(p) for c, p in zip(classes, proba)}

        # Reserve
        reserve = float(self.reserve_model.predict(X)[0])

        # Fraud
        fraud_score = float(self.fraud_model.score(X)[0])
        siu_flag = bool(self.fraud_model.flag_for_siu(X)[0])

        # Adjuster
        rec = self.adjuster_recommender.recommend(
            line_of_business=claim.get("line_of_business", "General Liability"),
            severity=severity,
            fraud_score=fraud_score,
            loss_amount=claim.get("loss_amount", 0.0),
            litigation_flag=claim.get("litigation_flag", False),
            body_part=claim.get("body_part"),
            loss_type=claim.get("loss_type"),
        )

        return ClaimScore(
            claim_number=claim.get("claim_number", "UNKNOWN"),
            severity=severity,
            severity_label=SEVERITY_LEVELS.get(severity, "Unknown"),
            severity_probabilities=severity_probs,
            reserve_estimate=round(reserve, 2),
            fraud_score=round(fraud_score, 2),
            siu_referral=siu_flag,
            adjuster_recommendation=rec.primary_adjuster,
            secondary_adjuster=rec.secondary_adjuster,
            adjuster_confidence=rec.confidence,
            adjuster_reasoning=rec.reasoning,
        )

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_batch(self, df: pd.DataFrame) -> BatchScoreResult:
        """Score an entire DataFrame of claims.

        Parameters
        ----------
        df : pd.DataFrame
            Claims data (same schema as data generator output).

        Returns
        -------
        BatchScoreResult
        """
        self._check_ready()

        X = self.feature_engineer.transform(df)

        # Predictions
        severities = self.severity_model.predict(X)
        reserves = self.reserve_model.predict(X)
        fraud_scores = self.fraud_model.score(X)
        siu_flags = self.fraud_model.flag_for_siu(X)

        scored = df.copy()
        scored["severity_pred"] = severities
        scored["severity_label"] = [
            SEVERITY_LEVELS.get(int(s), "Unknown") for s in severities
        ]
        scored["reserve_estimate"] = np.round(reserves, 2)
        scored["fraud_score"] = np.round(fraud_scores, 2)
        scored["siu_referral"] = siu_flags

        # Adjuster recommendations
        scored = self.adjuster_recommender.recommend_batch(
            scored, severity_col="severity_pred", fraud_col="fraud_score"
        )

        # Summary
        sev_dist = {
            int(k): int(v)
            for k, v in pd.Series(severities).value_counts().sort_index().items()
        }
        n_siu = int(siu_flags.sum())

        summary = {
            "total_claims": len(df),
            "severity_distribution": sev_dist,
            "avg_reserve": float(np.mean(reserves)),
            "total_reserve": float(np.sum(reserves)),
            "avg_fraud_score": float(np.mean(fraud_scores)),
            "siu_referrals": n_siu,
            "siu_referral_pct": round(n_siu / len(df) * 100, 2),
        }

        return BatchScoreResult(
            scored_df=scored,
            summary=summary,
            n_claims=len(df),
            n_siu_referrals=n_siu,
            severity_distribution=sev_dist,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError(
                "Pipeline is not ready. Call train() first."
            )
