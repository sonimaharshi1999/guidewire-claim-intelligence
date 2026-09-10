# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
Adjuster recommendation engine.

Routes claims to the most appropriate adjuster type based on line of
business, predicted severity, fraud risk, and claim complexity.  This is
a rule-plus-score system, not a black-box ML model, because adjuster
assignment in real carriers follows explicit business rules that adjusters
and managers need to audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import ADJUSTER_LOB_MAPPING, ADJUSTER_TYPES


@dataclass
class AdjusterRecommendation:
    """A single adjuster assignment recommendation."""

    primary_adjuster: str
    confidence: float  # 0-1
    reasoning: str
    secondary_adjuster: Optional[str] = None
    siu_referral: bool = False


class AdjusterRecommender:
    """Rule-based adjuster recommendation engine.

    Matches claims to adjuster types using LOB mapping, severity thresholds,
    fraud flags, and complexity scoring.  Designed to be fully auditable --
    every recommendation includes a human-readable reasoning string.
    """

    def __init__(
        self,
        siu_score_threshold: float = 65.0,
        high_severity_threshold: int = 4,
    ) -> None:
        self.siu_score_threshold = siu_score_threshold
        self.high_severity_threshold = high_severity_threshold

    def recommend(
        self,
        line_of_business: str,
        severity: int,
        fraud_score: float,
        loss_amount: float,
        litigation_flag: bool = False,
        body_part: Optional[str] = None,
        loss_type: Optional[str] = None,
    ) -> AdjusterRecommendation:
        """Recommend an adjuster type for a single claim.

        Parameters
        ----------
        line_of_business : str
            The claim's LOB (e.g., "Personal Auto").
        severity : int
            Predicted severity (1-5).
        fraud_score : float
            Fraud risk score (0-100).
        loss_amount : float
            Claimed loss in dollars.
        litigation_flag : bool
            Whether the claim is in litigation.
        body_part : str, optional
            Injured body part (Workers Comp only).
        loss_type : str, optional
            Specific loss type.

        Returns
        -------
        AdjusterRecommendation
        """
        reasons: List[str] = []

        # Rule 1: SIU referral for high fraud scores ----------------------
        siu_referral = fraud_score >= self.siu_score_threshold
        if siu_referral:
            reasons.append(
                f"Fraud score {fraud_score:.0f} exceeds SIU threshold "
                f"({self.siu_score_threshold})"
            )

        # Rule 2: LOB-based primary assignment -----------------------------
        primary = self._lob_assignment(line_of_business, loss_type, body_part)
        reasons.append(f"LOB '{line_of_business}' maps to {primary}")

        # Rule 3: WC sub-specialization -----------------------------------
        if line_of_business == "Workers Compensation":
            if loss_type in ("Medical Only", "Occupational Disease"):
                primary = "WC Medical"
                reasons.append("Medical-only WC claim routed to WC Medical")
            else:
                primary = "WC Indemnity"
                reasons.append("Lost-time WC claim routed to WC Indemnity")

        # Rule 4: Severity escalation --------------------------------------
        secondary: Optional[str] = None
        if severity >= self.high_severity_threshold:
            secondary = "Commercial Lines"
            reasons.append(
                f"Severity {severity} >= {self.high_severity_threshold}: "
                "escalated to Commercial Lines as secondary"
            )

        # Rule 5: Litigation override to commercial -----------------------
        if litigation_flag and primary not in ("Commercial Lines", "SIU"):
            secondary = "Commercial Lines"
            reasons.append("Litigation active: Commercial Lines added as secondary")

        # Rule 6: SIU becomes primary for extreme fraud -------------------
        if siu_referral and fraud_score >= 85:
            secondary = primary
            primary = "SIU"
            reasons.append("Very high fraud score: SIU promoted to primary adjuster")

        # Confidence -------------------------------------------------------
        confidence = self._compute_confidence(
            severity, fraud_score, litigation_flag, loss_amount,
        )

        return AdjusterRecommendation(
            primary_adjuster=primary,
            confidence=confidence,
            reasoning="; ".join(reasons),
            secondary_adjuster=secondary,
            siu_referral=siu_referral,
        )

    def recommend_batch(
        self,
        df: pd.DataFrame,
        severity_col: str = "severity_pred",
        fraud_col: str = "fraud_score",
    ) -> pd.DataFrame:
        """Recommend adjusters for every row in *df*.

        Returns a copy of *df* with added recommendation columns.
        """
        results = []
        for _, row in df.iterrows():
            rec = self.recommend(
                line_of_business=row.get("line_of_business", "General Liability"),
                severity=int(row.get(severity_col, 3)),
                fraud_score=float(row.get(fraud_col, 0.0)),
                loss_amount=float(row.get("loss_amount", 0.0)),
                litigation_flag=bool(row.get("litigation_flag", False)),
                body_part=row.get("body_part"),
                loss_type=row.get("loss_type"),
            )
            results.append({
                "recommended_adjuster": rec.primary_adjuster,
                "secondary_adjuster": rec.secondary_adjuster or "",
                "adjuster_confidence": round(rec.confidence, 3),
                "siu_referral": rec.siu_referral,
                "adjuster_reasoning": rec.reasoning,
            })

        rec_df = pd.DataFrame(results, index=df.index)
        return pd.concat([df, rec_df], axis=1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lob_assignment(
        lob: str,
        loss_type: Optional[str] = None,
        body_part: Optional[str] = None,
    ) -> str:
        """Determine the primary adjuster type from LOB."""
        for adjuster, lobs in ADJUSTER_LOB_MAPPING.items():
            if adjuster == "SIU":
                continue  # SIU is assigned by fraud score, not LOB
            if lob in lobs:
                return adjuster
        return "Commercial Lines"  # fallback

    @staticmethod
    def _compute_confidence(
        severity: int,
        fraud_score: float,
        litigation_flag: bool,
        loss_amount: float,
    ) -> float:
        """Heuristic confidence in the recommendation (0-1).

        Lower confidence when the claim is complex (high severity, fraud
        risk, litigation) because such claims may need manager review.
        """
        conf = 0.95

        # Deductions for complexity
        if severity >= 4:
            conf -= 0.10
        if fraud_score >= 50:
            conf -= 0.10
        if litigation_flag:
            conf -= 0.08
        if loss_amount > 250_000:
            conf -= 0.07

        return max(round(conf, 3), 0.30)
