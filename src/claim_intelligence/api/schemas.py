# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
Pydantic request / response schemas for the claim scoring API.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ClaimRequest(BaseModel):
    """Inbound claim for scoring."""

    claim_number: str = Field(..., description="Unique claim identifier")
    line_of_business: str = Field(..., description="Insurance line of business")
    loss_type: str = Field(..., description="Type of loss event")
    loss_amount: float = Field(..., ge=0, description="Claimed loss amount in USD")
    reported_delay_days: int = Field(0, ge=0, description="Days between loss and report")
    fault_rating: int = Field(50, ge=0, le=100, description="Fault percentage 0-100")
    prior_claims_count: int = Field(0, ge=0, description="Prior claims by claimant")
    policy_age_months: int = Field(12, ge=0, description="Months since policy inception")
    litigation_flag: bool = Field(False, description="Whether claim is in litigation")
    litigation_filed_days: Optional[int] = Field(None, ge=0, description="Days from loss to litigation filing")
    body_part: Optional[str] = Field(None, description="Injured body part (WC only)")
    damage_type: str = Field("Cosmetic", description="Category of damage")
    claimant_type: str = Field("Insured", description="Claimant relationship to policy")
    coverage_type: str = Field("Liability", description="Policy coverage type")

    model_config = {"json_schema_extra": {
        "examples": [{
            "claim_number": "CLM-TEST001",
            "line_of_business": "Personal Auto",
            "loss_type": "Collision",
            "loss_amount": 12500.00,
            "reported_delay_days": 3,
            "fault_rating": 75,
            "prior_claims_count": 1,
            "policy_age_months": 24,
            "litigation_flag": False,
            "damage_type": "Vehicle Repairable",
            "claimant_type": "Insured",
            "coverage_type": "Collision",
        }]
    }}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SeverityResponse(BaseModel):
    """Severity prediction detail."""

    level: int = Field(..., ge=1, le=5)
    label: str
    probabilities: Dict[str, float]


class FraudResponse(BaseModel):
    """Fraud scoring detail."""

    score: float = Field(..., ge=0, le=100)
    risk_level: str
    siu_referral: bool
    active_flags: List[str]


class AdjusterResponse(BaseModel):
    """Adjuster recommendation detail."""

    primary: str
    secondary: Optional[str] = None
    confidence: float
    reasoning: str


class ClaimScoreResponse(BaseModel):
    """Full scoring response for a single claim."""

    claim_number: str
    severity: SeverityResponse
    reserve_estimate: float
    fraud: FraudResponse
    adjuster: AdjusterResponse


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_ready: bool
    version: str


class TrainingMetricsResponse(BaseModel):
    """Training metrics response."""

    severity_accuracy: float
    severity_f1: float
    reserve_r2: float
    reserve_mae: float
    fraud_flagged_pct: float
    n_training_samples: int


class BatchScoreSummary(BaseModel):
    """Summary of batch scoring results."""

    total_claims: int
    severity_distribution: Dict[str, int]
    avg_reserve: float
    total_reserve: float
    avg_fraud_score: float
    siu_referrals: int
    siu_referral_pct: float
