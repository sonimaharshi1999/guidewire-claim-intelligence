# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
FastAPI REST API for real-time claim scoring.

Exposes endpoints for single-claim scoring, batch scoring, health checks,
and model metrics.  The pipeline trains on startup using synthetic data.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..scoring import ClaimScoringPipeline
from ..explanations import ClaimExplainer
from .schemas import (
    BatchScoreSummary,
    ClaimRequest,
    ClaimScoreResponse,
    AdjusterResponse,
    FraudResponse,
    HealthResponse,
    SeverityResponse,
    TrainingMetricsResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application-level state
# ---------------------------------------------------------------------------
pipeline: ClaimScoringPipeline | None = None
explainer: ClaimExplainer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Train models on startup."""
    global pipeline, explainer
    logger.info("Training claim intelligence models...")
    pipeline = ClaimScoringPipeline(n_training_samples=5000)
    metrics = pipeline.train()
    explainer = ClaimExplainer(
        severity_model=pipeline.severity_model,
        reserve_model=pipeline.reserve_model,
        fraud_model=pipeline.fraud_model,
        feature_engineer=pipeline.feature_engineer,
    )
    logger.info("Models trained: severity acc=%.3f, reserve R2=%.3f",
                metrics["severity"]["accuracy"],
                metrics["reserve"]["r2"])
    yield
    logger.info("Shutting down claim intelligence API.")


app = FastAPI(
    title="Guidewire Claim Intelligence API",
    description=(
        "ML-powered claim triage and fraud detection for P&C insurance. "
        "Predicts severity, estimates reserves, scores fraud risk, and "
        "recommends adjuster assignment."
    ),
    version=__version__,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check API health and model readiness."""
    return HealthResponse(
        status="healthy",
        model_ready=pipeline is not None and pipeline.is_ready,
        version=__version__,
    )


@app.get("/metrics", response_model=TrainingMetricsResponse, tags=["System"])
async def training_metrics() -> TrainingMetricsResponse:
    """Return model training metrics."""
    if pipeline is None or not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Models not ready")
    m = pipeline.training_metrics
    return TrainingMetricsResponse(
        severity_accuracy=m["severity"]["accuracy"],
        severity_f1=m["severity"]["f1_weighted"],
        reserve_r2=m["reserve"]["r2"],
        reserve_mae=m["reserve"]["mae"],
        fraud_flagged_pct=m["fraud"]["flagged_pct"],
        n_training_samples=m["n_training_samples"],
    )


@app.post("/score", response_model=ClaimScoreResponse, tags=["Scoring"])
async def score_claim(request: ClaimRequest) -> ClaimScoreResponse:
    """Score a single claim for severity, reserves, fraud, and adjuster."""
    if pipeline is None or not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Models not ready")

    claim_dict = request.model_dump()
    result = pipeline.score_claim(claim_dict)

    return ClaimScoreResponse(
        claim_number=result.claim_number,
        severity=SeverityResponse(
            level=result.severity,
            label=result.severity_label,
            probabilities={
                str(k): round(v, 4) for k, v in result.severity_probabilities.items()
            },
        ),
        reserve_estimate=result.reserve_estimate,
        fraud=FraudResponse(
            score=result.fraud_score,
            risk_level=_fraud_risk_level(result.fraud_score),
            siu_referral=result.siu_referral,
            active_flags=_get_fraud_flags(claim_dict),
        ),
        adjuster=AdjusterResponse(
            primary=result.adjuster_recommendation,
            secondary=result.secondary_adjuster,
            confidence=result.adjuster_confidence,
            reasoning=result.adjuster_reasoning,
        ),
    )


@app.post("/score/batch", response_model=BatchScoreSummary, tags=["Scoring"])
async def score_batch(claims: List[ClaimRequest]) -> BatchScoreSummary:
    """Score a batch of claims and return portfolio summary."""
    if pipeline is None or not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Models not ready")
    if len(claims) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit: 1000 claims")

    import pandas as pd
    df = pd.DataFrame([c.model_dump() for c in claims])
    result = pipeline.score_batch(df)

    return BatchScoreSummary(
        total_claims=result.n_claims,
        severity_distribution={str(k): v for k, v in result.severity_distribution.items()},
        avg_reserve=round(result.summary["avg_reserve"], 2),
        total_reserve=round(result.summary["total_reserve"], 2),
        avg_fraud_score=round(result.summary["avg_fraud_score"], 2),
        siu_referrals=result.n_siu_referrals,
        siu_referral_pct=result.summary["siu_referral_pct"],
    )


@app.post("/explain", tags=["Explanations"])
async def explain_claim(request: ClaimRequest) -> Dict[str, Any]:
    """Return a detailed explanation for a claim's scores."""
    if explainer is None or pipeline is None or not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Models not ready")

    claim_dict = request.model_dump()
    explanation = explainer.explain_claim(claim_dict)
    return explanation


@app.get("/features/severity", tags=["Explanations"])
async def severity_feature_importance() -> List[Dict[str, Any]]:
    """Return top features driving severity predictions."""
    if explainer is None:
        raise HTTPException(status_code=503, detail="Models not ready")
    fi = explainer.severity_feature_importance(top_n=10)
    return fi.to_dict("records")


@app.get("/features/reserve", tags=["Explanations"])
async def reserve_feature_importance() -> List[Dict[str, Any]]:
    """Return top features driving reserve estimates."""
    if explainer is None:
        raise HTTPException(status_code=503, detail="Models not ready")
    fi = explainer.reserve_feature_importance(top_n=10)
    return fi.to_dict("records")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fraud_risk_level(score: float) -> str:
    if score >= 85:
        return "Critical"
    if score >= 65:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def _get_fraud_flags(claim: Dict[str, Any]) -> List[str]:
    """Derive human-readable fraud flags from raw claim attributes."""
    flags: List[str] = []
    if claim.get("reported_delay_days", 0) > 30:
        flags.append("Late reporting (>30 days)")
    if claim.get("prior_claims_count", 0) > 3:
        flags.append("Excessive prior claims (>3)")
    if (claim.get("litigation_flag") and
            (claim.get("litigation_filed_days") or 999) <= 7):
        flags.append("Quick litigation (<7 days)")
    if claim.get("policy_age_months", 999) <= 3:
        flags.append("New policy (<3 months)")
    if claim.get("loss_amount", 0) % 1000 < 1:
        flags.append("Suspiciously round loss amount")
    return flags
