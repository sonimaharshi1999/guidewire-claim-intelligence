# Guidewire Claim Intelligence - Tests
# Author: Maharshi Soni | License: MIT

"""Tests for the explanation module."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

from claim_intelligence.explanations import ClaimExplainer
from claim_intelligence.scoring import ClaimScoringPipeline


@pytest.fixture(scope="module")
def explainer_instance(trained_pipeline: ClaimScoringPipeline) -> ClaimExplainer:
    """ClaimExplainer wired to the trained pipeline."""
    return ClaimExplainer(
        severity_model=trained_pipeline.severity_model,
        reserve_model=trained_pipeline.reserve_model,
        fraud_model=trained_pipeline.fraud_model,
        feature_engineer=trained_pipeline.feature_engineer,
    )


class TestClaimExplainer:
    """Tests for ClaimExplainer."""

    def test_severity_feature_importance(
        self, explainer_instance: ClaimExplainer
    ) -> None:
        """Should return a DataFrame with feature, importance, description."""
        fi = explainer_instance.severity_feature_importance(top_n=5)
        assert len(fi) == 5
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert "description" in fi.columns

    def test_reserve_feature_importance(
        self, explainer_instance: ClaimExplainer
    ) -> None:
        """Should return a non-empty DataFrame."""
        fi = explainer_instance.reserve_feature_importance(top_n=5)
        assert len(fi) == 5

    def test_explain_claim_structure(
        self,
        explainer_instance: ClaimExplainer,
        sample_claim: Dict[str, Any],
    ) -> None:
        """explain_claim should return severity, reserve, fraud, top_factors."""
        explanation = explainer_instance.explain_claim(sample_claim)
        assert "severity" in explanation
        assert "reserve" in explanation
        assert "fraud" in explanation
        assert "top_factors" in explanation
        assert explanation["severity"]["prediction"] in range(1, 6)
        assert explanation["reserve"]["estimate"] > 0

    def test_fraud_flags_for_fraud_claim(
        self,
        explainer_instance: ClaimExplainer,
        fraud_claim: Dict[str, Any],
    ) -> None:
        """Fraud claim should have active fraud flags in explanation."""
        explanation = explainer_instance.explain_claim(fraud_claim)
        assert len(explanation["fraud"]["active_flags"]) >= 1

    def test_fraud_risk_levels(
        self, explainer_instance: ClaimExplainer
    ) -> None:
        """Risk levels should map correctly."""
        assert ClaimExplainer._fraud_risk_level(90) == "Critical"
        assert ClaimExplainer._fraud_risk_level(70) == "High"
        assert ClaimExplainer._fraud_risk_level(50) == "Medium"
        assert ClaimExplainer._fraud_risk_level(20) == "Low"
