# Guidewire Claim Intelligence - Tests
# Author: Maharshi Soni | License: MIT

"""Tests for the adjuster recommendation engine."""

from __future__ import annotations

import pandas as pd
import pytest

from claim_intelligence.models.adjuster_recommender import (
    AdjusterRecommender,
    AdjusterRecommendation,
)


class TestAdjusterRecommender:
    """Tests for AdjusterRecommender."""

    @pytest.fixture
    def recommender(self) -> AdjusterRecommender:
        return AdjusterRecommender()

    def test_auto_claim_routes_to_auto_specialist(
        self, recommender: AdjusterRecommender
    ) -> None:
        """Personal Auto claim should route to Auto Specialist."""
        rec = recommender.recommend(
            line_of_business="Personal Auto",
            severity=2,
            fraud_score=10.0,
            loss_amount=5000.0,
        )
        assert rec.primary_adjuster == "Auto Specialist"

    def test_homeowner_claim_routes_to_property_specialist(
        self, recommender: AdjusterRecommender
    ) -> None:
        """Homeowners claim should route to Property Specialist."""
        rec = recommender.recommend(
            line_of_business="Homeowners",
            severity=2,
            fraud_score=5.0,
            loss_amount=15000.0,
        )
        assert rec.primary_adjuster == "Property Specialist"

    def test_wc_medical_routing(self, recommender: AdjusterRecommender) -> None:
        """WC Medical Only claim should route to WC Medical."""
        rec = recommender.recommend(
            line_of_business="Workers Compensation",
            severity=1,
            fraud_score=5.0,
            loss_amount=3000.0,
            loss_type="Medical Only",
        )
        assert rec.primary_adjuster == "WC Medical"

    def test_wc_indemnity_routing(self, recommender: AdjusterRecommender) -> None:
        """WC Lost Time claim should route to WC Indemnity."""
        rec = recommender.recommend(
            line_of_business="Workers Compensation",
            severity=3,
            fraud_score=10.0,
            loss_amount=45000.0,
            loss_type="Lost Time",
        )
        assert rec.primary_adjuster == "WC Indemnity"

    def test_high_fraud_triggers_siu_referral(
        self, recommender: AdjusterRecommender
    ) -> None:
        """Fraud score above threshold should flag SIU referral."""
        rec = recommender.recommend(
            line_of_business="Personal Auto",
            severity=3,
            fraud_score=70.0,
            loss_amount=50000.0,
        )
        assert rec.siu_referral is True

    def test_very_high_fraud_promotes_siu_to_primary(
        self, recommender: AdjusterRecommender
    ) -> None:
        """Fraud score >= 85 should make SIU the primary adjuster."""
        rec = recommender.recommend(
            line_of_business="Personal Auto",
            severity=3,
            fraud_score=90.0,
            loss_amount=50000.0,
        )
        assert rec.primary_adjuster == "SIU"
        assert rec.siu_referral is True

    def test_high_severity_adds_secondary(
        self, recommender: AdjusterRecommender
    ) -> None:
        """Severity >= 4 should add Commercial Lines as secondary."""
        rec = recommender.recommend(
            line_of_business="Personal Auto",
            severity=4,
            fraud_score=10.0,
            loss_amount=200000.0,
        )
        assert rec.secondary_adjuster == "Commercial Lines"

    def test_litigation_adds_secondary(
        self, recommender: AdjusterRecommender
    ) -> None:
        """Litigation flag should add Commercial Lines as secondary."""
        rec = recommender.recommend(
            line_of_business="Personal Auto",
            severity=2,
            fraud_score=10.0,
            loss_amount=10000.0,
            litigation_flag=True,
        )
        assert rec.secondary_adjuster == "Commercial Lines"

    def test_confidence_decreases_with_complexity(
        self, recommender: AdjusterRecommender
    ) -> None:
        """Confidence should be lower for complex claims."""
        simple = recommender.recommend(
            line_of_business="Personal Auto",
            severity=1,
            fraud_score=5.0,
            loss_amount=2000.0,
        )
        complex_claim = recommender.recommend(
            line_of_business="Personal Auto",
            severity=5,
            fraud_score=80.0,
            loss_amount=600000.0,
            litigation_flag=True,
        )
        assert simple.confidence > complex_claim.confidence

    def test_reasoning_is_populated(
        self, recommender: AdjusterRecommender
    ) -> None:
        """Every recommendation should include a reasoning string."""
        rec = recommender.recommend(
            line_of_business="General Liability",
            severity=3,
            fraud_score=30.0,
            loss_amount=75000.0,
        )
        assert len(rec.reasoning) > 10

    def test_batch_recommendation(
        self, recommender: AdjusterRecommender
    ) -> None:
        """Batch recommendation should add columns to the DataFrame."""
        df = pd.DataFrame([
            {
                "line_of_business": "Personal Auto",
                "severity_pred": 2,
                "fraud_score": 10.0,
                "loss_amount": 5000.0,
                "litigation_flag": False,
                "body_part": None,
                "loss_type": "Collision",
            },
            {
                "line_of_business": "Homeowners",
                "severity_pred": 3,
                "fraud_score": 40.0,
                "loss_amount": 60000.0,
                "litigation_flag": True,
                "body_part": None,
                "loss_type": "Fire",
            },
        ])
        result = recommender.recommend_batch(df)
        assert "recommended_adjuster" in result.columns
        assert "adjuster_confidence" in result.columns
        assert len(result) == 2
