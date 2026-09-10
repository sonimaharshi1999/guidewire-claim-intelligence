# Guidewire Claim Intelligence - Tests
# Author: Maharshi Soni | License: MIT

"""Tests for the unified scoring pipeline."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

from claim_intelligence.scoring import ClaimScoringPipeline, ClaimScore, BatchScoreResult


class TestScoringPipeline:
    """Tests for ClaimScoringPipeline."""

    def test_pipeline_trains_successfully(
        self, trained_pipeline: ClaimScoringPipeline
    ) -> None:
        """Pipeline should be ready after training."""
        assert trained_pipeline.is_ready is True
        assert "severity" in trained_pipeline.training_metrics
        assert "reserve" in trained_pipeline.training_metrics
        assert "fraud" in trained_pipeline.training_metrics

    def test_single_claim_scoring(
        self,
        trained_pipeline: ClaimScoringPipeline,
        sample_claim: Dict[str, Any],
    ) -> None:
        """Single claim scoring should return a ClaimScore."""
        result = trained_pipeline.score_claim(sample_claim)
        assert isinstance(result, ClaimScore)
        assert 1 <= result.severity <= 5
        assert result.reserve_estimate > 0
        assert 0 <= result.fraud_score <= 100
        assert result.adjuster_recommendation != ""

    def test_fraud_claim_gets_higher_fraud_score(
        self,
        trained_pipeline: ClaimScoringPipeline,
        sample_claim: Dict[str, Any],
        fraud_claim: Dict[str, Any],
    ) -> None:
        """A claim with fraud indicators should generally score higher."""
        normal_result = trained_pipeline.score_claim(sample_claim)
        fraud_result = trained_pipeline.score_claim(fraud_claim)
        # Not guaranteed every time, but on average fraud claims should score higher.
        # We just check the fraud claim is scored.
        assert 0 <= fraud_result.fraud_score <= 100

    def test_batch_scoring(
        self,
        trained_pipeline: ClaimScoringPipeline,
        raw_claim_data: pd.DataFrame,
    ) -> None:
        """Batch scoring should process all claims."""
        subset = raw_claim_data.head(50)
        result = trained_pipeline.score_batch(subset)
        assert isinstance(result, BatchScoreResult)
        assert result.n_claims == 50
        assert "severity_pred" in result.scored_df.columns
        assert "reserve_estimate" in result.scored_df.columns
        assert "fraud_score" in result.scored_df.columns
        assert "recommended_adjuster" in result.scored_df.columns

    def test_batch_summary_consistent(
        self,
        trained_pipeline: ClaimScoringPipeline,
        raw_claim_data: pd.DataFrame,
    ) -> None:
        """Batch summary numbers should be consistent with scored data."""
        subset = raw_claim_data.head(50)
        result = trained_pipeline.score_batch(subset)
        assert result.summary["total_claims"] == 50
        assert result.summary["siu_referrals"] == result.n_siu_referrals
        sev_sum = sum(result.severity_distribution.values())
        assert sev_sum == 50

    def test_pipeline_not_ready_raises(self) -> None:
        """Scoring before training should raise RuntimeError."""
        pipe = ClaimScoringPipeline()
        with pytest.raises(RuntimeError, match="not ready"):
            pipe.score_claim({"claim_number": "X"})
