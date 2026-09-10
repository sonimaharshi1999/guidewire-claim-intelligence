# Guidewire Claim Intelligence - Tests
# Author: Maharshi Soni | License: MIT

"""Tests for severity, reserve, and fraud models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from claim_intelligence.models.severity_model import SeverityModel
from claim_intelligence.models.reserve_model import ReserveModel
from claim_intelligence.models.fraud_model import FraudModel


class TestSeverityModel:
    """Tests for the severity prediction model."""

    def test_predictions_in_valid_range(
        self,
        trained_severity_model: SeverityModel,
        feature_matrix: pd.DataFrame,
    ) -> None:
        """Predictions should be integers 1-5."""
        preds = trained_severity_model.predict(feature_matrix)
        assert all(1 <= p <= 5 for p in preds), "Predictions outside 1-5 range"

    def test_predict_proba_sums_to_one(
        self,
        trained_severity_model: SeverityModel,
        feature_matrix: pd.DataFrame,
    ) -> None:
        """Class probabilities should sum to approximately 1."""
        proba = trained_severity_model.predict_proba(feature_matrix)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_accuracy_above_threshold(
        self, trained_severity_model: SeverityModel
    ) -> None:
        """Severity accuracy should be above 40% (5-class problem)."""
        assert trained_severity_model.metrics["accuracy"] > 0.40

    def test_feature_importances_non_empty(
        self, trained_severity_model: SeverityModel
    ) -> None:
        """Feature importances should be available after training."""
        fi = trained_severity_model.feature_importances()
        assert len(fi) > 0
        assert fi["importance"].sum() > 0

    def test_save_and_load(
        self,
        trained_severity_model: SeverityModel,
        feature_matrix: pd.DataFrame,
    ) -> None:
        """Model can be saved and loaded without changing predictions."""
        preds_before = trained_severity_model.predict(feature_matrix[:10])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "severity.pkl"
            trained_severity_model.save(path)
            loaded = SeverityModel()
            loaded.load(path)
            preds_after = loaded.predict(feature_matrix[:10])
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_severity_label_mapping(self) -> None:
        """severity_label returns correct string labels."""
        assert SeverityModel.severity_label(1) == "Minor"
        assert SeverityModel.severity_label(5) == "Catastrophic"
        assert SeverityModel.severity_label(99) == "Unknown"


class TestReserveModel:
    """Tests for the reserve estimation model."""

    def test_predictions_positive(
        self,
        trained_reserve_model: ReserveModel,
        feature_matrix: pd.DataFrame,
    ) -> None:
        """Reserve estimates should be positive."""
        preds = trained_reserve_model.predict(feature_matrix)
        assert (preds > 0).all(), "Reserve estimates should all be positive"

    def test_r2_above_threshold(self, trained_reserve_model: ReserveModel) -> None:
        """R-squared should be positive (better than mean baseline)."""
        assert trained_reserve_model.metrics["r2"] > 0.0

    def test_feature_importances_available(
        self, trained_reserve_model: ReserveModel
    ) -> None:
        """Feature importances should be available after training."""
        fi = trained_reserve_model.feature_importances()
        assert len(fi) > 0

    def test_save_and_load(
        self,
        trained_reserve_model: ReserveModel,
        feature_matrix: pd.DataFrame,
    ) -> None:
        """Model survives pickle round-trip."""
        preds_before = trained_reserve_model.predict(feature_matrix[:10])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reserve.pkl"
            trained_reserve_model.save(path)
            loaded = ReserveModel()
            loaded.load(path)
            preds_after = loaded.predict(feature_matrix[:10])
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6)


class TestFraudModel:
    """Tests for the fraud detection model."""

    def test_scores_in_0_100_range(
        self,
        trained_fraud_model: FraudModel,
        feature_matrix: pd.DataFrame,
    ) -> None:
        """Fraud scores should be between 0 and 100."""
        scores = trained_fraud_model.score(feature_matrix)
        assert scores.min() >= 0, f"Min score {scores.min()} < 0"
        assert scores.max() <= 100, f"Max score {scores.max()} > 100"

    def test_siu_flags_are_boolean(
        self,
        trained_fraud_model: FraudModel,
        feature_matrix: pd.DataFrame,
    ) -> None:
        """SIU referral flags should be boolean."""
        flags = trained_fraud_model.flag_for_siu(feature_matrix)
        assert set(np.unique(flags)) <= {True, False}

    def test_flagged_percentage_reasonable(
        self, trained_fraud_model: FraudModel
    ) -> None:
        """Flagged percentage should be between 1% and 30%."""
        pct = trained_fraud_model.metrics["flagged_pct"]
        assert 1 < pct < 30, f"Flagged {pct}% -- outside expected range"

    def test_save_and_load(
        self,
        trained_fraud_model: FraudModel,
        feature_matrix: pd.DataFrame,
    ) -> None:
        """Model survives pickle round-trip."""
        scores_before = trained_fraud_model.score(feature_matrix[:10])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fraud.pkl"
            trained_fraud_model.save(path)
            loaded = FraudModel()
            loaded.load(path)
            scores_after = loaded.score(feature_matrix[:10])
        np.testing.assert_allclose(scores_before, scores_after, rtol=1e-6)
