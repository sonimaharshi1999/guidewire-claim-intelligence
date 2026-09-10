# Guidewire Claim Intelligence - Tests
# Author: Maharshi Soni | License: MIT

"""Tests for the feature engineering pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from claim_intelligence.feature_engineering import ClaimFeatureEngineer, build_features


class TestClaimFeatureEngineer:
    """Tests for ClaimFeatureEngineer."""

    def test_fit_transform_produces_numeric_df(
        self, raw_claim_data: pd.DataFrame, feature_matrix: pd.DataFrame
    ) -> None:
        """Output should be a fully numeric DataFrame."""
        for col in feature_matrix.columns:
            assert feature_matrix[col].dtype.kind in ("i", "f", "b", "u"), (
                f"Column {col} has non-numeric dtype {feature_matrix[col].dtype}"
            )

    def test_no_nans_in_output(self, feature_matrix: pd.DataFrame) -> None:
        """Feature matrix should have no NaN values."""
        assert feature_matrix.isna().sum().sum() == 0, "NaN values found in feature matrix"

    def test_fraud_flags_are_binary(self, feature_matrix: pd.DataFrame) -> None:
        """Fraud flag columns should contain only 0 or 1."""
        fraud_cols = [c for c in feature_matrix.columns if c.startswith("fraud_flag_")]
        for col in fraud_cols:
            unique = set(feature_matrix[col].unique())
            assert unique <= {0, 1}, f"{col} has non-binary values: {unique}"

    def test_fraud_indicator_sum_bounded(self, feature_matrix: pd.DataFrame) -> None:
        """fraud_indicator_sum should be between 0 and 5 (number of flags)."""
        assert feature_matrix["fraud_indicator_sum"].min() >= 0
        assert feature_matrix["fraud_indicator_sum"].max() <= 5

    def test_log_loss_amount_positive(self, feature_matrix: pd.DataFrame) -> None:
        """log_loss_amount should be positive (log1p of non-negative)."""
        assert (feature_matrix["log_loss_amount"] >= 0).all()

    def test_build_features_convenience(self, raw_claim_data: pd.DataFrame) -> None:
        """build_features returns X, y, and a fitted engineer."""
        X, y, eng = build_features(raw_claim_data, target_col="severity")
        assert len(X) == len(y)
        assert eng.is_fitted
        assert len(eng.feature_names) > 0

    def test_transform_requires_fit(self) -> None:
        """Calling transform before fit raises RuntimeError."""
        eng = ClaimFeatureEngineer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            eng.transform(pd.DataFrame({"loss_amount": [100]}))

    def test_feature_names_populated(
        self, feature_engineer: ClaimFeatureEngineer
    ) -> None:
        """feature_names should be non-empty after fit."""
        assert len(feature_engineer.feature_names) > 10
