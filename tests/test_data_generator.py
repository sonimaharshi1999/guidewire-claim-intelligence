# Guidewire Claim Intelligence - Tests
# Author: Maharshi Soni | License: MIT

"""Tests for the synthetic claim data generator."""

from __future__ import annotations

import pandas as pd
import pytest

from claim_intelligence.config import LINES_OF_BUSINESS, LOSS_TYPES, SEVERITY_THRESHOLDS
from claim_intelligence.data_generator import ClaimDataGenerator, generate_claim_data, DataGeneratorConfig


class TestClaimDataGenerator:
    """Tests for ClaimDataGenerator."""

    def test_generates_correct_row_count(self, raw_claim_data: pd.DataFrame) -> None:
        """Generator produces the requested number of rows."""
        assert len(raw_claim_data) == 1500

    def test_all_lobs_represented(self, raw_claim_data: pd.DataFrame) -> None:
        """All 8 lines of business appear in the output."""
        lobs = raw_claim_data["line_of_business"].unique()
        for expected_lob in LINES_OF_BUSINESS:
            assert expected_lob in lobs, f"Missing LOB: {expected_lob}"

    def test_required_columns_present(self, raw_claim_data: pd.DataFrame) -> None:
        """All expected columns exist in the generated data."""
        required = [
            "claim_number", "line_of_business", "loss_type", "loss_date",
            "report_date", "reported_delay_days", "loss_amount", "severity",
            "fault_rating", "litigation_flag", "prior_claims_count",
            "policy_age_months", "damage_type", "claimant_type",
            "coverage_type", "is_fraud",
        ]
        for col in required:
            assert col in raw_claim_data.columns, f"Missing column: {col}"

    def test_severity_matches_loss_amount(self, raw_claim_data: pd.DataFrame) -> None:
        """Severity labels are consistent with SEVERITY_THRESHOLDS."""
        for _, row in raw_claim_data.iterrows():
            amount = row["loss_amount"]
            sev = row["severity"]
            lo, hi = SEVERITY_THRESHOLDS[sev]
            assert lo <= amount < hi, (
                f"Amount {amount} does not match severity {sev} "
                f"(expected [{lo}, {hi}))"
            )

    def test_loss_types_valid_for_lob(self, raw_claim_data: pd.DataFrame) -> None:
        """Every loss_type belongs to the valid set for its LOB."""
        for _, row in raw_claim_data.iterrows():
            lob = row["line_of_business"]
            lt = row["loss_type"]
            valid = LOSS_TYPES.get(lob, [])
            assert lt in valid, f"'{lt}' is not valid for LOB '{lob}'"

    def test_body_part_only_for_wc(self, raw_claim_data: pd.DataFrame) -> None:
        """body_part is populated only for Workers Compensation claims."""
        non_wc = raw_claim_data[raw_claim_data["line_of_business"] != "Workers Compensation"]
        assert non_wc["body_part"].isna().all(), "body_part should be null for non-WC claims"

        wc = raw_claim_data[raw_claim_data["line_of_business"] == "Workers Compensation"]
        if len(wc) > 0:
            assert wc["body_part"].notna().all(), "body_part should be set for WC claims"

    def test_fraud_rate_approximately_8_percent(self, raw_claim_data: pd.DataFrame) -> None:
        """Fraud injection rate should be roughly 8%."""
        fraud_pct = raw_claim_data["is_fraud"].mean() * 100
        assert 2 < fraud_pct < 20, f"Fraud rate {fraud_pct:.1f}% is outside expected range"

    def test_reproducibility(self) -> None:
        """Same seed produces identical data."""
        df1 = generate_claim_data(n_samples=100, random_state=77)
        df2 = generate_claim_data(n_samples=100, random_state=77)
        pd.testing.assert_frame_equal(df1, df2)

    def test_subset_lob_generation(self) -> None:
        """Generator works with a subset of LOBs."""
        df = generate_claim_data(
            n_samples=200,
            random_state=42,
            lines_of_business=["Personal Auto", "Homeowners"],
        )
        assert set(df["line_of_business"].unique()) <= {"Personal Auto", "Homeowners"}
        assert len(df) == 200

    def test_claim_numbers_unique(self, raw_claim_data: pd.DataFrame) -> None:
        """Every claim number is unique."""
        assert raw_claim_data["claim_number"].nunique() == len(raw_claim_data)
