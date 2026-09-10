# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
Feature engineering pipeline for claim intelligence models.

Transforms raw ClaimCenter-style claim records into model-ready feature
matrices.  Handles categorical encoding, derived fraud-signal features,
interaction terms, and missing-value imputation following insurance domain
conventions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import (
    BODY_PARTS,
    CLAIMANT_TYPES,
    COVERAGE_TYPES,
    DAMAGE_TYPES,
    FRAUD_THRESHOLDS,
    LINES_OF_BUSINESS,
)


# ---------------------------------------------------------------------------
# Categorical columns that need encoding
# ---------------------------------------------------------------------------
_CATEGORICAL_COLS: List[str] = [
    "line_of_business",
    "loss_type",
    "damage_type",
    "claimant_type",
    "coverage_type",
]

_OPTIONAL_CATEGORICAL_COLS: List[str] = [
    "body_part",
]


class ClaimFeatureEngineer:
    """Transform raw claim data into ML-ready features.

    Encodes categoricals, creates domain-driven fraud indicator features,
    and builds interaction terms that capture real underwriting intuition.

    Attributes
    ----------
    label_encoders : dict[str, LabelEncoder]
        Fitted encoders for each categorical column.
    is_fitted : bool
        Whether :meth:`fit` has been called.
    feature_names : list[str]
        Ordered list of output feature column names (available after fit).
    """

    def __init__(self) -> None:
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.is_fitted: bool = False
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ClaimFeatureEngineer":
        """Learn encoding vocabularies from *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Raw claim data (as produced by :mod:`data_generator`).
        """
        for col in _CATEGORICAL_COLS:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le

        for col in _OPTIONAL_CATEGORICAL_COLS:
            le = LabelEncoder()
            vals = df[col].fillna("N/A").astype(str)
            le.fit(vals)
            self.label_encoders[col] = le

        # Derive feature names from a sample transform
        sample = self._transform_impl(df.head(5))
        self.feature_names = list(sample.columns)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned transformations to *df*.

        Returns a numeric :class:`~pandas.DataFrame` suitable for
        scikit-learn estimators.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "ClaimFeatureEngineer has not been fitted. Call fit() first."
            )
        return self._transform_impl(df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform in one call."""
        self.fit(df)
        return self.transform(df)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transform_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core transformation logic."""
        out = pd.DataFrame(index=df.index)

        # 1. Encode categoricals -------------------------------------------
        for col in _CATEGORICAL_COLS:
            le = self.label_encoders[col]
            encoded = df[col].astype(str).map(
                {c: i for i, c in enumerate(le.classes_)}
            )
            # Unseen labels get -1
            out[f"{col}_enc"] = encoded.fillna(-1).astype(int)

        for col in _OPTIONAL_CATEGORICAL_COLS:
            le = self.label_encoders[col]
            vals = df[col].fillna("N/A").astype(str)
            encoded = vals.map({c: i for i, c in enumerate(le.classes_)})
            out[f"{col}_enc"] = encoded.fillna(-1).astype(int)

        # 2. Numeric pass-through ------------------------------------------
        numeric_cols = [
            "loss_amount",
            "reported_delay_days",
            "fault_rating",
            "prior_claims_count",
            "policy_age_months",
        ]
        for col in numeric_cols:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # 3. Boolean flags -------------------------------------------------
        out["litigation_flag"] = (
            df["litigation_flag"].astype(bool).astype(int)
            if "litigation_flag" in df.columns
            else 0
        )

        if "litigation_filed_days" in df.columns:
            out["litigation_filed_days"] = (
                pd.to_numeric(df["litigation_filed_days"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        else:
            out["litigation_filed_days"] = 0

        # 4. Domain-driven fraud indicators --------------------------------
        out["fraud_flag_late_report"] = (
            out["reported_delay_days"] > FRAUD_THRESHOLDS.reported_delay_days
        ).astype(int)

        out["fraud_flag_prior_claims"] = (
            out["prior_claims_count"] > FRAUD_THRESHOLDS.prior_claims_count
        ).astype(int)

        out["fraud_flag_quick_litigation"] = (
            (out["litigation_flag"] == 1)
            & (out["litigation_filed_days"] <= FRAUD_THRESHOLDS.early_litigation_days)
            & (out["litigation_filed_days"] > 0)
        ).astype(int)

        out["fraud_flag_new_policy"] = (
            out["policy_age_months"] <= FRAUD_THRESHOLDS.new_policy_months
        ).astype(int)

        # Round-number amount flag
        remainder = np.mod(out["loss_amount"], FRAUD_THRESHOLDS.round_number_tolerance)
        out["fraud_flag_round_amount"] = (remainder < 1.0).astype(int)

        # Composite fraud score (sum of individual flags)
        fraud_cols = [c for c in out.columns if c.startswith("fraud_flag_")]
        out["fraud_indicator_sum"] = out[fraud_cols].sum(axis=1)

        # 5. Interaction features ------------------------------------------
        out["log_loss_amount"] = np.log1p(out["loss_amount"])
        out["delay_x_prior"] = out["reported_delay_days"] * out["prior_claims_count"]
        out["loss_x_litigation"] = out["loss_amount"] * out["litigation_flag"]
        out["new_policy_x_prior"] = (
            out["fraud_flag_new_policy"] * out["prior_claims_count"]
        )

        return out


def build_features(
    df: pd.DataFrame,
    target_col: str = "severity",
    engineer: Optional[ClaimFeatureEngineer] = None,
) -> Tuple[pd.DataFrame, pd.Series, ClaimFeatureEngineer]:
    """One-call helper: engineer features and split X / y.

    Parameters
    ----------
    df : pd.DataFrame
        Raw claim data.
    target_col : str
        Column to use as the prediction target.
    engineer : ClaimFeatureEngineer, optional
        A pre-fitted engineer.  If *None*, a new one is fitted on *df*.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target series.
    engineer : ClaimFeatureEngineer
        The (possibly newly fitted) feature engineer.
    """
    if engineer is None:
        engineer = ClaimFeatureEngineer()
        X = engineer.fit_transform(df)
    else:
        X = engineer.transform(df)

    y = df[target_col]
    return X, y, engineer
