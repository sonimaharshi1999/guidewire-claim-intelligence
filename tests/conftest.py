# Guidewire Claim Intelligence - Tests
# Author: Maharshi Soni | License: MIT

"""
Shared pytest fixtures for claim intelligence tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytest

# Ensure the src package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from claim_intelligence.config import DATA_GENERATOR_CONFIG, DataGeneratorConfig
from claim_intelligence.data_generator import ClaimDataGenerator, generate_claim_data
from claim_intelligence.feature_engineering import ClaimFeatureEngineer, build_features
from claim_intelligence.models.severity_model import SeverityModel
from claim_intelligence.models.reserve_model import ReserveModel
from claim_intelligence.models.fraud_model import FraudModel
from claim_intelligence.models.adjuster_recommender import AdjusterRecommender
from claim_intelligence.scoring import ClaimScoringPipeline


@pytest.fixture(scope="session")
def small_config() -> DataGeneratorConfig:
    """A small config for fast test runs."""
    return DataGeneratorConfig(n_samples=1500, random_state=99)


@pytest.fixture(scope="session")
def raw_claim_data(small_config: DataGeneratorConfig) -> pd.DataFrame:
    """1500-row synthetic claim dataset."""
    gen = ClaimDataGenerator(small_config)
    return gen.generate()


@pytest.fixture(scope="session")
def feature_engineer(raw_claim_data: pd.DataFrame) -> ClaimFeatureEngineer:
    """Fitted feature engineer."""
    eng = ClaimFeatureEngineer()
    eng.fit(raw_claim_data)
    return eng


@pytest.fixture(scope="session")
def feature_matrix(
    raw_claim_data: pd.DataFrame, feature_engineer: ClaimFeatureEngineer
) -> pd.DataFrame:
    """Transformed feature matrix."""
    return feature_engineer.transform(raw_claim_data)


@pytest.fixture(scope="session")
def trained_severity_model(
    feature_matrix: pd.DataFrame, raw_claim_data: pd.DataFrame
) -> SeverityModel:
    """Trained severity model."""
    model = SeverityModel()
    model.train(feature_matrix, raw_claim_data["severity"])
    return model


@pytest.fixture(scope="session")
def trained_reserve_model(
    feature_matrix: pd.DataFrame, raw_claim_data: pd.DataFrame
) -> ReserveModel:
    """Trained reserve model."""
    model = ReserveModel()
    model.train(feature_matrix, raw_claim_data["loss_amount"])
    return model


@pytest.fixture(scope="session")
def trained_fraud_model(
    feature_matrix: pd.DataFrame, raw_claim_data: pd.DataFrame
) -> FraudModel:
    """Trained fraud model."""
    model = FraudModel()
    model.train(feature_matrix, raw_claim_data["is_fraud"])
    return model


@pytest.fixture(scope="session")
def trained_pipeline() -> ClaimScoringPipeline:
    """Fully trained scoring pipeline (uses 500 samples for speed)."""
    pipe = ClaimScoringPipeline(n_training_samples=1500, random_state=99)
    pipe.train()
    return pipe


@pytest.fixture
def sample_claim() -> Dict[str, Any]:
    """A realistic Personal Auto claim for single-record tests."""
    return {
        "claim_number": "CLM-TEST001",
        "line_of_business": "Personal Auto",
        "loss_type": "Collision",
        "loss_date": "2024-03-15",
        "report_date": "2024-03-18",
        "reported_delay_days": 3,
        "loss_amount": 12500.00,
        "severity": 2,
        "fault_rating": 75,
        "litigation_flag": False,
        "litigation_filed_days": None,
        "prior_claims_count": 1,
        "policy_age_months": 24,
        "body_part": None,
        "damage_type": "Vehicle Repairable",
        "claimant_type": "Insured",
        "coverage_type": "Collision",
        "is_fraud": False,
    }


@pytest.fixture
def fraud_claim() -> Dict[str, Any]:
    """A claim exhibiting multiple fraud indicators."""
    return {
        "claim_number": "CLM-FRAUD01",
        "line_of_business": "Personal Auto",
        "loss_type": "Collision",
        "loss_date": "2024-06-01",
        "report_date": "2024-07-15",
        "reported_delay_days": 44,
        "loss_amount": 50000.00,
        "severity": 3,
        "fault_rating": 20,
        "litigation_flag": True,
        "litigation_filed_days": 5,
        "prior_claims_count": 6,
        "policy_age_months": 2,
        "body_part": None,
        "damage_type": "Vehicle Total Loss",
        "claimant_type": "Insured",
        "coverage_type": "Collision",
        "is_fraud": True,
    }
