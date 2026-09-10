# Guidewire Claim Intelligence - Tests
# Author: Maharshi Soni | License: MIT

"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from claim_intelligence.api.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client with lifespan context (trains models)."""
    with TestClient(app) as c:
        yield c


class TestAPI:
    """Tests for the FastAPI REST API."""

    def test_health_endpoint(self, client: TestClient) -> None:
        """GET /health should return 200 with model_ready=True."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_ready"] is True

    def test_metrics_endpoint(self, client: TestClient) -> None:
        """GET /metrics should return training metrics."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "severity_accuracy" in data
        assert "reserve_r2" in data
        assert data["n_training_samples"] > 0

    def test_score_single_claim(self, client: TestClient) -> None:
        """POST /score should return a full scoring response."""
        payload = {
            "claim_number": "CLM-API-001",
            "line_of_business": "Personal Auto",
            "loss_type": "Collision",
            "loss_amount": 15000.0,
            "reported_delay_days": 5,
            "fault_rating": 60,
            "prior_claims_count": 1,
            "policy_age_months": 36,
            "litigation_flag": False,
            "damage_type": "Vehicle Repairable",
            "claimant_type": "Insured",
            "coverage_type": "Collision",
        }
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["claim_number"] == "CLM-API-001"
        assert 1 <= data["severity"]["level"] <= 5
        assert data["reserve_estimate"] > 0
        assert 0 <= data["fraud"]["score"] <= 100
        assert data["adjuster"]["primary"] != ""

    def test_score_fraud_claim(self, client: TestClient) -> None:
        """POST /score with fraud indicators should return elevated fraud score."""
        payload = {
            "claim_number": "CLM-FRAUD-API",
            "line_of_business": "Personal Auto",
            "loss_type": "Collision",
            "loss_amount": 50000.0,
            "reported_delay_days": 45,
            "fault_rating": 15,
            "prior_claims_count": 7,
            "policy_age_months": 2,
            "litigation_flag": True,
            "litigation_filed_days": 3,
            "damage_type": "Vehicle Total Loss",
            "claimant_type": "Insured",
            "coverage_type": "Collision",
        }
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["fraud"]["active_flags"]) >= 2

    def test_batch_scoring(self, client: TestClient) -> None:
        """POST /score/batch should return a summary."""
        claims = [
            {
                "claim_number": f"CLM-BATCH-{i:03d}",
                "line_of_business": "Personal Auto",
                "loss_type": "Collision",
                "loss_amount": 5000.0 + i * 1000,
                "damage_type": "Vehicle Repairable",
                "claimant_type": "Insured",
                "coverage_type": "Collision",
            }
            for i in range(5)
        ]
        resp = client.post("/score/batch", json=claims)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_claims"] == 5

    def test_explain_endpoint(self, client: TestClient) -> None:
        """POST /explain should return explanation with top_factors."""
        payload = {
            "claim_number": "CLM-EXPLAIN-001",
            "line_of_business": "Homeowners",
            "loss_type": "Fire",
            "loss_amount": 85000.0,
            "damage_type": "Structural",
            "claimant_type": "Insured",
            "coverage_type": "Property",
        }
        resp = client.post("/explain", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "severity" in data
        assert "reserve" in data
        assert "fraud" in data
        assert "top_factors" in data
        assert len(data["top_factors"]) > 0

    def test_feature_importance_endpoints(self, client: TestClient) -> None:
        """GET /features/severity and /features/reserve should return lists."""
        for endpoint in ["/features/severity", "/features/reserve"]:
            resp = client.get(endpoint)
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) > 0
            assert "feature" in data[0]
            assert "importance" in data[0]
