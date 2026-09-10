# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
Model package for claim intelligence.

Exports the three core predictors and the adjuster recommendation engine.
"""

from .severity_model import SeverityModel
from .reserve_model import ReserveModel
from .fraud_model import FraudModel
from .adjuster_recommender import AdjusterRecommender

__all__ = ["SeverityModel", "ReserveModel", "FraudModel", "AdjusterRecommender"]
