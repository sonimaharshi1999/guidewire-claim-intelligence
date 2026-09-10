# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
Configuration constants for claim intelligence models.

Defines insurance domain enumerations, severity thresholds, adjuster types,
fraud indicator thresholds, and model hyperparameters. All values reflect
real-world P&C insurance workflows observed across 5 years of ClaimCenter
automation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Insurance Lines of Business
# ---------------------------------------------------------------------------
LINES_OF_BUSINESS: List[str] = [
    "Personal Auto",
    "Homeowners",
    "Workers Compensation",
    "Commercial Package Liability",
    "Businessowners",
    "Farmowners",
    "Commercial Auto",
    "General Liability",
]

# ---------------------------------------------------------------------------
# Loss Types by Line of Business
# ---------------------------------------------------------------------------
LOSS_TYPES: Dict[str, List[str]] = {
    "Personal Auto": [
        "Collision",
        "Comprehensive",
        "Uninsured Motorist",
        "Personal Injury Protection",
        "Medical Payments",
    ],
    "Homeowners": [
        "Fire",
        "Water Damage",
        "Wind/Hail",
        "Theft",
        "Liability",
        "Other Structure",
    ],
    "Workers Compensation": [
        "Lost Time",
        "Medical Only",
        "Fatality",
        "Occupational Disease",
        "Cumulative Trauma",
    ],
    "Commercial Package Liability": [
        "Bodily Injury",
        "Property Damage",
        "Products Liability",
        "Completed Operations",
        "Advertising Injury",
    ],
    "Businessowners": [
        "Property Damage",
        "Business Interruption",
        "Liability",
        "Equipment Breakdown",
        "Theft",
    ],
    "Farmowners": [
        "Crop Damage",
        "Livestock Loss",
        "Equipment Damage",
        "Dwelling Fire",
        "Liability",
    ],
    "Commercial Auto": [
        "Collision",
        "Cargo Damage",
        "Bodily Injury",
        "Comprehensive",
        "Uninsured Motorist",
    ],
    "General Liability": [
        "Slip and Fall",
        "Product Defect",
        "Professional Liability",
        "Advertising Injury",
        "Contractual Liability",
    ],
}

# ---------------------------------------------------------------------------
# Body Parts (Workers Compensation)
# ---------------------------------------------------------------------------
BODY_PARTS: List[str] = [
    "Head",
    "Neck",
    "Shoulder",
    "Upper Back",
    "Lower Back",
    "Arm",
    "Wrist",
    "Hand",
    "Hip",
    "Knee",
    "Ankle",
    "Foot",
    "Multiple",
    "Internal",
]

# ---------------------------------------------------------------------------
# Damage Types (Property lines)
# ---------------------------------------------------------------------------
DAMAGE_TYPES: List[str] = [
    "Structural",
    "Cosmetic",
    "Electrical",
    "Plumbing",
    "HVAC",
    "Roof",
    "Foundation",
    "Contents",
    "Landscaping",
    "Vehicle Total Loss",
    "Vehicle Repairable",
    "Minor Scratch/Dent",
]

# ---------------------------------------------------------------------------
# Claimant Types
# ---------------------------------------------------------------------------
CLAIMANT_TYPES: List[str] = [
    "Insured",
    "Third Party",
    "Employee",
    "Pedestrian",
    "Passenger",
]

# ---------------------------------------------------------------------------
# Coverage Types
# ---------------------------------------------------------------------------
COVERAGE_TYPES: List[str] = [
    "Liability",
    "Collision",
    "Comprehensive",
    "Medical Payments",
    "Uninsured Motorist",
    "Property",
    "Business Income",
    "Workers Comp Medical",
    "Workers Comp Indemnity",
    "General Liability",
]

# ---------------------------------------------------------------------------
# Severity Levels
# ---------------------------------------------------------------------------
SEVERITY_LEVELS: Dict[int, str] = {
    1: "Minor",         # Under $5K
    2: "Moderate",      # $5K - $25K
    3: "Significant",   # $25K - $100K
    4: "Severe",        # $100K - $500K
    5: "Catastrophic",  # $500K+
}

SEVERITY_THRESHOLDS: Dict[int, Tuple[float, float]] = {
    1: (0.0, 5_000.0),
    2: (5_000.0, 25_000.0),
    3: (25_000.0, 100_000.0),
    4: (100_000.0, 500_000.0),
    5: (500_000.0, float("inf")),
}

# ---------------------------------------------------------------------------
# Adjuster Types & Expertise
# ---------------------------------------------------------------------------
ADJUSTER_TYPES: List[str] = [
    "Auto Specialist",
    "Property Specialist",
    "WC Medical",
    "WC Indemnity",
    "Commercial Lines",
    "SIU",  # Special Investigation Unit
]

ADJUSTER_LOB_MAPPING: Dict[str, List[str]] = {
    "Auto Specialist": ["Personal Auto", "Commercial Auto"],
    "Property Specialist": ["Homeowners", "Farmowners", "Businessowners"],
    "WC Medical": ["Workers Compensation"],
    "WC Indemnity": ["Workers Compensation"],
    "Commercial Lines": [
        "Commercial Package Liability",
        "General Liability",
        "Businessowners",
    ],
    "SIU": LINES_OF_BUSINESS,  # SIU can handle any LOB
}

# ---------------------------------------------------------------------------
# Fraud Indicator Thresholds
# ---------------------------------------------------------------------------

@dataclass
class FraudThresholds:
    """Thresholds that indicate elevated fraud risk.

    These values come from common SIU referral triggers observed in
    ClaimCenter workflows.
    """

    reported_delay_days: int = 30
    prior_claims_count: int = 3
    early_litigation_days: int = 7
    new_policy_months: int = 3
    round_number_tolerance: float = 100.0

    # Isolation Forest parameters
    contamination: float = 0.08
    n_estimators: int = 200
    random_state: int = 42


FRAUD_THRESHOLDS = FraudThresholds()

# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class SeverityModelConfig:
    """Hyperparameters for the severity prediction gradient boosting model."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    random_state: int = 42


@dataclass
class ReserveModelConfig:
    """Hyperparameters for the reserve estimation regression model."""

    n_estimators: int = 250
    max_depth: int = 7
    learning_rate: float = 0.08
    subsample: float = 0.85
    min_samples_split: int = 8
    min_samples_leaf: int = 4
    random_state: int = 42


@dataclass
class DataGeneratorConfig:
    """Configuration for the synthetic claim data generator."""

    n_samples: int = 5000
    random_state: int = 42
    lines_of_business: List[str] = field(default_factory=lambda: LINES_OF_BUSINESS.copy())

    # Distribution weights for LOBs (reflects typical P&C portfolio)
    lob_weights: List[float] = field(default_factory=lambda: [
        0.25,   # Personal Auto (most common)
        0.20,   # Homeowners
        0.15,   # Workers Compensation
        0.10,   # Commercial Package Liability
        0.08,   # Businessowners
        0.05,   # Farmowners
        0.10,   # Commercial Auto
        0.07,   # General Liability
    ])


SEVERITY_MODEL_CONFIG = SeverityModelConfig()
RESERVE_MODEL_CONFIG = ReserveModelConfig()
DATA_GENERATOR_CONFIG = DataGeneratorConfig()
