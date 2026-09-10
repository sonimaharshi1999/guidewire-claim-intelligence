# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
Synthetic claim data generator for P&C insurance.

Produces realistic ClaimCenter-style data covering 8 insurance lines with
domain-accurate distributions for loss amounts, reporting delays, body parts,
damage types, and fraud indicators. Every distribution reflects patterns
observed during 5 years of ClaimCenter automation across 15+ lines.

The generator is fully configurable: callers can adjust the number of samples,
lines of business, portfolio weights, and random seed.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    BODY_PARTS,
    CLAIMANT_TYPES,
    COVERAGE_TYPES,
    DAMAGE_TYPES,
    LINES_OF_BUSINESS,
    LOSS_TYPES,
    SEVERITY_THRESHOLDS,
    DataGeneratorConfig,
    DATA_GENERATOR_CONFIG,
)


# ---------------------------------------------------------------------------
# Loss-amount distribution parameters by LOB
# ---------------------------------------------------------------------------
# (log-normal mu, log-normal sigma) -- produces realistic claim-cost curves
_LOB_LOSS_PARAMS: Dict[str, tuple[float, float]] = {
    "Personal Auto":                (8.5, 1.2),   # median ~$4,900
    "Homeowners":                   (9.0, 1.4),   # median ~$8,100
    "Workers Compensation":         (8.8, 1.5),   # median ~$6,600
    "Commercial Package Liability": (9.8, 1.6),   # median ~$18,000
    "Businessowners":               (9.2, 1.3),   # median ~$9,900
    "Farmowners":                   (8.7, 1.3),   # median ~$6,000
    "Commercial Auto":              (9.0, 1.3),   # median ~$8,100
    "General Liability":            (9.5, 1.5),   # median ~$13,400
}

# Reporting-delay shape by LOB (Poisson lambda in days)
_LOB_DELAY_LAMBDA: Dict[str, float] = {
    "Personal Auto":                2.0,
    "Homeowners":                   5.0,
    "Workers Compensation":         3.0,
    "Commercial Package Liability": 8.0,
    "Businessowners":               6.0,
    "Farmowners":                   7.0,
    "Commercial Auto":              2.5,
    "General Liability":            10.0,
}


def _severity_from_amount(amount: float) -> int:
    """Derive severity label (1-5) from a dollar loss amount."""
    for level, (lo, hi) in SEVERITY_THRESHOLDS.items():
        if lo <= amount < hi:
            return level
    return 5


def _generate_claim_number(index: int, seed: int) -> str:
    """Produce a deterministic, ClaimCenter-style claim number."""
    raw = f"{seed}-{index}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:8].upper()
    return f"CLM-{digest}"


class ClaimDataGenerator:
    """Generates synthetic but realistic P&C insurance claim datasets.

    Parameters
    ----------
    config : DataGeneratorConfig, optional
        Controls sample count, LOB mix, and random seed.  Falls back to the
        module-level ``DATA_GENERATOR_CONFIG`` when omitted.
    """

    def __init__(self, config: Optional[DataGeneratorConfig] = None) -> None:
        self.config = config or DATA_GENERATOR_CONFIG
        self._rng = np.random.RandomState(self.config.random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """Generate *n_samples* synthetic claim records.

        Returns a :class:`~pandas.DataFrame` with columns that mirror a
        typical ClaimCenter extract.
        """
        n = n_samples or self.config.n_samples
        records: List[Dict[str, Any]] = []

        lobs = self._rng.choice(
            self.config.lines_of_business,
            size=n,
            p=self.config.lob_weights,
        )

        base_date = datetime(2023, 1, 1)

        for i, lob in enumerate(lobs):
            record = self._generate_single_claim(i, lob, base_date)
            records.append(record)

        df = pd.DataFrame(records)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_single_claim(
        self, index: int, lob: str, base_date: datetime
    ) -> Dict[str, Any]:
        """Build a single claim record with domain-realistic values."""

        claim_number = _generate_claim_number(index, self.config.random_state)

        # Loss type ---------------------------------------------------------
        loss_type = self._rng.choice(LOSS_TYPES.get(lob, ["Other"]))

        # Loss amount (log-normal, clipped at $50 minimum) ------------------
        mu, sigma = _LOB_LOSS_PARAMS.get(lob, (8.5, 1.2))
        loss_amount = float(np.clip(self._rng.lognormal(mu, sigma), 50, 2_500_000))

        # Severity derived from amount --------------------------------------
        severity = _severity_from_amount(loss_amount)

        # Reported delay (Poisson, heavier tail for commercial) -------------
        lam = _LOB_DELAY_LAMBDA.get(lob, 5.0)
        reported_delay_days = int(self._rng.poisson(lam))

        # Dates -------------------------------------------------------------
        loss_date = base_date + timedelta(days=int(self._rng.randint(0, 730)))
        report_date = loss_date + timedelta(days=reported_delay_days)

        # Policy age (exponential, most policies are seasoned) ---------------
        policy_age_months = int(np.clip(self._rng.exponential(36), 1, 240))

        # Prior claims (geometric distribution -- most have 0-1) ------------
        prior_claims_count = int(self._rng.geometric(0.55) - 1)

        # Fault rating (0-100, skewed toward high for 1st party) ------------
        fault_rating = int(np.clip(self._rng.beta(2, 5) * 100, 0, 100))

        # Litigation flag (more likely for high severity) --------------------
        lit_prob = 0.02 + 0.08 * (severity - 1)
        litigation_flag = bool(self._rng.random() < lit_prob)

        # Litigation filed days (only when litigated) -----------------------
        litigation_days: Optional[int] = None
        if litigation_flag:
            litigation_days = int(np.clip(self._rng.exponential(30), 1, 365))

        # Body part (only for WC) -------------------------------------------
        body_part: Optional[str] = None
        if lob == "Workers Compensation":
            # Lower back injuries most common in WC
            wc_weights = [
                0.05, 0.06, 0.10, 0.08, 0.22,  # Head..Lower Back
                0.08, 0.06, 0.07, 0.05, 0.10,  # Arm..Knee
                0.04, 0.03, 0.04, 0.02,         # Ankle..Internal
            ]
            body_part = self._rng.choice(BODY_PARTS, p=wc_weights)

        # Damage type -------------------------------------------------------
        damage_type = self._pick_damage_type(lob, loss_amount)

        # Claimant type -----------------------------------------------------
        claimant_type = self._pick_claimant_type(lob)

        # Coverage type -----------------------------------------------------
        coverage_type = self._pick_coverage_type(lob)

        # Fraud injection (~8 % of claims get synthetic fraud markers) ------
        is_fraud_candidate = self._rng.random() < 0.08
        if is_fraud_candidate:
            record = self._inject_fraud_signals(
                loss_amount=loss_amount,
                reported_delay_days=reported_delay_days,
                prior_claims_count=prior_claims_count,
                policy_age_months=policy_age_months,
                litigation_flag=litigation_flag,
                litigation_days=litigation_days,
            )
            loss_amount = record["loss_amount"]
            reported_delay_days = record["reported_delay_days"]
            prior_claims_count = record["prior_claims_count"]
            policy_age_months = record["policy_age_months"]
            litigation_flag = record["litigation_flag"]
            litigation_days = record["litigation_days"]
            # Re-derive severity after fraud adjustment
            severity = _severity_from_amount(loss_amount)

        return {
            "claim_number": claim_number,
            "line_of_business": lob,
            "loss_type": loss_type,
            "loss_date": loss_date.strftime("%Y-%m-%d"),
            "report_date": report_date.strftime("%Y-%m-%d"),
            "reported_delay_days": reported_delay_days,
            "loss_amount": round(loss_amount, 2),
            "severity": severity,
            "fault_rating": fault_rating,
            "litigation_flag": litigation_flag,
            "litigation_filed_days": litigation_days,
            "prior_claims_count": prior_claims_count,
            "policy_age_months": policy_age_months,
            "body_part": body_part,
            "damage_type": damage_type,
            "claimant_type": claimant_type,
            "coverage_type": coverage_type,
            "is_fraud": is_fraud_candidate,
        }

    def _inject_fraud_signals(
        self,
        loss_amount: float,
        reported_delay_days: int,
        prior_claims_count: int,
        policy_age_months: int,
        litigation_flag: bool,
        litigation_days: Optional[int],
    ) -> Dict[str, Any]:
        """Adjust claim attributes to mimic common fraud patterns."""

        # Round-number loss amounts (a classic SIU red flag)
        loss_amount = round(loss_amount / 1000) * 1000
        if loss_amount < 1000:
            loss_amount = 1000.0

        # Delayed reporting
        if self._rng.random() < 0.6:
            reported_delay_days = int(self._rng.uniform(31, 90))

        # Many prior claims
        if self._rng.random() < 0.5:
            prior_claims_count = int(self._rng.uniform(4, 10))

        # New policy
        if self._rng.random() < 0.45:
            policy_age_months = int(self._rng.uniform(1, 3))

        # Quick litigation
        if self._rng.random() < 0.4:
            litigation_flag = True
            litigation_days = int(self._rng.uniform(1, 7))

        return {
            "loss_amount": loss_amount,
            "reported_delay_days": reported_delay_days,
            "prior_claims_count": prior_claims_count,
            "policy_age_months": policy_age_months,
            "litigation_flag": litigation_flag,
            "litigation_days": litigation_days,
        }

    def _pick_damage_type(self, lob: str, loss_amount: float) -> str:
        """Select a damage type appropriate for the LOB and loss size."""
        if lob in ("Personal Auto", "Commercial Auto"):
            if loss_amount > 15_000:
                return self._rng.choice(["Vehicle Total Loss", "Structural"])
            return self._rng.choice(
                ["Vehicle Repairable", "Minor Scratch/Dent", "Cosmetic"]
            )
        if lob in ("Homeowners", "Farmowners", "Businessowners"):
            return self._rng.choice(
                ["Structural", "Roof", "Plumbing", "Electrical", "Contents", "HVAC"]
            )
        if lob == "Workers Compensation":
            return self._rng.choice(["Structural", "Contents", "Cosmetic"])
        return self._rng.choice(DAMAGE_TYPES)

    def _pick_claimant_type(self, lob: str) -> str:
        """Select claimant type based on LOB."""
        if lob == "Workers Compensation":
            return "Employee"
        if lob in ("Personal Auto", "Homeowners", "Farmowners"):
            return self._rng.choice(["Insured", "Third Party"], p=[0.7, 0.3])
        return self._rng.choice(CLAIMANT_TYPES)

    def _pick_coverage_type(self, lob: str) -> str:
        """Select a coverage type relevant to the LOB."""
        mapping: Dict[str, List[str]] = {
            "Personal Auto": ["Liability", "Collision", "Comprehensive",
                              "Medical Payments", "Uninsured Motorist"],
            "Homeowners": ["Property", "Liability"],
            "Workers Compensation": ["Workers Comp Medical", "Workers Comp Indemnity"],
            "Commercial Package Liability": ["General Liability", "Property"],
            "Businessowners": ["Property", "Business Income", "General Liability"],
            "Farmowners": ["Property", "Liability"],
            "Commercial Auto": ["Liability", "Collision", "Comprehensive"],
            "General Liability": ["General Liability"],
        }
        options = mapping.get(lob, COVERAGE_TYPES)
        return self._rng.choice(options)


def generate_claim_data(
    n_samples: int = 5000,
    random_state: int = 42,
    lines_of_business: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convenience wrapper to generate a claim dataset.

    Parameters
    ----------
    n_samples : int
        Number of claim records to produce.
    random_state : int
        Seed for reproducibility.
    lines_of_business : list[str], optional
        Subset of LOBs to include.  Defaults to all 8 P&C lines.

    Returns
    -------
    pd.DataFrame
        Synthetic claim data with realistic distributions.
    """
    lobs = lines_of_business or LINES_OF_BUSINESS
    # Rebuild weights proportionally if a subset is requested
    full_weights = DATA_GENERATOR_CONFIG.lob_weights
    indices = [LINES_OF_BUSINESS.index(l) for l in lobs]
    raw = [full_weights[i] for i in indices]
    total = sum(raw)
    normed = [w / total for w in raw]

    config = DataGeneratorConfig(
        n_samples=n_samples,
        random_state=random_state,
        lines_of_business=lobs,
        lob_weights=normed,
    )
    generator = ClaimDataGenerator(config)
    return generator.generate()
