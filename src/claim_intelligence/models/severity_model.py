# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
Claim severity prediction model (1-5 scale).

Uses Gradient Boosting classification trained on engineered features to
predict how severe a claim will be.  Severity drives adjuster assignment,
reserve setting, and SIU referral priority.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

from ..config import SEVERITY_LEVELS, SEVERITY_MODEL_CONFIG, SeverityModelConfig


class SeverityModel:
    """Gradient-boosted classifier for claim severity (1-5).

    Parameters
    ----------
    config : SeverityModelConfig, optional
        Hyperparameters.  Defaults to module-level config.
    """

    def __init__(self, config: Optional[SeverityModelConfig] = None) -> None:
        self.config = config or SEVERITY_MODEL_CONFIG
        self.model: Optional[GradientBoostingClassifier] = None
        self.is_trained: bool = False
        self.metrics: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the severity classifier and return evaluation metrics.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix from :class:`ClaimFeatureEngineer`.
        y : pd.Series
            Severity labels (1-5).
        test_size : float
            Fraction held out for evaluation.

        Returns
        -------
        dict
            Accuracy, weighted-F1, and per-class report.
        """
        self.feature_names = list(X.columns)

        # Use stratified split when every class has enough samples;
        # fall back to unstratified when rare classes have < 2 members.
        min_class_count = y.value_counts().min()
        stratify = y if min_class_count >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.random_state,
            stratify=stratify,
        )

        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
            "test_size": len(y_test),
            "train_size": len(y_train),
        }
        return self.metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return severity predictions (1-5) for each row in *X*."""
        self._check_trained()
        return self.model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for severity levels."""
        self._check_trained()
        return self.model.predict_proba(X)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances sorted descending."""
        self._check_trained()
        importances = self.model.feature_importances_  # type: ignore[union-attr]
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the trained model to *path*."""
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.model, "feature_names": self.feature_names,
                 "metrics": self.metrics, "config": self.config},
                f,
            )

    def load(self, path: str | Path) -> None:
        """Load a previously saved model from *path*."""
        with open(Path(path), "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.metrics = data["metrics"]
        self.config = data["config"]
        self.is_trained = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def severity_label(level: int) -> str:
        """Map numeric severity to its human-readable label."""
        return SEVERITY_LEVELS.get(level, "Unknown")

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")
