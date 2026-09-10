# Guidewire Claim Intelligence - ML-Powered Claim Triage & Fraud Detection
# Author: Maharshi Soni | License: MIT

"""
Reserve amount estimation model.

Uses Gradient Boosting regression to predict the expected claim payout.
Accurate initial reserves reduce carried-surplus drag and improve loss-ratio
forecasting for actuarial teams.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..config import RESERVE_MODEL_CONFIG, ReserveModelConfig


class ReserveModel:
    """Gradient-boosted regressor for claim reserve estimation.

    Predicts the ``loss_amount`` (used as a proxy for initial reserve)
    from the same feature set consumed by the severity model.

    Parameters
    ----------
    config : ReserveModelConfig, optional
        Hyperparameters.
    """

    def __init__(self, config: Optional[ReserveModelConfig] = None) -> None:
        self.config = config or RESERVE_MODEL_CONFIG
        self.model: Optional[GradientBoostingRegressor] = None
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
        """Train the reserve regressor and return evaluation metrics.

        The target *y* should be ``loss_amount`` (or log-transformed).
        """
        self.feature_names = list(X.columns)

        # Log-transform target for better regression behaviour
        y_log = np.log1p(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=self.config.random_state,
        )

        self.model = GradientBoostingRegressor(
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

        y_pred_log = self.model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_actual = np.expm1(y_test)

        self.metrics = {
            "r2": float(r2_score(y_actual, y_pred)),
            "mae": float(mean_absolute_error(y_actual, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_actual, y_pred))),
            "median_ae": float(np.median(np.abs(y_actual - y_pred))),
            "test_size": len(y_test),
            "train_size": len(y_train),
        }
        return self.metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return reserve estimates in original dollar scale."""
        self._check_trained()
        y_log = self.model.predict(X)  # type: ignore[union-attr]
        return np.expm1(y_log)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self) -> pd.DataFrame:
        """Return feature importances sorted descending."""
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

    def _check_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")
