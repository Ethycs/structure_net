"""
Experiment Surrogate Model

Predicts experiment outcomes (accuracy, efficiency, etc.) from architecture
and hyperparameter features. Sits in the **Application** layer of the
microkernel, consuming data from the **Service** layer (ExperimentDatasetBuilder).

Usage::

    from neural_architecture_lab.experiment_dataset import ExperimentDatasetBuilder
    from neural_architecture_lab.surrogate_model import ExperimentSurrogate

    builder = ExperimentDatasetBuilder()
    df = builder.to_dataframe(builder.load_from_disk(Path('data/experiment_queue')))

    surrogate = ExperimentSurrogate()
    surrogate.fit(df, target='accuracy')
    print(surrogate.feature_importance())
    print(surrogate.predict({'architecture': [784, 256, 10], 'sparsity': 0.05, 'epochs': 50}))
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports — graceful degradation
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class ExperimentSurrogate:
    """
    Predict experiment outcomes from architecture + hyperparameter features.

    Uses a RandomForestRegressor by default (good balance of accuracy,
    interpretability, and robustness to small datasets).
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Args:
            model_type: "random_forest" or "gradient_boosting".
        """
        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for ExperimentSurrogate. "
                "Install with: pip install scikit-learn"
            )
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.target_name: str = "accuracy"
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df: "pd.DataFrame",
        target: str = "accuracy",
        n_estimators: int = 100,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Train the surrogate model on historical experiment data.

        Args:
            df: DataFrame from ExperimentDatasetBuilder.to_dataframe().
            target: Target column name to predict.
            n_estimators: Number of trees/estimators.
            cv_folds: Cross-validation folds for evaluation.

        Returns:
            Dict with training metrics (cv_score, feature_importance, etc.)
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas is required")

        from .experiment_dataset import ExperimentDatasetBuilder

        builder = ExperimentDatasetBuilder()
        self.feature_columns = builder.get_feature_names(df, target=target)
        self.target_name = target

        if target not in df.columns:
            raise ValueError(f"Target '{target}' not in DataFrame. Available: {list(df.columns)}")

        # Prepare features
        X = df[self.feature_columns].fillna(0).values.astype(float)
        y = df[target].fillna(0).values.astype(float)

        if len(X) < 5:
            raise ValueError(f"Need at least 5 experiments, got {len(X)}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create model
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )

        # Cross-validate
        actual_folds = min(cv_folds, len(X))
        if actual_folds >= 2:
            cv_scores = cross_val_score(
                self.model, X_scaled, y, cv=actual_folds, scoring="r2"
            )
            cv_r2 = float(np.mean(cv_scores))
        else:
            cv_r2 = 0.0

        # Fit on all data
        self.model.fit(X_scaled, y)
        self._is_fitted = True

        train_r2 = float(self.model.score(X_scaled, y))

        result = {
            "n_samples": len(X),
            "n_features": len(self.feature_columns),
            "cv_r2": cv_r2,
            "train_r2": train_r2,
            "target": target,
            "model_type": self.model_type,
            "feature_importance": self.feature_importance(),
        }

        logger.info(
            "Surrogate fitted: %d samples, CV R²=%.3f, Train R²=%.3f",
            len(X), cv_r2, train_r2,
        )
        return result

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, config: Dict[str, Any]) -> float:
        """
        Predict outcome for a new experiment configuration.

        The config dict should have keys matching the feature columns
        (architecture features, training params, etc.). Missing keys
        default to 0.
        """
        self._check_fitted()
        features = np.array(
            [float(config.get(col, 0)) for col in self.feature_columns]
        ).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return float(self.model.predict(features_scaled)[0])

    def predict_batch(self, configs: List[Dict[str, Any]]) -> np.ndarray:
        """Predict outcomes for multiple configurations."""
        self._check_fitted()
        X = np.array(
            [[float(c.get(col, 0)) for col in self.feature_columns] for c in configs]
        )
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    # ------------------------------------------------------------------
    # Suggestion / search
    # ------------------------------------------------------------------

    def suggest_configs(
        self,
        base_config: Dict[str, Any],
        vary_keys: List[str],
        n_candidates: int = 1000,
        n_suggestions: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Suggest promising configurations via random search + prediction.

        Args:
            base_config: Base configuration (fixed params).
            vary_keys: Keys to randomly vary.
            n_candidates: Number of random candidates to evaluate.
            n_suggestions: Number of top candidates to return.

        Returns:
            List of (config, predicted_score) tuples, sorted by score.
        """
        self._check_fitted()

        # Generate random variations
        candidates = []
        for _ in range(n_candidates):
            config = dict(base_config)
            for key in vary_keys:
                if key in base_config:
                    val = base_config[key]
                    if isinstance(val, float):
                        config[key] = val * np.random.uniform(0.1, 10.0)
                    elif isinstance(val, int) and val > 0:
                        config[key] = max(1, int(val * np.random.uniform(0.25, 4.0)))
                    elif isinstance(val, bool):
                        config[key] = np.random.choice([True, False])
            candidates.append(config)

        # Predict all
        predictions = self.predict_batch(candidates)

        # Return top-k
        top_indices = np.argsort(predictions)[-n_suggestions:][::-1]
        return [
            {**candidates[i], f"predicted_{self.target_name}": float(predictions[i])}
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores, sorted descending."""
        self._check_fitted()
        importances = self.model.feature_importances_
        pairs = sorted(
            zip(self.feature_columns, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return {name: float(score) for name, score in pairs}

    def top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return the top-n most important features."""
        imp = self.feature_importance()
        return list(imp.items())[:n]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the fitted surrogate model to disk."""
        self._check_fitted()
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "target_name": self.target_name,
            "model_type": self.model_type,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Surrogate saved to %s", path)

    def load(self, path: Path) -> None:
        """Load a previously saved surrogate model."""
        import pickle

        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.model = state["model"]
        self.scaler = state["scaler"]
        self.feature_columns = state["feature_columns"]
        self.target_name = state["target_name"]
        self.model_type = state["model_type"]
        self._is_fitted = True
        logger.info("Surrogate loaded from %s (%d features)", path, len(self.feature_columns))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Surrogate model not fitted. Call .fit() first.")
