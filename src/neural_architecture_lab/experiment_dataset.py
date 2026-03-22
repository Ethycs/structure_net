"""
Experiment Dataset Builder

Converts stored experiment results into ML-ready tabular data for
meta-learning (predicting experiment outcomes from configurations).

This is the **return path** in the microkernel data flow:
    Kernel → Plugins → App → Logging  →  DatasetBuilder → ML

Usage::

    from neural_architecture_lab.experiment_dataset import ExperimentDatasetBuilder
    builder = ExperimentDatasetBuilder()
    experiments = builder.load_from_disk(Path('data/experiment_queue'))
    df = builder.to_dataframe(experiments)
    X, y = builder.to_features_and_targets(df, target='accuracy')
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional pandas import — graceful degradation
try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class ExperimentDatasetBuilder:
    """Load experiment results from disk/ChromaDB and convert to ML features."""

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_disk(self, *dirs: Path) -> List[Dict[str, Any]]:
        """
        Load all JSON experiment files from one or more directories.

        Typical usage::

            builder.load_from_disk(
                Path('data/experiment_queue'),
                Path('data/experiment_sent'),
            )
        """
        results: List[Dict[str, Any]] = []
        for directory in dirs:
            directory = Path(directory)
            if not directory.exists():
                logger.debug("Directory %s does not exist, skipping", directory)
                continue
            for json_file in sorted(directory.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text())
                    results.append(data)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Skipping %s: %s", json_file.name, exc)
        logger.info("Loaded %d experiment(s) from disk", len(results))
        return results

    def load_from_chromadb(
        self,
        collection_name: str = "experiments",
        chromadb_path: str = "data/chroma_db",
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Load all experiments from a ChromaDB collection as dicts."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=chromadb_path)
            collection = client.get_collection(collection_name)
            results = collection.get(limit=limit, include=["metadatas", "documents"])

            experiments = []
            for i, exp_id in enumerate(results["ids"]):
                entry: Dict[str, Any] = {"experiment_id": exp_id}
                if results["metadatas"] and i < len(results["metadatas"]):
                    entry.update(results["metadatas"][i])
                if results["documents"] and i < len(results["documents"]):
                    entry["_document"] = results["documents"][i]
                experiments.append(entry)

            logger.info("Loaded %d experiment(s) from ChromaDB", len(experiments))
            return experiments
        except Exception as exc:
            logger.warning("ChromaDB load failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def to_dataframe(self, experiments: List[Dict[str, Any]]) -> "pd.DataFrame":
        """
        Convert a list of experiment dicts into a flat pandas DataFrame.

        Extracts features suitable for ML:
          - Architecture: depth, widths, param_count, sparsity
          - Training: lr, batch_size, epochs, lr_strategy, optimizer
          - Results: accuracy, loss, training_time, convergence_speed
          - Categorical: dataset, hypothesis_category
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas is required for to_dataframe()")

        rows = []
        for exp in experiments:
            row = self._extract_features(exp)
            if row is not None:
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info("Created DataFrame with %d rows, %d columns", len(df), len(df.columns))
        return df

    def to_features_and_targets(
        self,
        df: "pd.DataFrame",
        target: str = "accuracy",
        exclude_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split DataFrame into feature matrix X and target vector y.

        Non-numeric columns are automatically dropped.
        """
        if not _HAS_PANDAS:
            raise ImportError("pandas is required")

        exclude = set(exclude_columns or [])
        exclude.add(target)
        # Also exclude ID-like columns
        exclude.update({"experiment_id", "hypothesis_id", "error", "timestamp"})

        if target not in df.columns:
            raise ValueError(f"Target '{target}' not in DataFrame columns: {list(df.columns)}")

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        feature_cols = [c for c in numeric_df.columns if c not in exclude]

        X = numeric_df[feature_cols].fillna(0).values
        y = df[target].fillna(0).values.astype(float)

        logger.info("Features: %d columns, %d samples", X.shape[1], X.shape[0])
        return X, y

    def get_feature_names(
        self,
        df: "pd.DataFrame",
        target: str = "accuracy",
        exclude_columns: Optional[List[str]] = None,
    ) -> List[str]:
        """Return the feature column names that to_features_and_targets would use."""
        if not _HAS_PANDAS:
            raise ImportError("pandas is required")

        exclude = set(exclude_columns or [])
        exclude.update({target, "experiment_id", "hypothesis_id", "error", "timestamp"})
        numeric_df = df.select_dtypes(include=[np.number])
        return [c for c in numeric_df.columns if c not in exclude]

    # ------------------------------------------------------------------
    # Internal feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, exp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract a flat feature dict from a single experiment record."""
        row: Dict[str, Any] = {}

        # IDs
        row["experiment_id"] = exp.get("experiment_id", "")
        row["hypothesis_id"] = exp.get("hypothesis_id", "")

        # --- Architecture features ---
        arch = self._get_architecture(exp)
        if arch:
            row["arch_depth"] = len(arch) - 1  # number of layers
            row["arch_input_size"] = arch[0]
            row["arch_output_size"] = arch[-1]
            row["arch_max_width"] = max(arch)
            row["arch_min_width"] = min(arch[1:]) if len(arch) > 1 else arch[0]
            row["arch_mean_width"] = float(np.mean(arch[1:-1])) if len(arch) > 2 else float(arch[-1])
            row["arch_total_params_approx"] = sum(
                arch[i] * arch[i + 1] for i in range(len(arch) - 1)
            )
        else:
            row["arch_depth"] = 0
            row["arch_input_size"] = 0
            row["arch_output_size"] = 0
            row["arch_max_width"] = 0
            row["arch_min_width"] = 0
            row["arch_mean_width"] = 0.0
            row["arch_total_params_approx"] = 0

        # --- Config / training features ---
        config = exp.get("experiment_config", exp.get("config", exp.get("parameters", {})))
        if isinstance(config, dict):
            row["sparsity"] = float(config.get("sparsity", 0.0))
            row["learning_rate"] = float(config.get("base_lr", config.get("learning_rate", 0.001)))
            row["batch_size"] = int(config.get("batch_size", 128))
            row["epochs"] = int(config.get("epochs", 0))
            row["enable_growth"] = int(bool(config.get("enable_growth", False)))
            row["growth_interval"] = int(config.get("growth_interval", 0))
            row["enable_metrics"] = int(bool(config.get("enable_metrics", False)))
            row["quick_test"] = int(bool(config.get("quick_test", False)))

            # Categorical → one-hot-friendly
            row["dataset"] = config.get("dataset", "unknown")
            row["optimizer"] = config.get("optimizer", "adam")
            row["lr_strategy"] = config.get("lr_strategy", "none")
            row["primary_metric_type"] = config.get("primary_metric_type", "accuracy")
        else:
            row["sparsity"] = 0.0
            row["learning_rate"] = 0.001
            row["batch_size"] = 128
            row["epochs"] = 0
            row["enable_growth"] = 0
            row["growth_interval"] = 0
            row["enable_metrics"] = 0
            row["quick_test"] = 0
            row["dataset"] = "unknown"
            row["optimizer"] = "adam"
            row["lr_strategy"] = "none"
            row["primary_metric_type"] = "accuracy"

        # --- Result features ---
        metrics = exp.get("metrics", {})
        if isinstance(metrics, dict):
            row["accuracy"] = float(metrics.get("accuracy", 0.0))
            row["loss"] = float(metrics.get("loss", metrics.get("test_loss", 0.0)))
            row["final_train_accuracy"] = float(metrics.get("final_train_accuracy", 0.0))
            row["growth_events"] = int(metrics.get("growth_events", 0))
            row["final_parameters"] = int(metrics.get("final_parameters", 0))
            row["final_sparsity"] = float(metrics.get("sparsity", 0.0))
        else:
            row["accuracy"] = float(exp.get("accuracy", exp.get("primary_metric", 0.0)))

        row["model_parameters"] = int(exp.get("model_parameters", row.get("final_parameters", 0)))
        row["training_time"] = float(exp.get("training_time", 0.0))
        row["primary_metric"] = float(exp.get("primary_metric", row.get("accuracy", 0.0)))
        row["error"] = exp.get("error")

        # --- Training dynamics (from history) ---
        history = exp.get("training_history", [])
        if history and isinstance(history, list) and len(history) > 0:
            accs = [
                h.get("test_accuracy", h.get("val_accuracy", 0.0))
                for h in history
                if isinstance(h, dict)
            ]
            if accs:
                row["convergence_speed"] = self._convergence_speed(accs)
                row["training_stability"] = 1.0 - float(np.std(accs[-5:])) if len(accs) >= 5 else 0.5
                row["final_test_accuracy"] = accs[-1]
                row["best_test_accuracy"] = max(accs)
            else:
                row["convergence_speed"] = 0.0
                row["training_stability"] = 0.5
        else:
            row["convergence_speed"] = 0.0
            row["training_stability"] = 0.5

        # --- Environment ---
        env = exp.get("environment", {})
        if isinstance(env, dict):
            row["device"] = env.get("device", "unknown")

        # --- Composition ---
        row["composition_hash"] = exp.get("composition_hash", "")

        return row

    def _get_architecture(self, exp: Dict[str, Any]) -> Optional[List[int]]:
        """Extract architecture list from various experiment formats."""
        # Direct field
        arch = exp.get("model_architecture")
        if isinstance(arch, list) and arch:
            return [int(x) for x in arch]

        # Nested in config
        config = exp.get("experiment_config", exp.get("config", exp.get("parameters", {})))
        if isinstance(config, dict):
            arch = config.get("architecture")
            if isinstance(arch, list) and arch:
                return [int(x) for x in arch]

        # String representation
        arch_str = exp.get("architecture", "")
        if isinstance(arch_str, str) and arch_str.startswith("["):
            try:
                parsed = json.loads(arch_str)
                if isinstance(parsed, list):
                    return [int(x) for x in parsed]
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    @staticmethod
    def _convergence_speed(accuracies: List[float]) -> float:
        """
        Compute convergence speed as normalized epoch to reach 90% of final accuracy.

        Returns value in [0, 1] where higher = faster convergence.
        """
        if not accuracies:
            return 0.0
        final = accuracies[-1]
        if final <= 0:
            return 0.0
        target = final * 0.9
        for i, acc in enumerate(accuracies):
            if acc >= target:
                return 1.0 - (i / len(accuracies))
        return 0.0
