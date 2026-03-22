#!/usr/bin/env python3
"""
Component-Based Logger for Structure Net — v2

Bridges the v2 composition schemas with the existing StandardizedLogger,
providing strict validation while maintaining backwards compatibility.

Changes from v1:
- Works with the new flexible ExperimentComposition (no 5-slot requirement).
- Logs resolved class names when available.
- Provides a ``log_composed_iteration`` convenience method for the runner hook.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import hashlib
import logging

from .standardized_logging import StandardizedLogger, LoggingConfig

logger = logging.getLogger(__name__)
from .component_schemas import (
    ExperimentComposition,
    ExperimentExecution,
    ExperimentTemplate,
    IterationData,
    ComponentSpec,
    ModelSpec,
    TrainingSpec,
    HypothesisSpec,
    validate_component_compatibility,
    STANDARD_TEMPLATES,
    # Deprecated aliases kept for import compat
    MetricSchema,
    EvolverSchema,
    ModelSchema,
    TrainerSchema,
    NALSchema,
)


class ComponentLogger:
    """
    Logger that enforces component-based schema validation.

    Wraps StandardizedLogger to provide:
    1. Component-based experiment composition
    2. Strict schema validation for each slot
    3. Template-based experiment setup
    4. Automatic logging of component interactions
    """

    def __init__(self, config: LoggingConfig = None):
        self.standard_logger = StandardizedLogger(config or LoggingConfig())
        self.active_executions: Dict[str, ExperimentExecution] = {}

    # ------------------------------------------------------------------
    # Experiment creation
    # ------------------------------------------------------------------

    def create_experiment_from_template(
        self,
        template_name: str,
        execution_id: str,
        **customizations,
    ) -> ExperimentExecution:
        """Create an experiment execution from a registered template."""
        if template_name not in STANDARD_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")

        template = STANDARD_TEMPLATES[template_name]
        composition = template.instantiate(**customizations)

        execution = ExperimentExecution(
            execution_id=execution_id,
            composition=composition,
        )

        self.active_executions[execution_id] = execution
        self._log_composition(execution)
        return execution

    def create_experiment_from_composition(
        self,
        execution_id: str,
        composition: ExperimentComposition,
    ) -> ExperimentExecution:
        """
        Create an experiment execution from a composition spec.

        This is the primary v2 entry point — accepts the flexible
        ExperimentComposition directly.
        """
        warnings = validate_component_compatibility(composition)
        for w in warnings:
            logger.warning("Component compatibility warning: %s", w)

        execution = ExperimentExecution(
            execution_id=execution_id,
            composition=composition,
        )
        self.active_executions[execution_id] = execution
        self._log_composition(execution)
        return execution

    # ------------------------------------------------------------------
    # Iteration logging
    # ------------------------------------------------------------------

    def log_iteration(
        self,
        execution_id: str,
        iteration: int,
        accuracy: Optional[float] = None,
        loss: Optional[float] = None,
        metric_outputs: Dict[str, Any] = None,
        trainer_metrics: Dict[str, Any] = None,
        evolver_actions: List[str] = None,
        model_changes: Dict[str, Any] = None,
    ):
        """Log one iteration of experiment execution."""
        if execution_id not in self.active_executions:
            raise ValueError(f"No active execution with ID: {execution_id}")

        execution = self.active_executions[execution_id]

        iteration_data = IterationData(
            iteration=iteration,
            metric_outputs=metric_outputs or {},
            evolver_actions=evolver_actions or [],
            model_changes=model_changes or {},
            trainer_metrics=trainer_metrics or {},
            accuracy=accuracy,
            loss=loss,
        )
        execution.add_iteration(iteration_data)

        # Delegate real-time metrics to the standard logger
        rt_metrics: Dict[str, Any] = {"epoch": iteration}
        if accuracy is not None:
            rt_metrics["accuracy"] = accuracy
        if loss is not None:
            rt_metrics["loss"] = loss
        if trainer_metrics:
            rt_metrics.update(trainer_metrics)

        self.standard_logger.log_metrics(
            experiment_id=execution_id,
            metrics=rt_metrics,
        )

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize_experiment(
        self,
        execution_id: str,
        final_metrics: Dict[str, Any],
        status: str = "completed",
        error: str = None,
    ) -> str:
        """Finalize an experiment execution and queue the artifact."""
        if execution_id not in self.active_executions:
            raise ValueError(f"No active execution with ID: {execution_id}")

        execution = self.active_executions[execution_id]
        execution.final_metrics = final_metrics
        execution.finalize(status=status, error=error)

        artifact_data = self._execution_to_artifact_format(execution)

        json_payload = json.dumps(artifact_data, separators=(",", ":"), default=str)
        content_hash = hashlib.sha256(json_payload.encode()).hexdigest()[:16]
        queue_file = self.standard_logger.queue_dir / f"{content_hash}.json"
        queue_file.write_text(json_payload)
        logger.info("Queued composition artifact %s (%d bytes)", content_hash, len(json_payload))

        self.standard_logger.update_experiment_status(
            execution_id, status, **final_metrics,
        )

        del self.active_executions[execution_id]
        return content_hash

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_experiments_by_component(
        self,
        component_type: str,
        component_name: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for experiments using specific components."""
        collection = self.standard_logger.experiments_collection
        if collection is None:
            return []

        try:
            type_key = f"{component_type}_type"
            query_text = f"{component_type} {component_name}"

            results = collection.query(
                query_texts=[query_text],
                n_results=limit,
                where={type_key: component_name}
                if component_type in ("analyzer", "strategy", "evolver", "trainer", "metric")
                else None,
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            experiments = []
            for i, exp_id in enumerate(results["ids"][0]):
                entry = {"experiment_id": exp_id}
                if results["metadatas"] and results["metadatas"][0]:
                    entry.update(results["metadatas"][0][i])
                if results["documents"] and results["documents"][0]:
                    entry["description"] = results["documents"][0][i]
                experiments.append(entry)
            return experiments
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_composition(self, execution: ExperimentExecution):
        """Log composition details to ChromaDB."""
        comp = execution.composition

        metadata: Dict[str, Any] = {
            "composition_hash": comp.generate_hash(),
            "model_factory": comp.model.factory_name,
            "architecture": str(comp.model.architecture),
        }

        # Record first component name of each type for easy filtering
        for field_name, meta_key in [
            ("analyzers", "analyzer_type"),
            ("strategies", "strategy_type"),
            ("evolvers", "evolver_type"),
            ("trainers", "trainer_type"),
            ("metrics", "metric_type"),
        ]:
            specs = getattr(comp, field_name)
            if specs:
                metadata[meta_key] = specs[0].component_name

        if comp.hypothesis:
            metadata["hypothesis"] = comp.hypothesis.hypothesis

        self.standard_logger.register_experiment_start(
            experiment_id=execution.execution_id,
            hypothesis_id=comp.hypothesis.hypothesis if comp.hypothesis else "",
            **metadata,
        )

    def _execution_to_artifact_format(self, execution: ExperimentExecution) -> Dict[str, Any]:
        """Convert execution to StandardizedLogger-compatible artifact."""
        comp = execution.composition

        # Component summary
        components: Dict[str, Any] = {
            "model": comp.model.model_dump(),
        }
        for field_name in ("analyzers", "strategies", "evolvers", "trainers", "metrics", "schedulers"):
            specs = getattr(comp, field_name)
            if specs:
                components[field_name] = [s.model_dump() for s in specs]

        artifact_data = {
            "experiment_id": execution.execution_id,
            "timestamp": execution.started_at.isoformat(),
            "schema_version": "2.0",
            "composition": {
                "id": comp.composition_id,
                "name": comp.name,
                "template": comp.template_name,
                "hash": comp.generate_hash(),
            },
            "components": components,
            "execution": {
                "status": execution.status,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "execution_time": execution.execution_time,
                "error": execution.error,
            },
            "results": {
                "final_metrics": execution.final_metrics,
                "iteration_count": len(execution.iteration_log),
                "peak_accuracy": max(
                    (i.accuracy for i in execution.iteration_log if i.accuracy is not None),
                    default=0,
                ),
            },
            "iteration_log": [d.model_dump() for d in execution.iteration_log],
        }

        # Hypothesis result
        if execution.status == "completed" and comp.hypothesis:
            criteria = comp.hypothesis.success_criteria
            confirmed = all(
                execution.final_metrics.get(metric, 0) >= threshold
                for metric, threshold in criteria.items()
            )
            artifact_data["results"]["hypothesis_confirmed"] = confirmed

        return artifact_data


# ============================================================================
# Convenience functions
# ============================================================================

def compare_compositions(
    comp_a: ExperimentComposition,
    comp_b: ExperimentComposition,
) -> Dict[str, Any]:
    """Compare two experiment compositions side-by-side."""
    diff: Dict[str, Any] = {
        "same_hash": comp_a.generate_hash() == comp_b.generate_hash(),
        "model_diff": {},
        "component_diffs": {},
        "summary": [],
    }

    # Model diff
    a_model = comp_a.model.model_dump()
    b_model = comp_b.model.model_dump()
    model_changes = {k: {"a": a_model.get(k), "b": b_model.get(k)} for k in set(a_model) | set(b_model) if a_model.get(k) != b_model.get(k)}
    diff["model_diff"] = model_changes
    if model_changes:
        diff["summary"].append(f"model: {len(model_changes)} field(s) differ")

    # Per-category diffs
    for field_name in ("analyzers", "strategies", "evolvers", "trainers", "metrics", "schedulers"):
        a_names = sorted(s.component_name for s in getattr(comp_a, field_name))
        b_names = sorted(s.component_name for s in getattr(comp_b, field_name))
        if a_names != b_names:
            diff["component_diffs"][field_name] = {"a": a_names, "b": b_names}
            diff["summary"].append(f"{field_name}: {a_names} vs {b_names}")

    return diff


def list_templates() -> Dict[str, Dict[str, str]]:
    """List all available experiment templates."""
    result = {}
    for key, template in STANDARD_TEMPLATES.items():
        result[key] = {
            "name": template.name,
            "description": template.description,
            "category": template.category,
            "tags": template.tags,
            "parameters": list(template.parameters.keys()),
        }
    return result
