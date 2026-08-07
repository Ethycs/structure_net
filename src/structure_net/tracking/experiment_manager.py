"""
Kernel-level experiment tracking.

This is deliberately lightweight bookkeeping — which components ran in which
experiment, with what headline metrics — not a replacement for the NAL
research harness or the standardized evidence ledger. NAL remains the
scientific-methodology layer; these records exist so kernel services
(scorecards, health, event subscribers) can attribute activity to
experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ExperimentRecord:
    """Record of one kernel-tracked experiment."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_name: str = ""

    # Components used
    components_used: List[str] = field(default_factory=list)
    component_versions: Dict[str, str] = field(default_factory=dict)

    # Results
    metrics: Dict[str, float] = field(default_factory=dict)
    best_performance: float = 0.0

    # Status: pending, running, completed, failed
    status: str = "pending"

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0


class ExperimentManager:
    """Tracks active and completed kernel experiments."""

    def __init__(self, kernel: Any = None):
        self.kernel = kernel
        self.active_experiments: Dict[str, ExperimentRecord] = {}
        self.completed_experiments: List[ExperimentRecord] = []

    def start_experiment(self, name: str, components: List[Any]) -> str:
        """Start tracking an experiment and return its ID."""
        experiment = ExperimentRecord(
            experiment_name=name, started_at=datetime.now(), status="running"
        )
        for component in components:
            experiment.components_used.append(component.name)
            experiment.component_versions[component.name] = str(
                component.contract.version
            )
        self.active_experiments[experiment.experiment_id] = experiment
        if self.kernel is not None:
            self.kernel.log_event(
                "experiment.started",
                experiment_id=experiment.experiment_id,
                name=name,
                components=list(experiment.components_used),
            )
            for component in components:
                self.kernel.scorecard_manager.record_experiment_use(
                    component.name, experiment.experiment_id
                )
        return experiment.experiment_id

    def update_metrics(self, experiment_id: str, metrics: Dict[str, float]) -> None:
        experiment = self.active_experiments.get(experiment_id)
        if experiment is None:
            return
        experiment.metrics.update(metrics)
        if metrics:
            experiment.best_performance = max(
                experiment.best_performance, max(metrics.values())
            )

    def complete_experiment(
        self,
        experiment_id: str,
        metrics: Optional[Dict[str, float]] = None,
        success: bool = True,
    ) -> Optional[ExperimentRecord]:
        """Complete an experiment and move it to the completed list."""
        experiment = self.active_experiments.pop(experiment_id, None)
        if experiment is None:
            return None
        experiment.completed_at = datetime.now()
        experiment.duration_seconds = (
            experiment.completed_at - experiment.started_at
        ).total_seconds()
        if metrics:
            experiment.metrics.update(metrics)
            experiment.best_performance = max(
                [experiment.best_performance, *metrics.values()]
            )
        experiment.status = "completed" if success else "failed"
        self.completed_experiments.append(experiment)
        if self.kernel is not None:
            self.kernel.log_event(
                "experiment.completed",
                experiment_id=experiment_id,
                success=success,
                duration_seconds=experiment.duration_seconds,
            )
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        if experiment_id in self.active_experiments:
            return self.active_experiments[experiment_id]
        for experiment in self.completed_experiments:
            if experiment.experiment_id == experiment_id:
                return experiment
        return None
