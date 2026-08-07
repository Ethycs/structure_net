"""
Component scorecards.

A scorecard is the kernel's running health and usage record for one component
instance: executions, failures, timing, last error, and experiment usage. The
:class:`ScorecardManager` owns all scorecards and derives a coarse health
grade from the observed success rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ComponentHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"


@dataclass
class ComponentScorecard:
    """Complete scorecard for a component."""

    # Identity
    component_name: str
    component_type: str
    version: str
    maturity: str

    # Health
    health: ComponentHealth = ComponentHealth.HEALTHY
    health_score: float = 100.0

    # Performance
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0

    # Messages
    last_message: Optional[str] = None
    last_error: Optional[str] = None

    # Experiment usage
    experiment_count: int = 0
    last_experiment_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 1.0
        return self.successful_executions / self.total_executions


# Success-rate floors for each health grade, checked in order.
_HEALTH_LADDER = (
    (0.95, ComponentHealth.HEALTHY),
    (0.80, ComponentHealth.DEGRADED),
    (0.50, ComponentHealth.FAILING),
)


class ScorecardManager:
    """Manages scorecards for all components created through the kernel."""

    def __init__(self, kernel: Any = None):
        self.kernel = kernel
        self.scorecards: Dict[str, ComponentScorecard] = {}

    def create_scorecard(self, component: Any) -> ComponentScorecard:
        """Create (or return the existing) scorecard for a component."""
        if component.name in self.scorecards:
            return self.scorecards[component.name]
        contract = component.contract
        scorecard = ComponentScorecard(
            component_name=component.name,
            component_type=type(component).__name__,
            version=str(contract.version),
            maturity=contract.maturity.value,
        )
        self.scorecards[component.name] = scorecard
        return scorecard

    def get_scorecard(self, component_name: str) -> Optional[ComponentScorecard]:
        return self.scorecards.get(component_name)

    def update_execution(
        self,
        component_name: str,
        execution_time: float,
        success: bool,
        message: Optional[str] = None,
    ) -> None:
        """Record one execution and refresh the derived health grade."""
        scorecard = self.scorecards.get(component_name)
        if scorecard is None:
            return
        scorecard.total_executions += 1
        if success:
            scorecard.successful_executions += 1
            if message is not None:
                scorecard.last_message = message
        else:
            scorecard.failed_executions += 1
            if message is not None:
                scorecard.last_error = message
        scorecard.average_execution_time = (
            scorecard.average_execution_time * (scorecard.total_executions - 1)
            + execution_time
        ) / scorecard.total_executions
        scorecard.health_score = scorecard.success_rate * 100.0
        scorecard.health = self._grade(scorecard.success_rate)
        scorecard.last_updated = datetime.now()

    def record_experiment_use(self, component_name: str, experiment_id: str) -> None:
        scorecard = self.scorecards.get(component_name)
        if scorecard is None:
            return
        scorecard.experiment_count += 1
        scorecard.last_experiment_id = experiment_id
        scorecard.last_updated = datetime.now()

    @staticmethod
    def _grade(success_rate: float) -> ComponentHealth:
        for floor, grade in _HEALTH_LADDER:
            if success_rate >= floor:
                return grade
        return ComponentHealth.CRITICAL

    def summary(self) -> Dict[str, List[str]]:
        """Component names grouped by current health grade."""
        output: Dict[str, List[str]] = {grade.value: [] for grade in ComponentHealth}
        for name, scorecard in sorted(self.scorecards.items()):
            output[scorecard.health.value].append(name)
        return output
