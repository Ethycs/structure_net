"""
Component health monitoring.

Derives per-component and system-level health from the scorecards the kernel
already maintains. No background threads: callers (or event subscribers)
invoke checks explicitly, which keeps behavior deterministic under test.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .scorecard import ComponentHealth


class ComponentHealthMonitor:
    """Reads scorecards and reports component and system health."""

    def __init__(self, kernel: Any):
        self.kernel = kernel

    def check_component(self, component_name: str) -> ComponentHealth:
        scorecard = self.kernel.scorecard_manager.get_scorecard(component_name)
        if scorecard is None:
            raise KeyError(f"no scorecard for component '{component_name}'")
        return scorecard.health

    def check_all(self) -> Dict[str, ComponentHealth]:
        return {
            name: card.health
            for name, card in self.kernel.scorecard_manager.scorecards.items()
        }

    def unhealthy_components(self) -> List[str]:
        return [
            name
            for name, health in self.check_all().items()
            if health is not ComponentHealth.HEALTHY
        ]

    def report(self) -> Dict[str, Any]:
        """System-level health summary."""
        grouped = self.kernel.scorecard_manager.summary()
        total = sum(len(names) for names in grouped.values())
        return {
            "total_components": total,
            "by_health": grouped,
            "system_healthy": not any(
                grouped[grade.value]
                for grade in ComponentHealth
                if grade is not ComponentHealth.HEALTHY
            ),
        }
