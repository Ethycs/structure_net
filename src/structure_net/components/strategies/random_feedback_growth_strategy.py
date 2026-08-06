"""Random-placement control strategy for delayed TinyLLM feedback growth."""

from __future__ import annotations

from typing import Optional

from ...core import (
    AnalysisReport,
    BaseStrategy,
    ComponentContract,
    ComponentVersion,
    EvolutionContext,
    EvolutionPlan,
    Maturity,
    ResourceLevel,
    ResourceRequirements,
)


class RandomFeedbackGrowthStrategy(BaseStrategy):
    """Propose reproducible random backward edges as an experiment control."""

    def __init__(
        self,
        *,
        count: int = 1,
        num_neurons: int = 16,
        connection_density: float = 0.05,
        gate_init: float = 0.0,
        refinement_steps: int = 1,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name or "RandomFeedbackGrowthStrategy", strategy_type="structural")
        if count <= 0 or num_neurons <= 0:
            raise ValueError("count and num_neurons must be positive")
        if not 0.0 < connection_density <= 1.0:
            raise ValueError("connection_density must be in (0, 1]")
        if refinement_steps < 1:
            raise ValueError("refinement_steps must be at least one")
        self.count = count
        self.num_neurons = num_neurons
        self.connection_density = connection_density
        self.gate_init = gate_init
        self.refinement_steps = refinement_steps
        self.seed = seed
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs=set(),
            optional_inputs={"experiment.seed"},
            provided_outputs={"plans.feedback_growth"},
            resources=ResourceRequirements(memory_level=ResourceLevel.LOW),
        )

    @property
    def contract(self) -> ComponentContract:
        return self._contract

    def _create_plan(
        self, report: AnalysisReport, context: EvolutionContext
    ) -> EvolutionPlan:
        seed = self.seed if self.seed is not None else int(context.get("seed", 0))
        plan = EvolutionPlan(
            {
                "type": "add_random_feedback_connections",
                "placement": "random_backward",
                "count": self.count,
                "num_neurons": self.num_neurons,
                "connection_density": self.connection_density,
                "gate_init": self.gate_init,
                "refinement_steps": self.refinement_steps,
                "seed": seed,
                "reason": "random-placement feedback control",
            }
        )
        plan.priority = 0.5
        plan.estimated_impact = 0.0
        return plan


__all__ = ["RandomFeedbackGrowthStrategy"]
