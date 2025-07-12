from src.structure_net.core.base_components import BaseStrategy
from src.structure_net.core.interfaces import (
    ComponentContract,
    ComponentVersion,
    Maturity,
    ResourceRequirements,
    ResourceLevel,
    EvolutionPlan,
    AnalysisReport,
    EvolutionContext,
    IComponent,
)
from typing import List
import logging

class HybridGrowthStrategy(BaseStrategy):
    """A strategy that combines multiple sub-strategies, choosing the best plan."""

    def __init__(self, strategies: List[IComponent], name: str = None):
        super().__init__(name or "HybridGrowthStrategy", strategy_type="hybrid")
        self.strategies = strategies
        self._required_analysis = set()
        for strategy in self.strategies:
            if hasattr(strategy, 'contract'):
                self._required_analysis.update(strategy.contract.required_inputs)

    @property
    def contract(self) -> ComponentContract:
        """Declares the contract for this component."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(2, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs=self._required_analysis,
            provided_outputs={"plans.structural.hybrid"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False
            ),
        )

    def _create_plan(self, report: AnalysisReport, context: EvolutionContext) -> EvolutionPlan:
        """Create a growth plan by selecting the best plan from all sub-strategies."""
        best_plan = EvolutionPlan()
        best_plan.priority = 0.0

        for strategy in self.strategies:
            if all(req in report for req in strategy.contract.required_inputs):
                try:
                    plan = strategy._create_plan(report, context)
                    if plan and plan.priority > best_plan.priority:
                        best_plan = plan
                        best_plan.created_by = f"{self.name} -> {strategy.name}"
                except Exception as e:
                    self.log(logging.ERROR, f"Sub-strategy {strategy.name} failed: {e}")

        if best_plan.priority > 0:
            self.log(logging.INFO, f"Selected plan from {best_plan.created_by} with priority {best_plan.priority:.2f}")
        else:
            self.log(logging.INFO, "No suitable sub-strategy plan found.")
            
        return best_plan
