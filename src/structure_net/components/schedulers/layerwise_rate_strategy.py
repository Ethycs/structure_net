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
)
import logging

class LayerwiseRateStrategy(BaseStrategy):
    """Strategy for adapting learning rates per layer based on their position."""

    def __init__(self, base_lr: float = 0.01, early_rate: float = 0.02, middle_rate: float = 0.01, late_rate: float = 0.005, name: str = None):
        super().__init__(name or "LayerwiseRateStrategy", strategy_type="learning_rate")
        self.base_lr = base_lr
        self.early_rate = early_rate
        self.middle_rate = middle_rate
        self.late_rate = late_rate
        self._required_analysis = {"analyzers.network_stats"}

    @property
    def contract(self) -> ComponentContract:
        """Declares the contract for this component."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(2, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs=self._required_analysis,
            provided_outputs={"plans.learning_rate"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False
            ),
        )

    def _create_plan(self, report: AnalysisReport, context: EvolutionContext) -> EvolutionPlan:
        """Create a learning rate adaptation plan based on layer position."""
        network_stats = report.get("analyzers.NetworkStatsAnalyzer", {})
        if not network_stats:
            self.log(logging.WARNING, "Network stats not found in report. Cannot create plan.")
            return EvolutionPlan()

        architecture = network_stats.get("architecture", [])
        total_layers = len(architecture)
        if total_layers == 0:
            return EvolutionPlan()

        lr_adjustments = {}
        for i in range(total_layers):
            if i < total_layers // 3:
                rate = self.early_rate
            elif i > 2 * total_layers // 3:
                rate = self.late_rate
            else:
                rate = self.middle_rate
            
            lr_adjustments[f"layer_{i}"] = {
                "new_lr": rate,
                "reason": f"Layer position {i+1}/{total_layers}"
            }

        plan = EvolutionPlan({
            "learning_rate_adjustments": lr_adjustments,
            "strategy_type": "layerwise_positional",
            "base_lr": self.base_lr,
            "epoch": context.epoch,
        })
        plan.priority = 0.7
        plan.created_by = self.name
        
        self.log(logging.INFO, f"Created layer-wise LR plan for {total_layers} layers.")
        return plan
