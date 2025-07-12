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
    ActionType,
)
import logging

class ResidualBlockGrowthStrategy(BaseStrategy):
    """Growth strategy that adds residual blocks to the network."""

    def __init__(self, num_layers: int = 2, activation_threshold: float = 0.2, name: str = None):
        super().__init__(name or "ResidualBlockGrowthStrategy", strategy_type="structural")
        self.num_layers = num_layers
        self.activation_threshold = activation_threshold
        self._required_analysis = {"analyzers.network_stats"}

    @property
    def contract(self) -> ComponentContract:
        """Declares the contract for this component."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(2, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs=self._required_analysis,
            provided_outputs={"plans.structural"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.MEDIUM,
                requires_gpu=False
            ),
        )

    def _create_plan(self, report: AnalysisReport, context: EvolutionContext) -> EvolutionPlan:
        """Create a growth plan to add a residual block if the network is deep enough."""
        network_stats = report.get("analyzers.NetworkStatsAnalyzer", {})
        if not network_stats:
            self.log(logging.WARNING, "Network stats not found in report. Cannot create plan.")
            return EvolutionPlan()

        architecture = network_stats.get("architecture", [])
        network_depth = len(architecture)

        if network_depth >= 4:
            position = network_depth // 2
            plan = EvolutionPlan({
                "action_type": ActionType.ADD_RESIDUAL_BLOCK,
                "position": position,
                "layer_count": self.num_layers,
                "reason": f"Network depth is {network_depth}, suitable for residual block.",
            })
            plan.priority = 0.75
            plan.estimated_impact = 0.1
            self.log(logging.INFO, f"Proposing to add residual block at position {position}.")
            return plan
        
        return EvolutionPlan()
