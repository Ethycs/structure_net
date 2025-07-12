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

class InformationFlowGrowthStrategy(BaseStrategy):
    """Growth strategy based on information flow analysis."""

    def __init__(self, bottleneck_threshold: float = 0.1, efficiency_threshold: float = 0.7, name: str = None):
        super().__init__(name or "InformationFlowGrowthStrategy", strategy_type="structural")
        self.bottleneck_threshold = bottleneck_threshold
        self.efficiency_threshold = efficiency_threshold
        self._required_analysis = {"analyzers.information_flow"}

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
        """Create a growth plan based on information flow analysis."""
        info_flow_report = report.get("analyzers.information_flow", {})
        if not info_flow_report:
            self.log(logging.WARNING, "Information flow report not found. Cannot create plan.")
            return EvolutionPlan()

        bottlenecks = info_flow_report.get("bottlenecks", [])
        efficiency = info_flow_report.get("information_efficiency", 1.0)

        if bottlenecks and bottlenecks[0].get("severity", 0) > self.bottleneck_threshold:
            plan = self._create_add_layer_plan(bottlenecks[0], context)
        elif efficiency < self.efficiency_threshold:
            plan = self._create_add_skip_connection_plan(efficiency)
        else:
            plan = EvolutionPlan()

        plan.created_by = self.name
        return plan

    def _create_add_layer_plan(self, bottleneck: dict, context: EvolutionContext) -> EvolutionPlan:
        """Creates a plan to add a new layer to address a bottleneck."""
        position = bottleneck.get("position", 1)
        info_loss = bottleneck.get("info_loss", 0.1)
        base_size = 64
        size_multiplier = min(4.0, max(1.0, info_loss * 10))
        size = int(base_size * size_multiplier)

        plan = EvolutionPlan({
            "action_type": ActionType.ADD_LAYER,
            "position": position,
            "size": size,
            "reason": f"Information bottleneck at position {position} with severity {bottleneck.get('severity', 0):.3f}",
        })
        plan.priority = 0.9
        plan.estimated_impact = 0.2
        self.log(logging.INFO, f"Proposing to add layer at position {position} to fix bottleneck.")
        return plan

    def _create_add_skip_connection_plan(self, efficiency: float) -> EvolutionPlan:
        """Creates a plan to add skip connections."""
        plan = EvolutionPlan({
            "action_type": ActionType.ADD_SKIP_CONNECTION,
            "reason": f"Low information efficiency: {efficiency:.3f}",
        })
        plan.priority = 0.5
        plan.estimated_impact = 0.1
        self.log(logging.INFO, "Proposing to add skip connections to improve information efficiency.")
        return plan
