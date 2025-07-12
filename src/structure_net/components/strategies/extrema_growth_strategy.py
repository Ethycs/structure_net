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

class ExtremaGrowthStrategy(BaseStrategy):
    """Growth strategy based on extrema analysis."""

    def __init__(self, extrema_threshold: float = 0.3, dead_neuron_threshold: int = 5, saturated_neuron_threshold: int = 5, patch_size: int = 3, name: str = None):
        super().__init__(name or "ExtremaGrowthStrategy", strategy_type="structural")
        self.extrema_threshold = extrema_threshold
        self.dead_neuron_threshold = dead_neuron_threshold
        self.saturated_neuron_threshold = saturated_neuron_threshold
        self.patch_size = patch_size
        self._required_analysis = {"analyzers.extrema_report"}

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
        """Create a growth plan based on extrema analysis."""
        extrema_report = report.get("analyzers.extrema_report", {})
        if not extrema_report:
            self.log(logging.WARNING, "Extrema report not found. Cannot create plan.")
            return EvolutionPlan()

        total_dead = extrema_report.get("total_dead_neurons", 0)
        total_saturated = extrema_report.get("total_saturated_neurons", 0)
        extrema_ratio = extrema_report.get("extrema_ratio", 0.0)

        if extrema_ratio > self.extrema_threshold:
            plan = self._create_add_layer_plan(extrema_ratio, context)
        elif total_dead >= self.dead_neuron_threshold or total_saturated >= self.saturated_neuron_threshold:
            plan = self._create_add_patches_plan(total_dead, total_saturated)
        else:
            plan = EvolutionPlan()

        plan.created_by = self.name
        return plan

    def _create_add_layer_plan(self, extrema_ratio: float, context: EvolutionContext) -> EvolutionPlan:
        """Creates a plan to add a new layer."""
        network_stats = context.get("analyzers.network_stats", {})
        architecture = network_stats.get("architecture", [])
        position = len(architecture) // 2 if architecture else 0
        size = max(32, min(256, architecture[position])) if architecture else 64
        
        plan = EvolutionPlan({
            "action_type": ActionType.ADD_LAYER,
            "position": position,
            "size": size,
            "reason": f"High extrema ratio: {extrema_ratio:.2f}",
        })
        plan.priority = 0.8
        plan.estimated_impact = 0.15
        self.log(logging.INFO, f"Proposing to add layer at position {position} due to high extrema ratio.")
        return plan

    def _create_add_patches_plan(self, total_dead: int, total_saturated: int) -> EvolutionPlan:
        """Creates a plan to add patches to the network."""
        plan = EvolutionPlan({
            "action_type": ActionType.ADD_PATCHES,
            "size": self.patch_size,
            "reason": f"Dead neurons: {total_dead}, Saturated: {total_saturated}",
        })
        plan.priority = 0.6
        plan.estimated_impact = 0.05
        self.log(logging.INFO, f"Proposing to add patches to address {total_dead} dead and {total_saturated} saturated neurons.")
        return plan
