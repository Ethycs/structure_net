from src.structure_net.core.base_components import BaseStrategyOrchestrator
from src.structure_net.core.interfaces import (
    EvolutionPlan,
    AnalysisReport,
    EvolutionContext,
    IComponent,
)
from typing import List
import logging

class HybridGrowthStrategy(BaseStrategyOrchestrator):
    """An orchestrator that selects the best plan from multiple growth strategies."""

    def select_best_plan(self, plans: List[EvolutionPlan]) -> EvolutionPlan:
        """Selects the plan with the highest priority."""
        if not plans:
            return EvolutionPlan()
        
        best_plan = max(plans, key=lambda p: p.priority)
        return best_plan
