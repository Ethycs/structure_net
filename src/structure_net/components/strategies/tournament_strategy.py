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
from typing import List
import logging

class TournamentStrategy(BaseStrategy):
    """A strategy for generating a tournament-style evolution plan."""

    def __init__(self, population: List[dict], name: str = None):
        super().__init__(name or "TournamentStrategy", strategy_type="tournament")
        self.population = population
        self._required_analysis = set()

    @property
    def contract(self) -> ComponentContract:
        """Declares the contract for this component."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            provided_outputs={"plans.tournament"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False
            ),
        )

    def _create_plan(self, report: AnalysisReport, context: EvolutionContext) -> EvolutionPlan:
        """Create a plan to evaluate the current population."""
        param_list = []
        for competitor in self.population:
            param_list.append({
                'architecture': competitor['architecture'],
                'sparsity': competitor['sparsity'],
                'lr_strategy': competitor['lr_strategy'],
                'competitor_id': competitor['id'],
                'seed_path': competitor.get('seed_path')
            })

        plan = EvolutionPlan({
            "action_type": "evaluate_population",
            "competitors": param_list,
            "reason": f"Evaluating generation {context.get('generation', 0)}",
        })
        plan.priority = 1.0
        plan.created_by = self.name
        
        self.log(logging.INFO, f"Created plan to evaluate {len(self.population)} competitors.")
        return plan