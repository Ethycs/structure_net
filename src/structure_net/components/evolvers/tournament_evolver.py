from src.structure_net.core.base_components import BaseEvolver
from src.structure_net.core.interfaces import (
    ComponentContract,
    ComponentVersion,
    Maturity,
    ResourceRequirements,
    ResourceLevel,
    EvolutionPlan,
    IModel,
    ITrainer,
)
from typing import List, Dict, Any
import numpy as np
import logging

class TournamentEvolver(BaseEvolver):
    """An evolver that processes tournament results and creates the next generation."""

    def __init__(self, tournament_size: int, mutation_rate: float, name: str = None):
        super().__init__(name or "TournamentEvolver")
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self._supported_plan_types = {"evaluate_population"}

    @property
    def contract(self) -> ComponentContract:
        """Declares the contract for this component."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"plans.tournament"},
            provided_outputs={"evolution.new_population"},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False
            ),
        )

    def _execute_plan(self, plan: EvolutionPlan, model: IModel, trainer: ITrainer, optimizer: Any) -> Dict[str, Any]:
        """Evolves the population based on the results of the last generation."""
        
        results = plan.get("results", [])
        current_population = plan.get("population", [])

        results_map = {res.metrics['competitor_id']: res for res in results if 'competitor_id' in res.metrics}
        
        for competitor in current_population:
            res = results_map.get(competitor['id'])
            if res and not res.error:
                competitor['fitness'] = res.metrics.get('fitness', 0.0)
            else:
                competitor['fitness'] = 0.0

        current_population.sort(key=lambda x: x['fitness'], reverse=True)
        
        next_gen = current_population[:int(self.tournament_size * 0.2)] # Elitism
        
        while len(next_gen) < self.tournament_size:
            p1, p2 = np.random.choice(current_population, 2, replace=False)
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            child['id'] = f"gen{plan.get('generation', 0) + 1}_{len(next_gen)}"
            next_gen.append(child)
            
        self.log(logging.INFO, f"Evolved population to generation {plan.get('generation', 0) + 1} with {len(next_gen)} competitors.")
        return {"new_population": next_gen}

    def _crossover(self, p1, p2):
        arch_len = (len(p1['architecture']) + len(p2['architecture'])) // 2
        new_arch = [p1['architecture'][0]] + [np.mean([p1['architecture'][i], p2['architecture'][i]], dtype=int) for i in range(1, min(len(p1['architecture']), len(p2['architecture']))-1)][:arch_len-2] + [p1['architecture'][-1]]
        return {'architecture': new_arch, 'sparsity': np.mean([p1['sparsity'], p2['sparsity']]), 'lr_strategy': p1['lr_strategy'], 'fitness': 0.0, 'seed_path': None}

    def _mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            individual['sparsity'] = np.clip(individual['sparsity'] * np.random.uniform(0.8, 1.2), 0.01, 0.3)
        return individual