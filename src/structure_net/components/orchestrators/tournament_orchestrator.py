from src.structure_net.core.base_components import BaseOrchestrator
from src.structure_net.core.interfaces import (
    EvolutionContext,
    ComponentContract,
    ComponentVersion,
    Maturity,
    ResourceRequirements,
    ResourceLevel,
    EvolutionPlan,
)
from src.structure_net.components.strategies.tournament_strategy import TournamentStrategy
from src.structure_net.components.evolvers.tournament_evolver import TournamentEvolver
from src.neural_architecture_lab import NeuralArchitectureLab, LabConfig, Hypothesis, HypothesisCategory
from src.data_factory import get_dataset_config
from typing import List, Dict, Any
import logging
import numpy as np

class TournamentOrchestrator(BaseOrchestrator):
    """Orchestrates a tournament-style evolution experiment."""

    def __init__(self, lab_config: LabConfig, stress_test_config, name: str = None):
        super().__init__(name or "TournamentOrchestrator")
        self.lab_config = lab_config
        self.stress_test_config = stress_test_config
        self.population = []
        self.generation_results = []

    @property
    def contract(self) -> ComponentContract:
        """Declares the contract for this component."""
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            resources=ResourceRequirements(
                memory_level=ResourceLevel.HIGH,
                requires_gpu=True
            ),
        )

    async def run_tournament(self):
        """Runs the full evolutionary tournament."""
        self._generate_initial_population()
        
        for generation in range(self.stress_test_config.generations):
            self.log(logging.INFO, f"Starting generation {generation + 1}/{self.stress_test_config.generations}")
            
            # 1. Create Strategy and Propose Plan
            strategy = TournamentStrategy(self.population)
            context = EvolutionContext(generation=generation)
            plan = strategy.propose_plan(None, context)

            # 2. Create Hypothesis from Plan
            hypothesis = self._create_hypothesis_from_plan(plan, generation)

            # 3. Run experiments using NAL
            lab = NeuralArchitectureLab(self.lab_config)
            lab.register_hypothesis(hypothesis)
            results = await lab.test_hypothesis(hypothesis.id)

            # 4. Evolve population
            evolver = TournamentEvolver(self.stress_test_config.tournament_size, self.stress_test_config.mutation_rate)
            evolution_plan = EvolutionPlan(results=results.experiment_results, population=self.population, generation=generation)
            evolution_result = evolver.apply_plan(evolution_plan, None, None, None)
            self.population = evolution_result["new_population"]
            
            self.generation_results.append(self.population)

        return self.generation_results

    def _generate_initial_population(self):
        """Generates the initial population."""
        dataset_config = get_dataset_config(self.stress_test_config.dataset_name)
        
        # This logic is simplified from the original script for brevity
        while len(self.population) < self.stress_test_config.tournament_size:
            n_layers = np.random.randint(3, 7)
            architecture = [dataset_config.input_size] + [max(32, int(512 * np.random.uniform(0.5, 1.0))) for _ in range(n_layers - 1)] + [dataset_config.num_classes]
            self.population.append({
                'id': f'random_{len(self.population)}',
                'architecture': architecture,
                'sparsity': np.random.uniform(0.01, 0.1),
                'lr_strategy': np.random.choice(self.stress_test_config.learning_rate_strategies),
                'fitness': 0.0,
                'seed_path': None
            })

    def _create_hypothesis_from_plan(self, plan: EvolutionPlan, generation: int) -> Hypothesis:
        """Creates a NAL Hypothesis from an EvolutionPlan."""
        from src.neural_architecture_lab.workers.tournament_worker import evaluate_competitor_task

        return Hypothesis(
            id=f"tournament_gen_{generation}",
            name=f"Tournament Generation {generation}",
            description="Evaluate a generation of tournament competitors.",
            question="Which architectures perform best?",
            prediction="Fitter architectures will emerge.",
            test_function=evaluate_competitor_task,
            parameter_space={'params': plan.get("competitors", [])},
            control_parameters={
                'dataset': self.stress_test_config.dataset_name,
                'epochs': self.stress_test_config.epochs_per_generation,
                'batch_size': self.stress_test_config.batch_size_base,
            },
            success_metrics={'fitness': 0.0},
            category=HypothesisCategory.ARCHITECTURE
        )