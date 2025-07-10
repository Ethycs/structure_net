#!/usr/bin/env python3
"""
Ultimate Structure Net Stress Test v2 - NAL-Powered Edition

This script uses the Neural Architecture Lab (NAL) framework to run a
tournament-style evolution stress test. It demonstrates how to use the NAL
for complex, multi-generational experiments.
"""

import os
import sys
import json
import time
import asyncio
import traceback
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NAL Imports
from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    Hypothesis,
    HypothesisCategory,
    Experiment,
    ExperimentResult
)
from src.neural_architecture_lab.core import LabConfigFactory

# structure_net Imports
from src.structure_net.core.network_factory import create_standard_network
from src.structure_net.core.io_operations import load_model_seed
from src.neural_architecture_lab.data_factory_integration import NALChromaIntegration
from src.data_factory import create_dataset, get_dataset_config
from src.neural_architecture_lab.data_factory_integration import create_memory_efficient_nal

@dataclass
class StressTestConfig:
    """Configuration for the ultimate stress test."""
    tournament_size: int = 32
    generations: int = 5
    mutation_rate: float = 0.3
    seed_model_dir: Optional[str] = None
    epochs_per_generation: int = 10
    batch_size_base: int = 128
    learning_rate_strategies: List[str] = field(default_factory=lambda: ['basic', 'advanced'])
    enable_growth: bool = True
    max_layers: int = 15
    dataset_name: str = 'cifar10'

def evaluate_competitor_task(experiment: Experiment, device_id: int) -> ExperimentResult:
    """
    NAL worker function for evaluating a single tournament competitor.
    This function is designed to be run in a separate process.
    """
    config = experiment.parameters
    device = f'cuda:{device_id}' if torch.cuda.is_available() and device_id >= 0 else 'cpu'
    start_time = time.time()

    try:
        dataset_name = config.get('dataset', 'cifar10')
        dataset = create_dataset(dataset_name, batch_size=config['batch_size'])
        train_loader = dataset['train_loader']
        test_loader = dataset['test_loader']

        if 'seed_path' in config and config['seed_path']:
            model, _ = load_model_seed(config['seed_path'], device=device)
        else:
            model = create_standard_network(
                architecture=config['architecture'],
                sparsity=config.get('sparsity', 0.02),
                device=device
            )
        
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for _ in range(config['epochs']):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                if data.dim() > 2 and hasattr(model, 'layers') and model.layers and isinstance(model.layers[0], nn.Linear):
                    data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Evaluation loop
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if data.dim() > 2 and hasattr(model, 'layers') and model.layers and isinstance(model.layers[0], nn.Linear):
                    data = data.view(data.size(0), -1)
                
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total if total > 0 else 0
        parameters = sum(p.numel() for p in model.parameters())
        
        metrics = {
            'accuracy': accuracy,
            'parameters': parameters,
            'fitness': (accuracy / (parameters / 1e6)) if parameters > 0 else 0,
            'competitor_id': config.get('competitor_id')
        }
        
        return ExperimentResult(
            experiment_id=experiment.id,
            hypothesis_id=experiment.hypothesis_id,
            metrics=metrics,
            primary_metric=metrics['fitness'],
            model_architecture=config['architecture'],
            model_parameters=parameters,
            training_time=time.time() - start_time
        )

    except Exception as e:
        return ExperimentResult(
            experiment_id=experiment.id,
            hypothesis_id=experiment.hypothesis_id,
            metrics={},
            primary_metric=0.0,
            model_architecture=config.get('architecture', []),
            model_parameters=0,
            training_time=time.time() - start_time,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )

class TournamentExecutor:
    """Executes tournament-style evolution using NAL."""
    
    def __init__(self, config: StressTestConfig, lab: NeuralArchitectureLab):
        self.config = config
        self.lab = lab
        self.logger = lab.logger
        self.population = []
        self.generation_results = []

    def create_hypothesis(self, generation: int) -> Hypothesis:
        """Creates the hypothesis for the current generation."""
        # Create a list of parameter dictionaries, one for each competitor
        param_list = []
        for c in self.population:
            param_list.append({
                'architecture': c['architecture'],
                'sparsity': c['sparsity'],
                'lr_strategy': c['lr_strategy'],
                'competitor_id': c['id'],
                'seed_path': c.get('seed_path')
            })

        return Hypothesis(
            id=f"tournament_gen_{generation}",
            name=f"Tournament Generation {generation}",
            description="Evaluate a generation of tournament competitors.",
            question="Which architectures perform best?",
            prediction="Fitter architectures will emerge.",
            test_function=evaluate_competitor_task,
            parameter_space={'params': param_list}, # Pass list of configs
            control_parameters={
                'dataset': self.config.dataset_name,
                'epochs': self.config.epochs_per_generation,
                'batch_size': self.config.batch_size_base,
                'enable_growth': self.config.enable_growth,
            },
            success_metrics={'fitness': 0.0},
            category=HypothesisCategory.ARCHITECTURE
        )

    def generate_initial_population(self):
        """Generates the initial population, using seeds if available."""
        dataset_config = get_dataset_config(self.config.dataset_name)
        
        if self.config.seed_model_dir and Path(self.config.seed_model_dir).exists():
            for seed_file in Path(self.config.seed_model_dir).glob("*.pt"):
                if len(self.population) >= self.config.tournament_size:
                    break
                try:
                    _, checkpoint_data = load_model_seed(str(seed_file), device='cpu')
                    self.population.append({
                        'id': f'seed_{len(self.population)}',
                        'architecture': checkpoint_data['architecture'],
                        'sparsity': checkpoint_data.get('metrics', {}).get('sparsity', 0.02),
                        'lr_strategy': np.random.choice(self.config.learning_rate_strategies),
                        'fitness': 0.0,
                        'seed_path': str(seed_file)
                    })
                except Exception as e:
                    print(f"Warning: Could not load seed model {seed_file}. Error: {e}")

        while len(self.population) < self.config.tournament_size:
            n_layers = np.random.randint(3, 7)
            architecture = [dataset_config.input_size] + [max(32, int(512 * np.random.uniform(0.5, 1.0))) for _ in range(n_layers - 1)] + [dataset_config.num_classes]
            self.population.append({
                'id': f'random_{len(self.population)}',
                'architecture': architecture,
                'sparsity': np.random.uniform(0.01, 0.1),
                'lr_strategy': np.random.choice(self.config.learning_rate_strategies),
                'fitness': 0.0,
                'seed_path': None
            })

    async def run_tournament(self):
        """Runs the full evolutionary tournament."""
        self.generate_initial_population()
        
        for generation in range(self.config.generations):
            print(f"\n--- Generation {generation}/{self.config.generations} ---")
            hypothesis = self.create_hypothesis(generation)
            self.lab.register_hypothesis(hypothesis)
            hypothesis_result = await self.lab.test_hypothesis(hypothesis.id)
            
            self.process_generation_results(hypothesis_result)
            self.evolve_population()

        return self.generation_results

    def process_generation_results(self, result):
        """Updates the population with the results from the NAL."""
        results_map = {res.metrics['competitor_id']: res for res in result.experiment_results if 'competitor_id' in res.metrics}
        
        for competitor in self.population:
            res = results_map.get(competitor['id'])
            if res and not res.error:
                competitor['fitness'] = res.metrics.get('fitness', 0.0)
                competitor['accuracy'] = res.metrics.get('accuracy', 0.0)
                competitor['parameters'] = res.model_parameters
            else:
                competitor['fitness'] = 0.0

        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        self.generation_results.append(self.population)

    def evolve_population(self):
        """Creates the next generation of the population."""
        next_gen = self.population[:int(self.config.tournament_size * 0.2)] # Elitism
        
        while len(next_gen) < self.config.tournament_size:
            p1, p2 = np.random.choice(self.population, 2, replace=False)
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            child['id'] = f"gen{len(self.generation_results)}_{len(next_gen)}"
            next_gen.append(child)
            
        self.population = next_gen

    def _crossover(self, p1, p2):
        # Simplified crossover
        arch_len = (len(p1['architecture']) + len(p2['architecture'])) // 2
        new_arch = [p1['architecture'][0]] + [np.mean([p1['architecture'][i], p2['architecture'][i]], dtype=int) for i in range(1, min(len(p1['architecture']), len(p2['architecture']))-1)][:arch_len-2] + [p1['architecture'][-1]]
        return {'architecture': new_arch, 'sparsity': np.mean([p1['sparsity'], p2['sparsity']]), 'lr_strategy': p1['lr_strategy'], 'fitness': 0.0, 'seed_path': None}

    def _mutate(self, individual):
        if np.random.rand() < self.config.mutation_rate:
            individual['sparsity'] = np.clip(individual['sparsity'] * np.random.uniform(0.8, 1.2), 0.01, 0.3)
        return individual

@dataclass
class StressTestConfig:
    """Configuration for the ultimate stress test."""
    tournament_size: int = 32
    generations: int = 5
    mutation_rate: float = 0.3
    seed_model_dir: Optional[str] = None
    epochs_per_generation: int = 10
    batch_size_base: int = 128
    learning_rate_strategies: List[str] = field(default_factory=lambda: ['basic', 'advanced'])
    enable_growth: bool = True
    max_layers: int = 15
    dataset_name: str = 'cifar10'

def get_default_stress_test_config() -> StressTestConfig:
    """Returns a default configuration for the stress test."""
    return StressTestConfig()

async def run_stress_test(lab_config: LabConfig, stress_test_config: StressTestConfig):
    """
    Runs the tournament stress test with the given configurations.

    Args:
        lab_config: Configuration for the Neural Architecture Lab.
        stress_test_config: Configuration for the stress test tournament.
    """
    lab = NeuralArchitectureLab(lab_config)
    executor = TournamentExecutor(stress_test_config, lab)
    
    results = await executor.run_tournament()
    
    print("\n--- Tournament Complete ---")
    for i, gen_pop in enumerate(results):
        if gen_pop:
            best = gen_pop[0]
            print(f"Generation {i}: Best Fitness={best.get('fitness', 0.0):.4f}, Best Acc={best.get('accuracy', 0.0):.2%}")
        else:
            print(f"Generation {i}: No results.")

def get_default_lab_config() -> LabConfig:
    """Returns a default configuration for the NAL lab."""
    return LabConfig(
        project_name="ultimate_stress_test",
        results_dir=f"data/nal_stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        max_parallel_experiments=4,
        log_level="INFO",
        enable_wandb=False,
        device_ids=[0]
    )

def main():
    """Main function for the NAL-powered stress test."""
    # 1. Define script-specific defaults
    stress_config = get_default_stress_test_config()
    lab_config = get_default_lab_config()

    # 2. Create parser and add script-specific arguments
    parser = argparse.ArgumentParser(description="NAL-Powered Ultimate Stress Test")
    parser.add_argument('--generations', type=int, help=f"Number of generations to evolve (default: {stress_config.generations}).")
    parser.add_argument('--tournament-size', type=int, help=f"Number of competitors per generation (default: {stress_config.tournament_size}).")
    parser.add_argument('--dataset', type=str, help=f"Dataset to use (default: {stress_config.dataset_name}).")
    parser.add_argument('--seed-model-dir', type=str, help="Directory of seed models to start the tournament.")

    # 3. Add standard NAL arguments to the parser
    LabConfigFactory.add_arguments(parser)
    
    # 4. Parse all arguments
    args = parser.parse_args()

    # 5. Create the final LabConfig by overriding defaults with provided args
    lab_config = LabConfigFactory.from_args(args, base_config=lab_config)

    # 6. Override stress_config defaults with provided args
    if args.generations is not None:
        stress_config.generations = args.generations
    if args.tournament_size is not None:
        stress_config.tournament_size = args.tournament_size
    if args.dataset is not None:
        stress_config.dataset_name = args.dataset
    if args.seed_model_dir is not None:
        stress_config.seed_model_dir = args.seed_model_dir
    
    # 7. Run the experiment
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_stress_test(lab_config, stress_config))

if __name__ == "__main__":
    main()