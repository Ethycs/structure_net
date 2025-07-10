#!/usr/bin/env python3
"""
Ultimate Structure Net Stress Test v2 - NAL-Powered Edition

This version replicates the original ultimate_stress_test.py functionality
but uses the Neural Architecture Lab (NAL) framework internally. It provides
the same interface and tournament-style evolution while leveraging NAL's
scientific approach and comprehensive analysis.
"""

import os
import sys
import json
import time
import asyncio
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
from src.neural_architecture_lab.core import (
    LabConfig,
    Hypothesis,
    HypothesisCategory
)
from src.neural_architecture_lab.lab import NeuralArchitectureLab
from src.neural_architecture_lab.advanced_runners import AdvancedExperimentRunner

# structure_net Imports
from src.structure_net.core.network_factory import create_standard_network
from src.structure_net.core.io_operations import load_model_seed
from src.data_factory import create_dataset, get_dataset_config

@dataclass
class StressTestConfig:
    """Configuration for the ultimate stress test."""
    num_gpus: int = 1
    processes_per_gpu: int = 1
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
    enable_profiling: bool = False

def evaluate_competitor_task(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """NAL test function for evaluating a single tournament competitor."""
    device = config.get('device', 'cpu')
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(config['epochs']):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            if data.dim() > 2 and model.layers and isinstance(model.layers[0], nn.Linear):
                data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if data.dim() > 2 and model.layers and isinstance(model.layers[0], nn.Linear):
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
        'fitness': (accuracy / (parameters / 1e6)) if parameters > 0 else 0
    }
    
    return model, metrics

class TournamentExecutor:
    """Executes tournament-style evolution using NAL."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.population = []
        self.generation_results = []
        
        self.data_dir = Path(f"/data/stress_test_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.nal_config = LabConfig(
            max_parallel_experiments=config.num_gpus * config.processes_per_gpu,
            device_ids=list(range(config.num_gpus)),
            results_dir=str(self.data_dir / "nal_results"),
        )
        self.lab = NeuralArchitectureLab(self.nal_config)
        self.lab.runner = AdvancedExperimentRunner(self.nal_config)

    def create_hypothesis(self, generation: int) -> Hypothesis:
        """Creates the hypothesis for the current generation."""
        return Hypothesis(
            id=f"tournament_gen_{generation}",
            name=f"Tournament Generation {generation}",
            description="Evaluate a generation of tournament competitors.",
            question="Which architectures perform best?",
            prediction="Fitter architectures will emerge.",
            test_function=evaluate_competitor_task,
            parameter_space={
                'architecture': [c['architecture'] for c in self.population],
                'sparsity': [c['sparsity'] for c in self.population],
                'lr_strategy': [c['lr_strategy'] for c in self.population],
                'competitor_id': [c['id'] for c in self.population],
                'seed_path': [c.get('seed_path') for c in self.population]
            },
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
        results_map = {res.experiment.parameters['competitor_id']: res for res in result.experiment_results}
        
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

def main():
    parser = argparse.ArgumentParser(description="NAL-Powered Ultimate Stress Test")
    parser.add_argument('--seed-model-dir', type=str, help="Directory of seed models.")
    args = parser.parse_args()

    config = StressTestConfig(seed_model_dir=args.seed_model_dir)
    
    executor = TournamentExecutor(config)
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(executor.run_tournament())
    
    print("\n--- Tournament Complete ---")
    for i, gen_pop in enumerate(results):
        best = gen_pop[0]
        print(f"Generation {i}: Best Fitness={best['fitness']:.4f}, Best Acc={best['accuracy']:.2%}")

if __name__ == "__main__":
    main()
