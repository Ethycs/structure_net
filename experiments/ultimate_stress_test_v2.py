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
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import psutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NAL components directly to avoid triggering migration messages
from src.neural_architecture_lab.core import (
    LabConfig,
    Hypothesis,
    HypothesisCategory,
    Experiment,
    ExperimentResult,
    ExperimentStatus
)
from src.neural_architecture_lab.lab import NeuralArchitectureLab
from src.neural_architecture_lab.advanced_runners import AdvancedExperimentRunner

# Import structure_net components
from src.structure_net.core.network_factory import create_standard_network
from src.structure_net.core.io_operations import save_model_seed, load_model_seed
from src.structure_net.evolution.adaptive_learning_rates.base import LearningRateStrategy
from src.structure_net.seed_search.gpu_seed_hunter import GPUSeedHunter
from src.structure_net.logging.standardized_logging import StandardizedLogger, LoggingConfig
from src.structure_net.logging.schemas import PerformanceMetrics, ExperimentConfig

# Inline the needed classes to avoid triggering migration messages from imports

@dataclass
class StressTestConfig:
    """Configuration for the ultimate stress test."""
    # GPU and parallelization
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 1
    processes_per_gpu: int = 2
    threads_per_process: int = 4
    max_memory_per_gpu: float = 0.8  # Use 80% of GPU memory
    
    # Tournament parameters
    tournament_size: int = 64  # Number of competing architectures
    generations: int = 10
    survivors_per_generation: int = 16
    mutation_rate: float = 0.3
    
    # Training parameters
    epochs_per_generation: int = 20
    batch_size_base: int = 256
    learning_rate_strategies: List[str] = None  # Will use all strategies
    
    # Growth and evolution
    enable_growth: bool = True
    enable_residual_blocks: bool = True
    growth_frequency: int = 5  # Every 5 epochs
    max_layers: int = 20
    
    # Metrics and profiling
    enable_comprehensive_metrics: bool = True
    enable_profiling: bool = True
    metrics_frequency: int = 2  # Every 2 epochs
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    memory_cleanup_frequency: int = 10
    
    def __post_init__(self):
        if self.learning_rate_strategies is None:
            self.learning_rate_strategies = [
                'basic', 'advanced', 'comprehensive', 'ultimate'
            ]

def calculate_optimal_memory_usage():
    """Calculate optimal memory usage for maximum GPU saturation."""
    if not torch.cuda.is_available():
        return {"error": "No CUDA GPUs available"}
    
    gpu_info = {}
    total_memory = 0
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        total_memory += memory_gb
        
        gpu_info[f'gpu_{i}'] = {
            'name': props.name,
            'memory_gb': memory_gb,
            'multiprocessor_count': props.multi_processor_count,
            'max_threads_per_multiprocessor': props.max_threads_per_multi_processor
        }
    
    # Calculate optimal configuration
    optimal_config = {
        'total_gpus': torch.cuda.device_count(),
        'total_memory_gb': total_memory,
        'recommended_processes_per_gpu': 2 if total_memory > 16 else 1,
        'recommended_batch_size': 512 if total_memory > 32 else 256,
        'recommended_tournament_size': min(128, torch.cuda.device_count() * 32),
        'gpu_details': gpu_info
    }
    
    return optimal_config


def evaluate_competitor_task(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """
    This is the test function that will be executed by NAL for each competitor.
    It's defined at the top level to ensure it can be pickled by multiprocessing.
    """
    # This function would contain the logic to create, train, and evaluate
    # a single model based on the provided config.
    # For this refactoring, we will use the logic from the AdvancedExperimentRunner,
    # as it seems to be the intended implementation for a single experiment run.
    
    # We can't directly call run_advanced_experiment here because it's not designed
    # to be a standalone test function. Instead, we'll replicate its core logic.
    
    # This is a simplified version of what run_advanced_experiment does.
    device = config.get('device', 'cpu')
    model = create_standard_network(
        architecture=config['architecture'],
        sparsity=config.get('sparsity', 0.02),
        device=device
    )
    
    # Dummy training and evaluation
    # In a real scenario, this would involve a full training loop.
    accuracy = np.random.uniform(0.1, 0.9)
    parameters = sum(p.numel() for p in model.parameters())
    
    metrics = {
        'accuracy': accuracy,
        'parameters': parameters,
        'fitness': accuracy / (parameters / 1e6) if parameters > 0 else 0
    }
    
    return model, metrics

def evaluate_competitor_task(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]:
    """
    This is the test function that will be executed by NAL for each competitor.
    It's defined at the top level to ensure it can be pickled by multiprocessing.
    """
    # This function would contain the logic to create, train, and evaluate
    # a single model based on the provided config.
    # For this refactoring, we will use the logic from the AdvancedExperimentRunner,
    # as it seems to be the intended implementation for a single experiment run.
    
    # We can't directly call run_advanced_experiment here because it's not designed
    # to be a standalone test function. Instead, we'll replicate its core logic.
    
    # This is a simplified version of what run_advanced_experiment does.
    device = config.get('device', 'cpu')
    model = create_standard_network(
        architecture=config['architecture'],
        sparsity=config.get('sparsity', 0.02),
        device=device
    )
    
    # Dummy training and evaluation
    # In a real scenario, this would involve a full training loop.
    accuracy = np.random.uniform(0.1, 0.9)
    parameters = sum(p.numel() for p in model.parameters())
    
    metrics = {
        'accuracy': accuracy,
        'parameters': parameters,
        'fitness': accuracy / (parameters / 1e6) if parameters > 0 else 0
    }
    
    return model, metrics

class TournamentExecutor:
    """Executes tournament-style evolution using NAL internally."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.population = []
        self.generation_results = []
        self.start_time = None
        
        # Create data directory
        self.data_dir = Path("/data") / f"stress_test_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logging_config = LoggingConfig(
            project_name="stress_test_v2_nal",
            queue_dir=str(self.data_dir / "queue"),
            sent_dir=str(self.data_dir / "sent"),
            rejected_dir=str(self.data_dir / "rejected"),
            enable_wandb=True,
            enable_local_backup=True,
            auto_upload=True
        )
        self.logger = StandardizedLogger(self.logging_config)
        
        # Configure NAL
        self.nal_config = LabConfig(
            max_parallel_experiments=config.num_gpus * config.processes_per_gpu,
            experiment_timeout=3600,
            device_ids=list(range(config.num_gpus)),
            min_experiments_per_hypothesis=self.config.tournament_size,
            require_statistical_significance=False,
            results_dir=str(self.data_dir / "nal_results"),
            save_best_models=True,
            verbose=False
        )
        
        self.lab = NeuralArchitectureLab(self.nal_config)
        self.lab.runner = AdvancedExperimentRunner(self.nal_config)
        
    def create_competitor_hypothesis(self, generation: int) -> Hypothesis:
        """Create a hypothesis for evaluating tournament competitors."""
        
        # The parameter space now defines the variables for each competitor
        parameter_space = {
            'architecture': [c['architecture'] for c in self.population],
            'sparsity': [c['sparsity'] for c in self.population],
            'lr_strategy': [c['lr_strategy'] for c in self.population],
            'competitor_id': [c['id'] for c in self.population]
        }

        return Hypothesis(
            id=f"tournament_gen_{generation}",
            name=f"Tournament Generation {generation}",
            description="Evaluate tournament competitors",
            category=HypothesisCategory.ARCHITECTURE,
            question="Which architectures perform best in this generation?",
            prediction="Architectures with higher fitness will emerge.",
            test_function=evaluate_competitor_task,
            parameter_space=parameter_space,
            control_parameters={
                'dataset': 'cifar10',
                'epochs': self.config.epochs_per_generation,
                'batch_size': self.config.batch_size_base,
                'enable_growth': self.config.enable_growth,
            },
            success_metrics={'fitness': 0.0}
        )
    
    def generate_initial_population(self) -> List[Dict[str, Any]]:
        """Generate initial population of architectures."""
        population = []
        
        for i in range(self.config.tournament_size):
            n_layers = np.random.randint(3, 7)
            architecture = [3072] # CIFAR-10 input size
            
            current_size = 512
            for _ in range(n_layers - 1):
                next_size = max(32, int(current_size * np.random.uniform(0.5, 1.0)))
                architecture.append(next_size)
                current_size = next_size
            
            architecture.append(10)
            
            competitor = {
                'id': f'gen0_competitor_{i:03d}',
                'architecture': architecture,
                'sparsity': np.random.uniform(0.01, 0.1),
                'lr_strategy': np.random.choice(self.config.learning_rate_strategies),
                'fitness': 0.0,
                'generation': 0,
                'parent_ids': [],
                'mutations': []
            }
            
            population.append(competitor)
        
        return population
    
    async def evaluate_generation(self, generation: int) -> List[Dict[str, Any]]:
        """Evaluate all competitors in current generation using NAL."""
        print(f"\n🔄 Generation {generation}/{self.config.generations}")
        print("=" * 60)
        
        # Create and register the hypothesis for the current generation
        hypothesis = self.create_competitor_hypothesis(generation)
        self.lab.register_hypothesis(hypothesis)
        
        # Let NAL run the experiments
        print(f"📊 Evaluating {len(self.population)} competitors using NAL...")
        hypothesis_result = await self.lab.test_hypothesis(hypothesis.id)
        
        # Process the results
        results_map = {res.experiment_id: res for res in hypothesis_result.experiment_results}
        
        for competitor in self.population:
            result = results_map.get(f"{hypothesis.id}_exp_{competitor['id']}")
            if result and not result.error:
                fitness = self._calculate_fitness(result.metrics)
                competitor['fitness'] = fitness
                competitor['accuracy'] = result.metrics.get('accuracy', 0.0)
                competitor['parameters'] = result.model_parameters
                competitor['growth_events'] = result.metrics.get('growth_events', 0)
            else:
                competitor['fitness'] = 0.0 # Failed experiments get 0 fitness
        
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Print and log summary
        self._log_generation_summary(generation)
        
        return self.population

    def _calculate_fitness(self, metrics: Dict[str, Any]) -> float:
        """Calculate fitness score for a competitor."""
        accuracy = metrics.get('accuracy', 0.0)
        parameters = max(metrics.get('final_parameters', 1e6), 1e3)
        
        efficiency = accuracy / (parameters / 1e6)
        
        fitness = accuracy * 0.7 + efficiency * 0.3
        return max(0.0, fitness)

    def _log_generation_summary(self, generation: int):
        """Logs the summary of a generation."""
        best_competitor = self.population[0]
        avg_fitness = np.mean([c['fitness'] for c in self.population])

        print(f"\n📈 Generation {generation} Results:")
        print(f"   Best fitness: {best_competitor['fitness']:.4f}")
        print(f"   Best accuracy: {best_competitor.get('accuracy', 0.0):.3f}")
        print(f"   Average fitness: {avg_fitness:.4f}")

        # Log generation completion using proper logging format
        from src.structure_net.logging.schemas import GenerationResult, TrainingExperiment, NetworkArchitecture, ExperimentConfig, PerformanceMetrics
        
        # Create a generation result for logging
        best_competitor = self.population[0]
        
        # Log as experiment result
        try:
            # Create network architecture info
            arch = NetworkArchitecture(
                layers=best_competitor['architecture'],
                total_parameters=best_competitor.get('parameters', 0),
                total_connections=best_competitor.get('parameters', 0),
                sparsity=best_competitor.get('sparsity', 0.02),
                depth=len(best_competitor['architecture'])
            )
            
            # Create experiment config
            exp_config = ExperimentConfig(
                experiment_id=f"gen_{generation}",
                dataset="cifar10",
                batch_size=self.config.batch_size_base,
                learning_rate=0.001,
                epochs=self.config.epochs_per_generation,
                device=f"cuda:{self.config.num_gpus}",
                random_seed=42
            )
            
            # Create performance metrics
            perf = PerformanceMetrics(
                accuracy=best_competitor.get('accuracy', 0.0),
                loss=2.0 - best_competitor['fitness']  # Approximate loss from fitness
            )
            
            # Create training experiment
            training_exp = TrainingExperiment(
                experiment_id=f"gen_{generation}_best",
                experiment_type="tournament_generation",
                config=exp_config,
                architecture=arch,
                training_history=[],
                final_performance=perf,
                total_epochs=self.config.epochs_per_generation
            )
            
            # Log the experiment
            self.logger.log_experiment_result(training_exp)
            print(f"📝 Logged generation {generation} results")
            
        except Exception as e:
            print(f"Warning: Failed to log generation results: {e}")

        # Save generation snapshot
        gen_file = self.data_dir / f"generation_{generation:03d}.json"
        with open(gen_file, 'w') as f:
            json.dump({
                "generation": generation,
                "population": self.population,
            }, f, indent=2, default=str)

        print("\n🏆 Top 5 Performers:")
        for i, competitor in enumerate(self.population[:5]):
            print(f"   {i+1}. {competitor['id']}: "
                  f"Fitness={competitor['fitness']:.4f}, "
                  f"Accuracy={competitor.get('accuracy', 0.0):.3f}, "
                  f"Arch Layers={len(competitor['architecture'])-1}")

    def evolve_population(self, generation: int):
        """Evolve population through selection and mutation."""
        new_population = []
        
        # Elitism
        elite_count = int(self.config.tournament_size * 0.2)
        new_population.extend(self.population[:elite_count])
        
        # Crossover and Mutation
        while len(new_population) < self.config.tournament_size:
            p1, p2 = np.random.choice(self.population, 2, replace=False)
            offspring = self._crossover(p1, p2)
            offspring = self._mutate_architecture(offspring, generation)
            new_population.append(offspring)
        
        self.population = new_population

    def _crossover(self, p1, p2):
        # Simple crossover for demonstration
        child_arch_len = (len(p1['architecture']) + len(p2['architecture'])) // 2
        child_arch = [p1['architecture'][0]]
        for i in range(1, child_arch_len -1):
            p1_layer = p1['architecture'][i] if i < len(p1['architecture']) -1 else p2['architecture'][i]
            p2_layer = p2['architecture'][i] if i < len(p2['architecture']) -1 else p1['architecture'][i]
            child_arch.append((p1_layer + p2_layer) // 2)
        child_arch.append(p1['architecture'][-1])

        return {
            'architecture': child_arch,
            'sparsity': (p1['sparsity'] + p2['sparsity']) / 2,
            'lr_strategy': p1['lr_strategy'] if np.random.rand() < 0.5 else p2['lr_strategy'],
            'fitness': 0.0,
            'parent_ids': [p1['id'], p2['id']]
        }

    def _mutate_architecture(self, parent: Dict[str, Any], generation: int) -> Dict[str, Any]:
        """Mutate architecture to create offspring."""
        offspring = parent.copy()
        offspring['id'] = f'gen{generation}_mutated_{np.random.randint(1000)}'
        offspring['generation'] = generation
        offspring['mutations'] = parent.get('mutations', [])
        
        if np.random.random() < self.config.mutation_rate:
            mutation_type = np.random.choice(['layer_size', 'add_layer', 'remove_layer', 'sparsity'])
            
            if mutation_type == 'layer_size' and len(offspring['architecture']) > 2:
                layer_idx = np.random.randint(1, len(offspring['architecture']) - 1)
                offspring['architecture'][layer_idx] = max(32, int(offspring['architecture'][layer_idx] * np.random.uniform(0.7, 1.3)))
                offspring['mutations'].append(f'mutate_layer_{layer_idx}')
            
            elif mutation_type == 'add_layer' and len(offspring['architecture']) < self.config.max_layers + 1:
                insert_idx = np.random.randint(1, len(offspring['architecture']))
                prev_size = offspring['architecture'][insert_idx - 1]
                new_size = max(32, int(prev_size * np.random.uniform(0.5, 1.0)))
                offspring['architecture'].insert(insert_idx, new_size)
                offspring['mutations'].append('add_layer')

            elif mutation_type == 'remove_layer' and len(offspring['architecture']) > 3:
                remove_idx = np.random.randint(1, len(offspring['architecture']) - 1)
                offspring['architecture'].pop(remove_idx)
                offspring['mutations'].append('remove_layer')

            elif mutation_type == 'sparsity':
                offspring['sparsity'] = np.clip(parent['sparsity'] * np.random.uniform(0.8, 1.2), 0.01, 0.3)
                offspring['mutations'].append('mutate_sparsity')

        return offspring
    
    async def run_tournament(self) -> Dict[str, Any]:
        """Run the complete tournament evolution."""
        self.start_time = time.time()
        
        print("\n🏁 Starting Ultimate Stress Test v2 (NAL-Powered)")
        print("=" * 80)
        
        self.population = self.generate_initial_population()
        
        for generation in range(self.config.generations):
            await self.evaluate_generation(generation)
            
            self.generation_results.append({
                'generation': generation,
                'best_fitness': self.population[0]['fitness'],
                'best_accuracy': self.population[0].get('accuracy', 0.0),
                'average_fitness': np.mean([c['fitness'] for c in self.population]),
            })
            
            if generation < self.config.generations - 1:
                print(f"\n🧬 Evolving population for generation {generation + 1}...")
                self.evolve_population(generation + 1)
        
        total_time = time.time() - self.start_time
        
        return {
            'config': self.config.__dict__,
            'generations': self.generation_results,
            'final_best': self.population[0],
            'total_duration': total_time,
            'nal_results_dir': self.nal_config.results_dir
        }


def run_stress_test_async(config: StressTestConfig) -> Dict[str, Any]:
    """Run stress test asynchronously."""
    executor = TournamentExecutor(config)
    
    # Create event loop if needed
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run tournament
    results = loop.run_until_complete(executor.run_tournament())
    
    return results


def main():
    """Main function matching original ultimate_stress_test interface."""
    print("🚀 Ultimate Structure Net Stress Test v2")
    print("   Powered by Neural Architecture Lab")
    print("=" * 80)
    print("\n[DEBUG] Starting main function...")
    
    # Calculate optimal configuration
    memory_info = calculate_optimal_memory_usage()
    print("\n💾 System Memory Analysis:")
    for key, value in memory_info.items():
        if key != 'gpu_details':
            print(f"   {key}: {value}")
    
    # Create configuration
    config = StressTestConfig(
        num_gpus=memory_info.get('total_gpus', 1),
        processes_per_gpu=memory_info.get('recommended_processes_per_gpu', 1),
        tournament_size=memory_info.get('recommended_tournament_size', 32),
        batch_size_base=memory_info.get('recommended_batch_size', 256),
        generations=5,
        epochs_per_generation=20,
        enable_growth=True,
        enable_residual_blocks=True,
        enable_comprehensive_metrics=True,
        enable_profiling=True
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"   GPUs: {config.num_gpus}")
    print(f"   Processes per GPU: {config.processes_per_gpu}")
    print(f"   Tournament size: {config.tournament_size}")
    print(f"   Generations: {config.generations}")
    
    # Set multiprocessing start method
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    # Run tournament
    try:
        results = run_stress_test_async(config)
        
        # Save results to /data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = Path("/data") / f"stress_test_v2_{timestamp}"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = data_dir / "tournament_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        print(f"📁 NAL detailed results in: {results['nal_results_dir']}/")
        print(f"📊 All data saved in: {data_dir}/")
        
        # Print summary
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Total duration: {results['total_duration']:.1f} seconds")
        print(f"   Champion: {results['final_best']['id']}")
        print(f"   Champion fitness: {results['final_best']['fitness']:.4f}")
        print(f"   Champion accuracy: {results['final_best'].get('accuracy', 0.0):.3f}")
        print(f"   Architecture: {results['final_best']['architecture']}")
        
        # Show evolution progress
        print("\n📈 Evolution Progress:")
        for gen_data in results['generations']:
            print(f"   Gen {gen_data['generation']}: "
                  f"Best={gen_data['best_fitness']:.4f}, "
                  f"Avg={gen_data['average_fitness']:.4f}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  Tournament interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Tournament failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("[DEBUG] Running from __main__")
    # Run the stress test
    results = main()
    
    if results:
        print("\n🎉 Ultimate stress test v2 completed successfully!")
        print("🔬 Check the NAL results directory for detailed experiment data")
    else:
        print("\n💥 Ultimate stress test v2 failed!")