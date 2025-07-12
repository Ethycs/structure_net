#!/usr/bin/env python3
"""
Ultimate Structure Net Stress Test v2 - NAL Architecture Compliant

This script implements a comprehensive tournament-style evolutionary experiment 
using the Neural Architecture Lab (NAL) framework and the new component-based 
architecture. It follows the proper separation of concerns where:

- StressTestConfig: Contains experiment-specific parameters
- LabConfig: Contains NAL framework parameters  
- TournamentOrchestrator: Translates between configs and orchestrates evolution
- NAL: Handles execution, parallelization, and result analysis

Architecture Flow:
StressTestConfig -> TournamentOrchestrator -> Hypothesis -> NAL -> Experiments -> Results
"""

import os
import sys
import asyncio
import torch.multiprocessing as mp
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NAL Framework Imports
from src.neural_architecture_lab import NeuralArchitectureLab, LabConfig, Hypothesis, HypothesisCategory
from src.neural_architecture_lab.core import LabConfigFactory

# Component-based Architecture Imports
from src.structure_net.components.orchestrators.tournament_orchestrator import TournamentOrchestrator
from src.structure_net.core.interfaces import ComponentVersion, Maturity


@dataclass
class StressTestConfig:
    """
    Configuration for the tournament-style stress test experiment.
    
    This contains all parameters specific to the tournament evolution,
    separate from the NAL framework configuration.
    """
    # Evolution Parameters
    generations: int = 10
    tournament_size: int = 32
    mutation_rate: float = 0.1
    elitism_rate: float = 0.1
    
    # Training Parameters
    epochs_per_generation: int = 5
    batch_size_base: int = 128
    learning_rate_base: float = 0.001
    
    # Architecture Search Space
    min_layers: int = 3
    max_layers: int = 8
    min_neurons: int = 32
    max_neurons: int = 512
    sparsity_range: tuple = (0.01, 0.1)
    
    # Dataset and Environment
    dataset_name: str = "cifar10"
    learning_rate_strategies: List[str] = field(default_factory=lambda: ["constant", "cosine", "step"])
    
    # Experiment Control
    seed_model_dir: Optional[str] = None
    enable_growth: bool = True
    enable_pruning: bool = True
    enable_adaptive_lr: bool = True
    
    # Output and Logging
    verbose: bool = False
    save_best_models: bool = True
    checkpoint_frequency: int = 5  # Save every N generations
    
    # Resource Management
    max_parallel_competitors: int = 8
    memory_limit_gb: float = 16.0
    timeout_per_competitor: int = 3600  # 1 hour


def get_default_lab_config() -> LabConfig:
    """
    Returns the default NAL configuration for stress testing.
    
    This configuration is optimized for intensive, parallel evolution experiments.
    """
    return LabConfig(
        # Execution Configuration
        max_parallel_experiments=8,
        max_retries=2,
        timeout_seconds=3600,
        
        # Resource Management
        gpu_memory_fraction=0.9,
        enable_memory_growth=True,
        
        # Output and Storage
        results_dir="stress_test_results",
        save_models=True,
        save_detailed_logs=True,
        
        # Analysis Configuration
        enable_statistical_analysis=True,
        confidence_level=0.95,
        
        # Performance Optimization
        prefetch_datasets=True,
        cache_results=True,
        
        # Logging
        log_level=logging.INFO,
        verbose=False
    )


def create_tournament_hypothesis(
    stress_config: StressTestConfig, 
    generation: int,
    population: List[Dict[str, Any]]
) -> Hypothesis:
    """
    Creates a properly formatted NAL Hypothesis for a tournament generation.
    
    This function translates the stress test configuration into the generic
    Hypothesis format that the NAL framework expects.
    
    Args:
        stress_config: The stress test configuration
        generation: Current generation number
        population: Current population of competitors
        
    Returns:
        A Hypothesis object ready for NAL execution
    """
    from src.neural_architecture_lab.workers.tournament_worker import evaluate_competitor_task
    
    # Create parameter space from population
    parameter_space = {
        'params': [
            {
                'competitor_id': competitor['id'],
                'architecture': competitor['architecture'],
                'sparsity': competitor['sparsity'],
                'lr_strategy': competitor['lr_strategy'],
                'seed_path': competitor.get('seed_path')
            }
            for competitor in population
        ]
    }
    
    # Package control parameters
    control_parameters = {
        'dataset': stress_config.dataset_name,
        'epochs': stress_config.epochs_per_generation,
        'batch_size': stress_config.batch_size_base,
        'learning_rate': stress_config.learning_rate_base,
        'enable_growth': stress_config.enable_growth,
        'enable_pruning': stress_config.enable_pruning,
        'enable_adaptive_lr': stress_config.enable_adaptive_lr,
        'generation': generation,
        'num_workers': 2,
        'pin_memory': True
    }
    
    return Hypothesis(
        id=f"tournament_stress_test_gen_{generation:03d}",
        name=f"Tournament Stress Test - Generation {generation}",
        description=f"Evolutionary tournament testing {len(population)} competitors in generation {generation}",
        question="Which neural architectures demonstrate superior fitness under resource constraints?",
        prediction="Architectures with optimal balance of accuracy and efficiency will emerge through tournament selection",
        
        # Core Execution
        test_function=evaluate_competitor_task,
        parameter_space=parameter_space,
        control_parameters=control_parameters,
        
        # Success Criteria
        success_metrics={
            'fitness': 0.1,  # Minimum fitness threshold
            'accuracy': 0.5,  # Minimum accuracy threshold
            'convergence_rate': 0.8  # Population convergence threshold
        },
        
        # Metadata
        category=HypothesisCategory.ARCHITECTURE,
        tags=['evolution', 'tournament', 'stress_test', 'architecture_search'],
        
        # Resource Requirements
        expected_runtime_minutes=60,
        memory_requirements_gb=stress_config.memory_limit_gb,
        
        # Analysis Configuration
        analysis_config={
            'track_population_diversity': True,
            'compute_fitness_statistics': True,
            'save_generation_snapshots': True,
            'enable_convergence_analysis': True
        }
    )


async def run_stress_test(lab_config: LabConfig, stress_config: StressTestConfig) -> List[Dict[str, Any]]:
    """
    Orchestrates the complete tournament stress test using the NAL framework.
    
    This is the main execution function that coordinates between the stress test
    configuration and the NAL framework to run the evolutionary experiment.
    
    Args:
        lab_config: NAL framework configuration
        stress_config: Stress test specific configuration
        
    Returns:
        List of generation results with population evolution data
    """
    print(f"🚀 Starting Ultimate Stress Test v2")
    print(f"   Generations: {stress_config.generations}")
    print(f"   Tournament Size: {stress_config.tournament_size}")
    print(f"   Dataset: {stress_config.dataset_name}")
    print(f"   NAL Config: {lab_config.results_dir}")
    print("=" * 60)
    
    # Initialize the tournament orchestrator with proper component architecture
    orchestrator = TournamentOrchestrator(
        lab_config=lab_config,
        stress_test_config=stress_config,
        name="StressTestOrchestrator"
    )
    
    try:
        # Run the tournament using the component-based architecture
        generation_results = await orchestrator.run_tournament()
        
        # Post-processing and analysis
        print("\n" + "=" * 60)
        print("🏆 TOURNAMENT COMPLETE - Final Results")
        print("=" * 60)
        
        for generation, population in enumerate(generation_results):
            if population:
                # Sort by fitness (descending)
                sorted_pop = sorted(population, key=lambda x: x.get('fitness', 0.0), reverse=True)
                best = sorted_pop[0]
                avg_fitness = sum(c.get('fitness', 0.0) for c in population) / len(population)
                
                print(f"Generation {generation + 1:2d}: "
                      f"Best Fitness={best.get('fitness', 0.0):.4f}, "
                      f"Best Accuracy={best.get('accuracy', 0.0):.2%}, "
                      f"Avg Fitness={avg_fitness:.4f}, "
                      f"Population Size={len(population)}")
                
                if stress_config.verbose and generation == len(generation_results) - 1:
                    print(f"   🥇 Champion Architecture: {best.get('architecture', 'Unknown')}")
                    print(f"   📊 Champion Parameters: {best.get('parameters', 'Unknown')}")
                    print(f"   🎯 Champion Sparsity: {best.get('sparsity', 'Unknown')}")
            else:
                print(f"Generation {generation + 1:2d}: No survivors")
        
        return generation_results
        
    except Exception as e:
        print(f"❌ Tournament failed with error: {e}")
        if stress_config.verbose:
            import traceback
            traceback.print_exc()
        raise


def main():
    """
    Main entry point for the NAL-compliant ultimate stress test.
    
    Handles command-line arguments, configuration setup, and orchestrates
    the execution of the tournament stress test using proper separation
    of concerns between experiment config and framework config.
    """
    # Initialize default configurations
    stress_config = StressTestConfig()
    lab_config = get_default_lab_config()
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Ultimate Structure Net Stress Test v2 - NAL Architecture Compliant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Stress Test Specific Arguments
    stress_group = parser.add_argument_group('Stress Test Parameters')
    stress_group.add_argument('--generations', type=int, default=stress_config.generations,
                             help='Number of evolutionary generations to run')
    stress_group.add_argument('--tournament-size', type=int, default=stress_config.tournament_size,
                             help='Number of competitors per generation')
    stress_group.add_argument('--mutation-rate', type=float, default=stress_config.mutation_rate,
                             help='Mutation rate for evolution')
    stress_group.add_argument('--epochs', type=int, default=stress_config.epochs_per_generation,
                             help='Training epochs per generation')
    stress_group.add_argument('--dataset', type=str, default=stress_config.dataset_name,
                             help='Dataset to use for training/testing')
    stress_group.add_argument('--seed-model-dir', type=str, default=stress_config.seed_model_dir,
                             help='Directory containing seed models to initialize population')
    
    # Control Arguments
    control_group = parser.add_argument_group('Experiment Control')
    control_group.add_argument('--disable-growth', action='store_true',
                              help='Disable dynamic network growth')
    control_group.add_argument('--disable-pruning', action='store_true',
                              help='Disable network pruning')
    control_group.add_argument('--disable-adaptive-lr', action='store_true',
                              help='Disable adaptive learning rates')
    control_group.add_argument('--verbose', action='store_true',
                              help='Enable detailed output and logging')
    control_group.add_argument('--memory-limit', type=float, default=stress_config.memory_limit_gb,
                              help='Memory limit in GB per experiment')
    
    # Add NAL framework arguments
    LabConfigFactory.add_arguments(parser)
    
    args = parser.parse_args()
    
    # Update configurations from arguments
    # Stress Test Config Updates
    stress_config.generations = args.generations
    stress_config.tournament_size = args.tournament_size
    stress_config.mutation_rate = args.mutation_rate
    stress_config.epochs_per_generation = args.epochs
    stress_config.dataset_name = args.dataset
    stress_config.seed_model_dir = args.seed_model_dir
    stress_config.enable_growth = not args.disable_growth
    stress_config.enable_pruning = not args.disable_pruning
    stress_config.enable_adaptive_lr = not args.disable_adaptive_lr
    stress_config.verbose = args.verbose
    stress_config.memory_limit_gb = args.memory_limit
    
    # NAL Config Updates (using LabConfigFactory)
    lab_config = LabConfigFactory.from_args(args, base_config=lab_config)
    lab_config.verbose = args.verbose
    
    # Adjust results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lab_config.results_dir = f"{lab_config.results_dir}/stress_test_{timestamp}"
    
    print(f"🔬 Neural Architecture Lab Ultimate Stress Test v2")
    print(f"📁 Results will be saved to: {lab_config.results_dir}")
    print(f"🎯 Target: {stress_config.generations} generations with {stress_config.tournament_size} competitors each")
    print(f"📊 Dataset: {stress_config.dataset_name}")
    print(f"⚡ Features: Growth={stress_config.enable_growth}, "
          f"Pruning={stress_config.enable_pruning}, "
          f"Adaptive LR={stress_config.enable_adaptive_lr}")
    
    # Execute the stress test
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(
            run_stress_test(lab_config, stress_config)
        )
        
        print(f"\n✅ Stress test completed successfully!")
        print(f"📈 Final population evolved through {len(results)} generations")
        
        if results and results[-1]:
            best_final = max(results[-1], key=lambda x: x.get('fitness', 0.0))
            print(f"🏆 Final Champion Fitness: {best_final.get('fitness', 0.0):.4f}")
            print(f"🎯 Final Champion Accuracy: {best_final.get('accuracy', 0.0):.2%}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Stress test interrupted by user")
    except Exception as e:
        print(f"\n❌ Stress test failed: {e}")
        if stress_config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        if 'loop' in locals():
            loop.close()


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    main()