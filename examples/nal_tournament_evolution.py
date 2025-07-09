#!/usr/bin/env python3
"""
Neural Architecture Lab Tournament Evolution

This script demonstrates how to use NAL to run tournament-style evolution
experiments similar to the original ultimate_stress_test but with more
scientific rigor and better analysis.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import asyncio
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch

from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    Hypothesis,
    HypothesisCategory
)
from src.structure_net.core.io_operations import load_model_seed


def create_tournament_hypothesis(
    tournament_size: int = 32,
    generations: int = 5,
    mutation_strategies: List[str] = None
) -> Hypothesis:
    """Create a hypothesis for tournament-style architecture evolution."""
    
    if mutation_strategies is None:
        mutation_strategies = ['layer_size', 'add_layer', 'remove_layer', 'sparsity', 'mixed']
    
    def test_tournament_evolution(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Tournament evolution test function - handled by custom runner."""
        return None, {}
    
    tournament_hypothesis = Hypothesis(
        id="architecture_tournament_evolution",
        name="Tournament-Style Architecture Evolution",
        description="Evolve neural architectures through competitive tournament selection",
        category=HypothesisCategory.ARCHITECTURE,
        question="Which evolutionary strategies produce the best architectures through tournament selection?",
        prediction="Mixed mutation strategies with adaptive rates will produce superior architectures",
        test_function=test_tournament_evolution,
        parameter_space={
            'tournament_size': [tournament_size],
            'generations': [generations],
            'mutation_strategy': mutation_strategies,
            'mutation_rate': [0.1, 0.2, 0.3],
            'selection_pressure': [0.1, 0.2, 0.3],  # Fraction eliminated each generation
            'crossover_rate': [0.0, 0.1, 0.2],
            'elitism_rate': [0.1, 0.2],  # Fraction of best to keep unchanged
            'initial_architectures': [
                [[784, 128, 10], [784, 256, 10], [784, 64, 32, 10]],  # Simple starting points
                [[784, 256, 128, 10], [784, 512, 256, 10], [784, 128, 64, 32, 10]],  # Medium
                'random'  # Generate random architectures
            ]
        },
        control_parameters={
            'dataset': 'cifar10',
            'epochs_per_generation': 20,
            'batch_size': 128,
            'lr_strategy': 'comprehensive',
            'enable_growth': True,
            'enable_metrics': True,
            'tournament_metric': 'fitness',  # Custom fitness function
            'primary_metric_type': 'accuracy'
        },
        success_metrics={
            'final_best_accuracy': 0.55,
            'improvement_over_baseline': 1.1,  # 10% improvement
            'architecture_diversity': 0.5  # Maintain diversity in population
        },
        tags=['tournament', 'evolution', 'architecture_search', 'competitive']
    )
    
    return tournament_hypothesis


def create_parallel_population_hypothesis(
    num_populations: int = 4,
    migration_rate: float = 0.1
) -> Hypothesis:
    """Create hypothesis for parallel population evolution with migration."""
    
    def test_parallel_populations(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    parallel_hypothesis = Hypothesis(
        id="parallel_population_evolution",
        name="Parallel Population Evolution with Migration",
        description="Test island model evolution with multiple parallel populations",
        category=HypothesisCategory.ARCHITECTURE,
        question="Does maintaining separate populations with migration improve evolution outcomes?",
        prediction="Migration between populations will prevent premature convergence and find better solutions",
        test_function=test_parallel_populations,
        parameter_space={
            'num_populations': [2, 4, 8],
            'population_size': [8, 16],  # Per population
            'migration_rate': [0.0, 0.05, 0.1, 0.2],
            'migration_strategy': ['best', 'random', 'tournament'],
            'migration_frequency': [1, 3, 5],  # Every N generations
            'population_initialization': ['diverse', 'similar', 'mixed']
        },
        control_parameters={
            'dataset': 'cifar10',
            'total_generations': 10,
            'epochs_per_generation': 15,
            'batch_size': 128,
            'lr_strategy': 'advanced',
            'primary_metric_type': 'accuracy'
        },
        success_metrics={
            'best_accuracy': 0.55,
            'population_diversity': 0.3,  # Maintain genetic diversity
            'convergence_rate': 0.8  # Fraction of populations that converge
        },
        tags=['parallel_evolution', 'island_model', 'migration', 'diversity']
    )
    
    return parallel_hypothesis


def create_adaptive_evolution_hypothesis() -> Hypothesis:
    """Create hypothesis for adaptive evolutionary strategies."""
    
    def test_adaptive_evolution(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    adaptive_hypothesis = Hypothesis(
        id="adaptive_evolution_strategies",
        name="Adaptive Evolution Strategy Selection",
        description="Test self-adapting evolutionary strategies based on population dynamics",
        category=HypothesisCategory.OPTIMIZATION,
        question="Can evolutionary algorithms adapt their own parameters for better results?",
        prediction="Self-adaptive mutation rates will outperform fixed strategies",
        test_function=test_adaptive_evolution,
        parameter_space={
            'adaptation_strategy': ['fitness_based', 'diversity_based', 'hybrid'],
            'initial_mutation_rate': [0.1, 0.2],
            'adaptation_rate': [0.01, 0.05, 0.1],
            'adaptation_window': [3, 5, 10],  # Generations to consider
            'mutation_bounds': [(0.01, 0.5), (0.05, 0.3)],
            'crossover_adaptation': [True, False]
        },
        control_parameters={
            'dataset': 'cifar10',
            'tournament_size': 32,
            'generations': 15,
            'epochs_per_generation': 20,
            'batch_size': 128,
            'lr_strategy': 'ultimate',
            'primary_metric_type': 'efficiency'
        },
        success_metrics={
            'final_accuracy': 0.55,
            'adaptation_effectiveness': 1.1,  # vs fixed strategies
            'parameter_stability': 0.8  # Convergence of adapted parameters
        },
        tags=['adaptive', 'self_tuning', 'meta_evolution']
    )
    
    return adaptive_hypothesis


def create_hybrid_search_hypothesis(seed_models: List[str] = None) -> Hypothesis:
    """Create hypothesis combining evolution with gradient-based NAS."""
    
    def test_hybrid_search(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        return None, {}
    
    hybrid_hypothesis = Hypothesis(
        id="hybrid_architecture_search",
        name="Hybrid Evolution and Gradient-Based Search",
        description="Combine evolutionary algorithms with gradient-based architecture optimization",
        category=HypothesisCategory.ARCHITECTURE,
        question="Does combining evolution with gradient-based methods improve architecture search?",
        prediction="Hybrid approach will find better architectures faster than pure evolution",
        test_function=test_hybrid_search,
        parameter_space={
            'search_strategy': ['evolution_only', 'gradient_only', 'hybrid_sequential', 'hybrid_parallel'],
            'evolution_generations': [5, 10],
            'gradient_steps': [50, 100],
            'hybrid_switch_point': [0.3, 0.5, 0.7],  # When to switch/combine
            'use_seed': [True, False] if seed_models else [False],
            'seed_path': seed_models[0] if seed_models else None
        },
        control_parameters={
            'dataset': 'cifar10',
            'total_budget': 100,  # Total epochs across all methods
            'batch_size': 128,
            'architecture_space': 'darts',  # Use DARTS-style continuous relaxation
            'primary_metric_type': 'accuracy'
        },
        success_metrics={
            'accuracy': 0.6,
            'search_efficiency': 1.5,  # vs pure evolution
            'architecture_quality': 0.9  # Measured by various metrics
        },
        tags=['hybrid_search', 'nas', 'gradient_based', 'evolution']
    )
    
    return hybrid_hypothesis


async def run_custom_tournament(lab: NeuralArchitectureLab, config: Dict[str, Any]):
    """Run a custom tournament evolution experiment."""
    print("\n🏆 Running Custom Tournament Evolution")
    print("=" * 50)
    
    # This is where you would implement custom tournament logic
    # For now, we'll use the standard hypothesis testing
    tournament_hypothesis = create_tournament_hypothesis(
        tournament_size=config.get('tournament_size', 32),
        generations=config.get('generations', 5)
    )
    
    lab.register_hypothesis(tournament_hypothesis)
    result = await lab.test_hypothesis(tournament_hypothesis.id)
    
    print(f"\n📊 Tournament Results:")
    print(f"  Confirmed: {'✓' if result.confirmed else '✗'}")
    print(f"  Best accuracy: {result.best_metrics.get('accuracy', 0):.3f}")
    print(f"  Effect size: {result.effect_size:.3f}")
    
    if result.key_insights:
        print("\n💡 Key Insights:")
        for insight in result.key_insights[:3]:
            print(f"  • {insight}")
    
    return result


async def main():
    """Run tournament-style evolution experiments with NAL."""
    parser = argparse.ArgumentParser(
        description="Neural Architecture Lab Tournament Evolution"
    )
    parser.add_argument("--mode", choices=['tournament', 'parallel', 'adaptive', 'hybrid', 'all'],
                       default='tournament', help="Evolution mode to test")
    parser.add_argument("--tournament-size", type=int, default=32,
                       help="Size of tournament population")
    parser.add_argument("--generations", type=int, default=5,
                       help="Number of evolution generations")
    parser.add_argument("--seeds", nargs='+', help="Seed model paths")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test")
    
    args = parser.parse_args()
    
    print("🔬 Neural Architecture Lab - Tournament Evolution")
    print("=" * 60)
    
    # Configure the lab
    config = LabConfig(
        max_parallel_experiments=8 if not args.quick else 4,
        experiment_timeout=1800 if not args.quick else 600,
        device_ids=[0, 1] if torch.cuda.device_count() > 1 else [0],
        min_experiments_per_hypothesis=3 if args.quick else 5,
        require_statistical_significance=not args.quick,
        results_dir=f"nal_tournament_results",
        save_best_models=True,
        verbose=True
    )
    
    lab = NeuralArchitectureLab(config)
    
    # Create and register hypotheses based on mode
    hypotheses = []
    
    if args.mode in ['tournament', 'all']:
        hypotheses.append(create_tournament_hypothesis(
            args.tournament_size, args.generations
        ))
    
    if args.mode in ['parallel', 'all']:
        hypotheses.append(create_parallel_population_hypothesis())
    
    if args.mode in ['adaptive', 'all']:
        hypotheses.append(create_adaptive_evolution_hypothesis())
    
    if args.mode in ['hybrid', 'all']:
        hypotheses.append(create_hybrid_search_hypothesis(args.seeds))
    
    # Register hypotheses
    for hypothesis in hypotheses:
        lab.register_hypothesis(hypothesis)
    
    print(f"\n📋 Registered {len(hypotheses)} evolution hypotheses")
    
    # Run experiments
    if args.mode == 'tournament':
        # Special handling for tournament mode
        result = await run_custom_tournament(lab, {
            'tournament_size': args.tournament_size,
            'generations': args.generations
        })
    else:
        # Run all registered hypotheses
        results = await lab.run_all_hypotheses()
        
        # Summary
        print("\n📊 Evolution Experiment Summary:")
        for hypothesis_id, result in results.items():
            hypothesis = lab.hypotheses[hypothesis_id]
            print(f"\n{hypothesis.name}:")
            print(f"  Confirmed: {'✓' if result.confirmed else '✗'}")
            print(f"  Best accuracy: {result.best_metrics.get('accuracy', 0):.3f}")
            print(f"  Key insight: {result.key_insights[0] if result.key_insights else 'None'}")
    
    print(f"\n🎯 Results saved to: {config.results_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
    print("\n🏁 Tournament evolution experiments complete!")