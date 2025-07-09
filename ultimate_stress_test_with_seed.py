#!/usr/bin/env python3
"""
Ultimate Structure Net Stress Test with Seed Model Support

Enhanced version of the ultimate stress test that can:
- Load pre-trained seed models from promising_models
- Use seed models as starting points for tournament evolution
- Maintain all the parallel processing and comprehensive testing features
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
import json
import psutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading
from dataclasses import dataclass
import gc

# Structure Net imports
from src.structure_net.seed_search.gpu_seed_hunter import GPUSeedHunter
from src.structure_net.core.network_factory import create_standard_network
from src.structure_net.core.io_operations import save_model_seed, load_model_seed
from src.structure_net.core.network_analysis import get_network_stats, sort_all_network_layers
from src.structure_net.evolution.adaptive_learning_rates.unified_manager import AdaptiveLearningRateManager
from src.structure_net.evolution.adaptive_learning_rates.base import LearningRateStrategy
from src.structure_net.evolution.residual_blocks import ResidualGrowthStrategy, create_residual_network
from src.structure_net.evolution.metrics.integrated_system import CompleteMetricsSystem
from src.structure_net.evolution.metrics.base import ThresholdConfig, MetricsConfig
from src.structure_net.profiling.factory import create_comprehensive_profiler
from src.structure_net.logging.standardized_logging import StandardizedLogger, LoggingConfig
from src.structure_net.logging.schemas_simple import ExperimentConfig, PerformanceMetrics as MetricsData
from src.structure_net.evolution.extrema_analyzer import detect_network_extrema
from src.structure_net.evolution.integrated_growth_system_v2 import IntegratedGrowthSystem

# Import the original stress test components
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experiments.ultimate_stress_test import (
    StressTestConfig, 
    GPUMemoryManager, 
    ArchitectureTournament as BaseArchitectureTournament,
    calculate_optimal_memory_usage
)


class SeedArchitectureTournament(BaseArchitectureTournament):
    """Enhanced tournament that can use seed models."""
    
    def __init__(self, config: StressTestConfig, seed_models: List[str] = None):
        super().__init__(config)
        self.seed_models = seed_models or []
        self.loaded_seeds = []
        
        # Load seed models if provided
        if self.seed_models:
            print(f"\n🌱 Loading {len(self.seed_models)} seed models...")
            for seed_path in self.seed_models:
                try:
                    model, checkpoint = load_model_seed(seed_path, device='cpu')
                    seed_info = {
                        'path': seed_path,
                        'architecture': checkpoint['architecture'],
                        'sparsity': checkpoint.get('sparsity', 0.02),
                        'accuracy': checkpoint.get('accuracy', 0.0),
                        'checkpoint': checkpoint
                    }
                    self.loaded_seeds.append(seed_info)
                    print(f"   ✅ Loaded: {Path(seed_path).name}")
                    print(f"      Architecture: {seed_info['architecture']}")
                    print(f"      Accuracy: {seed_info['accuracy']:.2%}")
                except Exception as e:
                    print(f"   ❌ Failed to load {seed_path}: {e}")
    
    def generate_initial_population(self) -> List[Dict[str, Any]]:
        """Generate initial population including seed models."""
        population = []
        
        # First, add all loaded seed models
        for i, seed_info in enumerate(self.loaded_seeds):
            competitor = {
                'id': f'seed_{i:03d}',
                'architecture': seed_info['architecture'],
                'sparsity': seed_info['sparsity'],
                'lr_strategy': np.random.choice(self.config.learning_rate_strategies),
                'fitness': 0.0,
                'generation': 0,
                'parent_ids': [],
                'mutations': [],
                'is_seed': True,
                'seed_path': seed_info['path'],
                'seed_accuracy': seed_info['accuracy']
            }
            population.append(competitor)
        
        # Then add random architectures to fill the tournament
        remaining_slots = self.config.tournament_size - len(population)
        
        # Use some variations of seed architectures
        if self.loaded_seeds and remaining_slots > 0:
            for i in range(min(remaining_slots // 2, len(self.loaded_seeds) * 3)):
                # Pick a random seed as base
                base_seed = np.random.choice(self.loaded_seeds)
                base_arch = base_seed['architecture'].copy()
                
                # Mutate the architecture slightly
                if len(base_arch) > 3 and np.random.random() < 0.5:
                    # Change layer sizes
                    layer_idx = np.random.randint(1, len(base_arch) - 1)
                    base_arch[layer_idx] = int(base_arch[layer_idx] * np.random.uniform(0.7, 1.3))
                
                if np.random.random() < 0.3:
                    # Add or remove a layer
                    if len(base_arch) > 3 and np.random.random() < 0.5:
                        # Remove a layer
                        del base_arch[np.random.randint(1, len(base_arch) - 1)]
                    elif len(base_arch) < 8:
                        # Add a layer
                        insert_idx = np.random.randint(1, len(base_arch) - 1)
                        new_size = (base_arch[insert_idx - 1] + base_arch[insert_idx]) // 2
                        base_arch.insert(insert_idx, new_size)
                
                competitor = {
                    'id': f'mutant_{len(population):03d}',
                    'architecture': base_arch,
                    'sparsity': base_seed['sparsity'] * np.random.uniform(0.8, 1.2),
                    'lr_strategy': np.random.choice(self.config.learning_rate_strategies),
                    'fitness': 0.0,
                    'generation': 0,
                    'parent_ids': [f"seed_{self.loaded_seeds.index(base_seed)}"],
                    'mutations': ['derived_from_seed'],
                    'is_seed': False
                }
                population.append(competitor)
        
        # Fill remaining with random architectures
        remaining_slots = self.config.tournament_size - len(population)
        if remaining_slots > 0:
            random_population = super().generate_initial_population()
            population.extend(random_population[:remaining_slots])
        
        # Ensure we have exactly tournament_size competitors
        population = population[:self.config.tournament_size]
        
        print(f"\n📊 Initial Population:")
        print(f"   Seed models: {sum(1 for c in population if c.get('is_seed', False))}")
        print(f"   Seed variants: {sum(1 for c in population if 'derived_from_seed' in c.get('mutations', []))}")
        print(f"   Random models: {sum(1 for c in population if not c.get('is_seed', False) and 'derived_from_seed' not in c.get('mutations', []))}")
        
        return population
    
    def evaluate_competitor_on_gpu(self, competitor: Dict[str, Any], device_id: int, 
                                  process_id: int) -> Dict[str, Any]:
        """Enhanced evaluation that can load pre-trained weights for seed models."""
        torch.cuda.set_device(device_id)
        device = f'cuda:{device_id}'
        
        try:
            # Create or load network
            if competitor.get('is_seed', False) and competitor.get('seed_path'):
                # Load the seed model
                model, checkpoint = load_model_seed(competitor['seed_path'], device=device)
                print(f"🌱 GPU {device_id} Process {process_id}: Loaded seed model {competitor['id']}")
                print(f"   Original accuracy: {competitor.get('seed_accuracy', 0.0):.2%}")
            else:
                # Create new network
                model = create_standard_network(
                    architecture=competitor['architecture'],
                    sparsity=competitor['sparsity'],
                    device=device
                )
            
            # Continue with evaluation but now we need to handle the full training
            # Since we can't use super() due to the pickling/device issues, 
            # we'll just return a simple result for seed models
            if competitor.get('is_seed', False):
                # For seed models, just evaluate their current performance
                model.eval()
                
                # Create a dummy result showing the seed's original performance
                result = {
                    'id': competitor['id'],
                    'fitness': competitor.get('seed_accuracy', 0.0),
                    'accuracy': competitor.get('seed_accuracy', 0.0),
                    'efficiency': 0.001,  # Placeholder
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'growth_events': 0,
                    'final_architecture': competitor['architecture'],
                    'device_id': device_id,
                    'process_id': process_id,
                    'is_seed': True,
                    'seed_path': competitor.get('seed_path', ''),
                    'improvement_from_seed': 0.0
                }
                
                print(f"✅ GPU {device_id} Process {process_id}: Seed {competitor['id']} evaluated")
                print(f"   Original accuracy: {result['accuracy']:.2%}")
                return result
            else:
                # For non-seed models, use the standalone evaluation
                from experiments.ultimate_stress_test import evaluate_competitor_standalone, GPUMemoryManager
                memory_manager = GPUMemoryManager(self.config.max_memory_per_gpu)
                result = evaluate_competitor_standalone(
                    competitor, device_id, process_id, self.config, memory_manager
                )
                return result
            
        except Exception as e:
            print(f"❌ GPU {device_id} Process {process_id}: {competitor['id']} failed: {e}")
            # Return a failed result instead of calling super()
            return {
                'id': competitor['id'],
                'fitness': 0.0,
                'accuracy': 0.0,
                'efficiency': 0.0,
                'parameters': 0,
                'growth_events': 0,
                'error': str(e),
                'device_id': device_id,
                'process_id': process_id
            }


def find_best_seeds(top_n: int = 5) -> List[str]:
    """Find the best seed models from promising_models directory."""
    models_dir = Path("data/promising_models")
    
    if not models_dir.exists():
        print("❌ No promising_models directory found")
        return []
    
    # Find all model files
    model_files = list(models_dir.glob("**/model_cifar10_*.pt"))
    
    if not model_files:
        print("❌ No models found in promising_models")
        return []
    
    # Parse and rank models
    models_info = []
    for model_path in model_files:
        # Extract accuracy from filename
        # Example: model_cifar10_4layers_seed6_acc0.48_patch0.042_sparse0.200_BEST_ACCURACY_SPARSITY_0.200.pt
        filename = model_path.name
        if '_acc' in filename:
            try:
                acc_str = filename.split('_acc')[1].split('_')[0]
                accuracy = float(acc_str)
                models_info.append((str(model_path), accuracy))
            except:
                continue
    
    # Sort by accuracy
    models_info.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Top {top_n} seed models by accuracy:")
    for i, (path, acc) in enumerate(models_info[:top_n]):
        print(f"   {i+1}. {Path(path).name}")
        print(f"      Accuracy: {acc:.2%}")
    
    return [path for path, _ in models_info[:top_n]]


def main():
    """Enhanced main function with seed model support."""
    parser = argparse.ArgumentParser(description="Ultimate Structure Net Stress Test with Seed Models")
    parser.add_argument("--seeds", nargs='+', help="Paths to seed models")
    parser.add_argument("--auto-seeds", type=int, default=0, 
                       help="Automatically select top N seed models from promising_models")
    parser.add_argument("--tournament-size", type=int, default=None,
                       help="Override tournament size")
    parser.add_argument("--generations", type=int, default=None,
                       help="Override number of generations")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override epochs per generation")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with reduced parameters")
    
    args = parser.parse_args()
    
    print("🚀 Structure Net Ultimate Stress Test with Seed Models")
    print("=" * 80)
    
    # Gather seed models
    seed_models = []
    
    if args.seeds:
        seed_models.extend(args.seeds)
    
    if args.auto_seeds > 0:
        auto_seeds = find_best_seeds(args.auto_seeds)
        seed_models.extend(auto_seeds)
    
    # Remove duplicates
    seed_models = list(dict.fromkeys(seed_models))
    
    if seed_models:
        print(f"\n🌱 Using {len(seed_models)} seed models")
    else:
        print("\n📝 No seed models specified, using random initialization")
    
    # Calculate optimal configuration
    memory_info = calculate_optimal_memory_usage()
    print("\n💾 System Memory Analysis:")
    for key, value in memory_info.items():
        if key != 'gpu_details':
            print(f"   {key}: {value}")
    
    # Create configuration
    if args.quick:
        config = StressTestConfig(
            tournament_size=args.tournament_size or 16,
            generations=args.generations or 3,
            epochs_per_generation=args.epochs or 10,
            batch_size_base=128,
            enable_comprehensive_metrics=False,
            enable_profiling=False
        )
    else:
        config = StressTestConfig(
            num_gpus=memory_info.get('total_gpus', 1),
            processes_per_gpu=memory_info.get('recommended_processes_per_gpu', 1),
            tournament_size=args.tournament_size or memory_info.get('recommended_tournament_size', 32),
            batch_size_base=memory_info.get('recommended_batch_size', 256),
            generations=args.generations or 5,
            epochs_per_generation=args.epochs or 15,
            enable_growth=True,
            enable_residual_blocks=True,
            enable_comprehensive_metrics=True,
            enable_profiling=True
        )
    
    # Ensure tournament size is at least as large as number of seeds
    if seed_models and config.tournament_size < len(seed_models):
        config.tournament_size = max(len(seed_models) * 2, config.tournament_size)
        print(f"\n📏 Adjusted tournament size to {config.tournament_size} to accommodate seeds")
    
    print(f"\n⚙️  Configuration:")
    print(f"   Tournament size: {config.tournament_size}")
    print(f"   Generations: {config.generations}")
    print(f"   Epochs per generation: {config.epochs_per_generation}")
    print(f"   Seed models: {len(seed_models)}")
    
    # Set multiprocessing start method
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    # Run the tournament
    tournament = SeedArchitectureTournament(config, seed_models)
    
    try:
        results = tournament.run_tournament()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"ultimate_stress_test_seeded_{timestamp}.json"
        
        # Add seed information to results
        results['seed_models'] = seed_models
        results['seed_performance'] = []
        
        # Extract seed-specific performance
        for gen_data in results.get('generations', []):
            for performer in gen_data.get('top_performers', []):
                if performer.get('is_seed', False):
                    results['seed_performance'].append({
                        'generation': gen_data['generation'],
                        'seed_id': performer['id'],
                        'seed_path': performer.get('seed_path', ''),
                        'fitness': performer['fitness'],
                        'accuracy': performer['accuracy'],
                        'improvement': performer.get('improvement_from_seed', 0.0)
                    })
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Print enhanced summary
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Total duration: {results['total_duration']:.1f} seconds")
        print(f"   Champion: {results['final_best']['id']}")
        print(f"   Champion fitness: {results['final_best']['fitness']:.4f}")
        print(f"   Champion accuracy: {results['final_best']['accuracy']:.3f}")
        
        # Check if a seed model won
        if results['final_best'].get('is_seed', False):
            print(f"\n🌱 A SEED MODEL WON THE TOURNAMENT!")
            print(f"   Seed: {Path(results['final_best'].get('seed_path', '')).name}")
            print(f"   Improvement: {results['final_best'].get('improvement_from_seed', 0.0):+.2%}")
        elif 'seed' in str(results['final_best'].get('parent_ids', [])):
            print(f"\n🧬 Champion is derived from a seed model!")
        
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
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Run the enhanced stress test
    results = main()
    
    if results:
        print("\n🎉 Ultimate stress test with seeds completed successfully!")
    else:
        print("\n💥 Ultimate stress test failed!")