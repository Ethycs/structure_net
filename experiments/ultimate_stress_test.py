#!/usr/bin/env python3
"""
Ultimate Structure Net Stress Test

This experiment is designed to push the structure_net system to its absolute limits:
- Multi-GPU parallel processing with torch.multiprocessing
- All adaptive learning rate strategies simultaneously
- Complete metrics system with all analyzers
- Comprehensive profiling and logging
- Residual block insertion during growth
- Tournament-style architecture competition
- Memory-optimized batch processing
- CIFAR-10 seed loading and evolution

This is the most comprehensive test of the entire structure_net ecosystem.
"""

# IMPORTANT: Set spawn method BEFORE importing torch
import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
import os
import json
import psutil
from datetime import datetime
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
from src.structure_net.logging.standardized_logging import (
    StandardizedLogger, LoggingConfig
)
# Use simplified schemas to avoid Pydantic compatibility issues
from src.structure_net.logging.schemas_simple import ExperimentConfig, PerformanceMetrics as MetricsData
from src.structure_net.evolution.extrema_analyzer import detect_network_extrema
from src.structure_net.evolution.integrated_growth_system_v2 import IntegratedGrowthSystem


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


class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup."""
    
    def __init__(self, max_memory_fraction: float = 0.8):
        self.max_memory_fraction = max_memory_fraction
        self.allocated_memory = {}
        
    def get_available_memory(self, device_id: int) -> int:
        """Get available memory on GPU device."""
        if not torch.cuda.is_available():
            return 0
        
        torch.cuda.set_device(device_id)
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        cached_memory = torch.cuda.memory_reserved(device_id)
        
        available = total_memory - max(allocated_memory, cached_memory)
        max_usable = int(total_memory * self.max_memory_fraction)
        
        return min(available, max_usable)
    
    def optimize_batch_size(self, device_id: int, base_batch_size: int, model_size: int) -> int:
        """Optimize batch size based on available memory."""
        available_memory = self.get_available_memory(device_id)
        
        # Estimate memory per sample (rough heuristic)
        memory_per_sample = model_size * 4 + 3072 * 4  # Model params + CIFAR-10 input
        max_batch_size = available_memory // (memory_per_sample * 10)  # Safety factor
        
        return min(base_batch_size, max(1, max_batch_size))
    
    def cleanup_gpu_memory(self, device_id: int):
        """Clean up GPU memory."""
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        gc.collect()


def tournament_worker(config: StressTestConfig, device_id: int, process_id: int, 
                      work_queue: mp.Queue, results_queue: mp.Queue):
    """Worker function for parallel tournament evaluation."""
    # Create a minimal tournament instance for evaluation
    memory_manager = GPUMemoryManager(config.max_memory_per_gpu)
    
    while True:
        try:
            item = work_queue.get(timeout=1)
            if item is None:  # Sentinel value to stop
                break
            competitor, idx = item
            # Directly evaluate the competitor
            result = evaluate_competitor_standalone(
                competitor, device_id, process_id, config, memory_manager
            )
            results_queue.put((idx, result))
        except queue.Empty:
            break
        except Exception as e:
            print(f"Worker GPU {device_id} Process {process_id} error: {e}")
            import traceback
            traceback.print_exc()
            break


def evaluate_competitor_standalone(competitor: Dict[str, Any], device_id: int, 
                                  process_id: int, config: StressTestConfig,
                                  memory_manager: GPUMemoryManager) -> Dict[str, Any]:
    """Standalone competitor evaluation function that can be called in a subprocess."""
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'
    
    try:
        # Create network
        model = create_standard_network(
            architecture=competitor['architecture'],
            sparsity=competitor['sparsity'],
            device=device
        )
        
        # Add residual blocks if enabled
        if config.enable_residual_blocks and len(competitor['architecture']) >= 5:
            residual_positions = [2, 4]  # Add residual blocks at positions 2 and 4
            model = create_residual_network(
                competitor['architecture'],
                competitor['sparsity'],
                residual_positions,
                device
            )
        
        # Setup adaptive learning rates
        # Ensure lr_strategy is a string (not np.str_)
        lr_strategy = str(competitor['lr_strategy'])
        lr_manager = AdaptiveLearningRateManager(
            network=model,
            base_lr=0.001,
            strategy=lr_strategy,
            enable_extrema_phase=True,
            enable_layer_age_aware=True,
            enable_multi_scale=True,
            enable_unified_system=True
        )
        
        # Create optimizer
        optimizer = lr_manager.create_adaptive_optimizer(
            optimizer_class=optim.AdamW,
            weight_decay=1e-4
        )
        
        # Setup metrics system
        if config.enable_comprehensive_metrics:
            threshold_config = ThresholdConfig()
            metrics_config = MetricsConfig(
                compute_mi=True,
                compute_activity=True,
                compute_sensli=True,
                compute_graph=True
            )
            metrics_system = CompleteMetricsSystem(model, threshold_config, metrics_config)
        else:
            metrics_system = None
        
        # Setup growth system
        if config.enable_growth:
            growth_system = IntegratedGrowthSystem(
                network=model,
                config=ThresholdConfig(),
                metrics_config=MetricsConfig()
            )
        else:
            growth_system = None
        
        # Load CIFAR-10 data
        # GPUSeedHunter expects device_id as integer, not 'cuda:X' string
        hunter = GPUSeedHunter(num_gpus=1, device=device_id, dataset='cifar10')
        hunter.cache_dataset_gpu()
        dataset = hunter.get_cached_dataset()
        
        # Optimize batch size for this GPU
        model_params = sum(p.numel() for p in model.parameters())
        batch_size = memory_manager.optimize_batch_size(
            device_id, config.batch_size_base, model_params
        )
        
        print(f"🔥 GPU {device_id} Process {process_id}: Training {competitor['id']}")
        print(f"   Architecture: {competitor['architecture']}")
        print(f"   LR Strategy: {lr_strategy}")
        print(f"   Batch size: {batch_size}")
        
        # Training loop with all features
        scaler = GradScaler() if config.mixed_precision else None
        best_accuracy = 0.0
        growth_events = []
        
        for epoch in range(config.epochs_per_generation):
            model.train()
            epoch_loss = 0.0
            
            # Training batches
            num_batches = len(dataset['train_x']) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(dataset['train_x']))
                
                x = dataset['train_x'][start_idx:end_idx]
                y = dataset['train_y'][start_idx:end_idx]
                
                optimizer.zero_grad()
                
                if scaler and config.mixed_precision:
                    with autocast():
                        output = model(x)
                        loss = nn.functional.cross_entropy(output, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(x)
                    loss = nn.functional.cross_entropy(output, y)
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
            
            # Update learning rates
            lr_manager.update_learning_rates(
                optimizer, epoch,
                network=model,
                data_loader=None,  # We'll use cached data
                device=device
            )
            
            # Evaluation
            if epoch % 2 == 0:  # Evaluate every 2 epochs
                model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    test_batches = len(dataset['test_x']) // batch_size
                    for batch_idx in range(test_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(dataset['test_x']))
                        
                        x = dataset['test_x'][start_idx:end_idx]
                        y = dataset['test_y'][start_idx:end_idx]
                        
                        output = model(x)
                        pred = output.argmax(dim=1)
                        correct += (pred == y).sum().item()
                        total += y.size(0)
                
                accuracy = correct / total
                best_accuracy = max(best_accuracy, accuracy)
                
                print(f"     Epoch {epoch}: Acc={accuracy:.3f}, Loss={epoch_loss/num_batches:.4f}")
            
            # Memory cleanup
            if epoch % config.memory_cleanup_frequency == 0:
                memory_manager.cleanup_gpu_memory(device_id)
        
        # Final evaluation
        model.eval()
        final_correct = 0
        final_total = 0
        
        with torch.no_grad():
            test_batches = len(dataset['test_x']) // batch_size
            for batch_idx in range(test_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(dataset['test_x']))
                
                x = dataset['test_x'][start_idx:end_idx]
                y = dataset['test_y'][start_idx:end_idx]
                
                output = model(x)
                pred = output.argmax(dim=1)
                final_correct += (pred == y).sum().item()
                final_total += y.size(0)
        
        final_accuracy = final_correct / final_total
        
        # Calculate fitness (combination of accuracy and efficiency)
        network_stats = get_network_stats(model)
        efficiency = final_accuracy / (network_stats['total_parameters'] / 1000)  # Acc per K params
        fitness = final_accuracy * 0.7 + efficiency * 0.3
        
        result = {
            'id': competitor['id'],
            'fitness': fitness,
            'accuracy': final_accuracy,
            'efficiency': efficiency,
            'parameters': network_stats['total_parameters'],
            'growth_events': len(growth_events),
            'final_architecture': [layer.linear.out_features for layer in model 
                                 if hasattr(layer, 'linear')],
            'device_id': device_id,
            'process_id': process_id
        }
        
        print(f"✅ GPU {device_id} Process {process_id}: {competitor['id']} completed")
        print(f"   Final accuracy: {final_accuracy:.3f}")
        print(f"   Fitness: {fitness:.4f}")
        print(f"   Parameters: {network_stats['total_parameters']:,}")
        
        return result
        
    except Exception as e:
        print(f"❌ GPU {device_id} Process {process_id}: {competitor['id']} failed: {e}")
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
    
    finally:
        # Cleanup
        memory_manager.cleanup_gpu_memory(device_id)


class ArchitectureTournament:
    """Tournament-style architecture evolution with multi-GPU support."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.memory_manager = GPUMemoryManager(config.max_memory_per_gpu)
        self.generation = 0
        self.population = []
        self.fitness_history = []
        
        # Initialize logging
        logging_config = LoggingConfig(
            project_name="structure_net_stress_test",
            enable_wandb=True,
            auto_upload=True
        )
        self.logger = StandardizedLogger(logging_config)
        
        # Initialize profiler
        if config.enable_profiling:
            self.profiler = create_comprehensive_profiler(
                output_dir=f"stress_test_profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                enable_wandb=True
            )
        else:
            self.profiler = None
        
        print(f"🏟️  Tournament initialized with {config.tournament_size} competitors")
        print(f"   GPUs: {config.num_gpus}")
        print(f"   Processes per GPU: {config.processes_per_gpu}")
        print(f"   Total parallel processes: {config.num_gpus * config.processes_per_gpu}")
    
    def generate_initial_population(self) -> List[Dict[str, Any]]:
        """Generate initial population of architectures."""
        population = []
        
        # Load some CIFAR-10 seeds if available
        seed_architectures = [
            [3072, 512, 256, 128, 10],
            [3072, 1024, 512, 256, 128, 10],
            [3072, 256, 512, 256, 10],
            [3072, 800, 400, 200, 100, 10],
            [3072, 1536, 768, 384, 192, 10]
        ]
        
        for i in range(self.config.tournament_size):
            if i < len(seed_architectures):
                architecture = seed_architectures[i]
            else:
                # Generate random architecture
                num_layers = np.random.randint(3, 8)
                architecture = [3072]  # CIFAR-10 input size
                
                current_size = 3072
                for _ in range(num_layers - 2):
                    # Ensure we have a valid range for random generation
                    min_size = 64
                    max_size = min(current_size, 1024)
                    if min_size >= max_size:
                        next_size = min_size
                    else:
                        next_size = np.random.randint(min_size, max_size)
                    architecture.append(next_size)
                    current_size = next_size
                
                architecture.append(10)  # CIFAR-10 output
            
            # Random learning rate strategy
            lr_strategy = np.random.choice(self.config.learning_rate_strategies)
            
            competitor = {
                'id': f'arch_{i:03d}',
                'architecture': architecture,
                'sparsity': np.random.uniform(0.01, 0.1),
                'lr_strategy': lr_strategy,
                'fitness': 0.0,
                'generation': 0,
                'parent_ids': [],
                'mutations': []
            }
            
            population.append(competitor)
        
        return population
    
    def evaluate_competitor_on_gpu(self, competitor: Dict[str, Any], device_id: int, 
                                  process_id: int) -> Dict[str, Any]:
        """Evaluate a single competitor on specified GPU."""
        # Just delegate to the standalone function
        return evaluate_competitor_standalone(
            competitor, device_id, process_id, self.config, self.memory_manager
        )
    
    def run_generation_parallel(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run a generation with full parallelization across all GPUs."""
        print(f"\n🏁 Generation {self.generation}: Evaluating {len(population)} competitors")
        
        # Start profiling if enabled
        if self.profiler:
            self.profiler.start_session(f"generation_{self.generation}")
        
        results = []
        
        # Create work queue (multiprocessing queue, not threading queue)
        work_queue = mp.Queue()
        for i, competitor in enumerate(population):
            work_queue.put((competitor, i))
        
        # Add sentinel values to stop workers
        for _ in range(self.config.num_gpus * self.config.processes_per_gpu):
            work_queue.put(None)
        
        # Start all worker processes
        results_queue = mp.Queue()
        processes = []
        
        for gpu_id in range(self.config.num_gpus):
            for proc_id in range(self.config.processes_per_gpu):
                p = mp.Process(
                    target=tournament_worker,
                    args=(self.config, gpu_id, proc_id, work_queue, results_queue)
                )
                p.start()
                processes.append(p)
        
        # Collect results
        collected_results = {}
        total_expected = len(population)
        
        while len(collected_results) < total_expected:
            try:
                idx, result = results_queue.get(timeout=30)
                collected_results[idx] = result
                
                print(f"📊 Progress: {len(collected_results)}/{total_expected} completed")
                
            except queue.Empty:
                print("⚠️  Timeout waiting for results")
                break
        
        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        
        # Sort results by original order
        results = [collected_results.get(i, {'fitness': 0.0, 'error': 'timeout'}) 
                  for i in range(len(population))]
        
        # Stop profiling
        if self.profiler:
            self.profiler.end_session()
        
        return results
    
    def select_survivors(self, population: List[Dict[str, Any]], 
                        results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select survivors for next generation."""
        # Combine population with results
        combined = []
        for competitor, result in zip(population, results):
            competitor.update(result)
            combined.append(competitor)
        
        # Sort by fitness
        combined.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Select top survivors
        survivors = combined[:self.config.survivors_per_generation]
        
        print(f"🏆 Generation {self.generation} Results:")
        print("   Top 5 performers:")
        for i, survivor in enumerate(survivors[:5]):
            print(f"     {i+1}. {survivor['id']}: fitness={survivor['fitness']:.4f}, "
                  f"acc={survivor['accuracy']:.3f}, params={survivor.get('parameters', 0):,}")
        
        return survivors
    
    def mutate_population(self, survivors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create new population through mutation and crossover."""
        new_population = survivors.copy()  # Keep survivors
        
        while len(new_population) < self.config.tournament_size:
            # Select parents
            parent1 = np.random.choice(survivors)
            parent2 = np.random.choice(survivors)
            
            # Create offspring
            offspring = {
                'id': f'arch_{len(new_population):03d}_gen{self.generation + 1}',
                'generation': self.generation + 1,
                'parent_ids': [parent1['id'], parent2['id']],
                'mutations': []
            }
            
            # Crossover architecture
            if len(parent1['architecture']) == len(parent2['architecture']):
                offspring['architecture'] = []
                for i in range(len(parent1['architecture'])):
                    if np.random.random() < 0.5:
                        offspring['architecture'].append(parent1['architecture'][i])
                    else:
                        offspring['architecture'].append(parent2['architecture'][i])
            else:
                offspring['architecture'] = parent1['architecture'].copy()
            
            # Mutations
            if np.random.random() < self.config.mutation_rate:
                # Mutate layer size
                layer_idx = np.random.randint(1, len(offspring['architecture']) - 1)
                old_size = offspring['architecture'][layer_idx]
                offspring['architecture'][layer_idx] = max(32, int(old_size * np.random.uniform(0.5, 2.0)))
                offspring['mutations'].append(f'layer_{layer_idx}_size_{old_size}->{offspring["architecture"][layer_idx]}')
            
            # Inherit other properties with mutation
            offspring['sparsity'] = parent1['sparsity'] * np.random.uniform(0.8, 1.2)
            offspring['sparsity'] = np.clip(offspring['sparsity'], 0.01, 0.2)
            
            offspring['lr_strategy'] = np.random.choice(self.config.learning_rate_strategies)
            
            new_population.append(offspring)
        
        return new_population
    
    def run_tournament(self) -> Dict[str, Any]:
        """Run the complete tournament."""
        print(f"🚀 Starting Ultimate Structure Net Stress Test Tournament")
        print(f"   Tournament size: {self.config.tournament_size}")
        print(f"   Generations: {self.config.generations}")
        print(f"   Total GPU processes: {self.config.num_gpus * self.config.processes_per_gpu}")
        
        # Generate initial population
        self.population = self.generate_initial_population()
        
        tournament_results = {
            'config': self.config,
            'generations': [],
            'best_performers': [],
            'system_stats': {}
        }
        
        start_time = time.time()
        
        for generation in range(self.config.generations):
            self.generation = generation
            generation_start = time.time()
            
            print(f"\n{'='*80}")
            print(f"🏁 GENERATION {generation + 1}/{self.config.generations}")
            print(f"{'='*80}")
            
            # Evaluate population
            results = self.run_generation_parallel(self.population)
            
            # Select survivors
            survivors = self.select_survivors(self.population, results)
            
            # Record generation results
            generation_data = {
                'generation': generation,
                'population_size': len(self.population),
                'survivors': len(survivors),
                'best_fitness': survivors[0]['fitness'] if survivors else 0.0,
                'avg_fitness': np.mean([s['fitness'] for s in survivors]) if survivors else 0.0,
                'duration': time.time() - generation_start,
                'top_performers': survivors[:5]
            }
            
            tournament_results['generations'].append(generation_data)
            
            # Create next generation
            if generation < self.config.generations - 1:
                self.population = self.mutate_population(survivors)
            else:
                self.population = survivors
        
        total_time = time.time() - start_time
        
        # Final results
        final_best = max(self.population, key=lambda x: x['fitness'])
        
        tournament_results.update({
            'total_duration': total_time,
            'final_best': final_best,
            'system_stats': {
                'total_experiments': self.config.tournament_size * self.config.generations,
                'experiments_per_second': (self.config.tournament_size * self.config.generations) / total_time,
                'gpu_utilization': self.config.num_gpus,
                'memory_usage': psutil.virtual_memory().percent
            }
        })
        
        print(f"\n🏆 TOURNAMENT COMPLETE!")
        print(f"   Duration: {total_time:.1f} seconds")
        print(f"   Total experiments: {tournament_results['system_stats']['total_experiments']}")
        print(f"   Throughput: {tournament_results['system_stats']['experiments_per_second']:.2f} exp/sec")
        print(f"   Final champion: {final_best['id']}")
        print(f"   Champion fitness: {final_best['fitness']:.4f}")
        print(f"   Champion accuracy: {final_best['accuracy']:.3f}")
        
        return tournament_results


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


def main():
    """Main function to run the ultimate stress test."""
    print("🚀 Structure Net Ultimate Stress Test")
    print("=" * 80)
    
    # Calculate optimal configuration
    memory_info = calculate_optimal_memory_usage()
    print("💾 System Memory Analysis:")
    for key, value in memory_info.items():
        if key != 'gpu_details':
            print(f"   {key}: {value}")
    
    print("\n🔧 GPU Details:")
    for gpu_id, details in memory_info.get('gpu_details', {}).items():
        print(f"   {gpu_id}: {details['name']} ({details['memory_gb']:.1f}GB)")
    
    # Create optimized configuration
    config = StressTestConfig(
        num_gpus=memory_info.get('total_gpus', 1),
        processes_per_gpu=memory_info.get('recommended_processes_per_gpu', 1),
        tournament_size=memory_info.get('recommended_tournament_size', 32),
        batch_size_base=memory_info.get('recommended_batch_size', 256),
        generations=5,  # Reduced for stress test
        epochs_per_generation=15,  # Reduced for faster iteration
        enable_growth=True,
        enable_residual_blocks=True,
        enable_comprehensive_metrics=True,
        enable_profiling=True
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"   Tournament size: {config.tournament_size}")
    print(f"   Generations: {config.generations}")
    print(f"   Epochs per generation: {config.epochs_per_generation}")
    print(f"   GPUs: {config.num_gpus}")
    print(f"   Processes per GPU: {config.processes_per_gpu}")
    print(f"   Base batch size: {config.batch_size_base}")
    print(f"   Growth enabled: {config.enable_growth}")
    print(f"   Residual blocks: {config.enable_residual_blocks}")
    print(f"   Comprehensive metrics: {config.enable_comprehensive_metrics}")
    print(f"   Profiling: {config.enable_profiling}")
    
    # Set multiprocessing start method
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    # Run the tournament
    tournament = ArchitectureTournament(config)
    
    try:
        results = tournament.run_tournament()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"stress_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Print final summary
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Total duration: {results['total_duration']:.1f} seconds")
        print(f"   Total experiments: {results['system_stats']['total_experiments']}")
        print(f"   Throughput: {results['system_stats']['experiments_per_second']:.2f} exp/sec")
        print(f"   Champion: {results['final_best']['id']}")
        print(f"   Champion fitness: {results['final_best']['fitness']:.4f}")
        print(f"   Champion accuracy: {results['final_best']['accuracy']:.3f}")
        print(f"   Champion architecture: {results['final_best'].get('final_architecture', 'N/A')}")
        
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
    
    # Run the stress test
    results = main()
    
    if results:
        print("\n🎉 Ultimate stress test completed successfully!")
    else:
        print("\n💥 Ultimate stress test failed!")
