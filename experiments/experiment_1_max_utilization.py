#!/usr/bin/env python3
"""
Experiment 1: Maximum GPU Utilization System

This implementation maximizes GPU utilization by:
1. Memory-aware batch sizing for each GPU
2. Multi-experiment queue management
3. Automatic parameter sweep generation
4. Real-time resource monitoring
5. Intelligent load balancing

Designed to keep both RTX 2060s constantly busy with experiments.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import threading
import queue
from datetime import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import itertools

sys.path.append('.')
from src.structure_net import create_multi_scale_network

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    gpu_id: int
    experiment_id: int
    epochs: int
    batch_size: int
    learning_rate: float
    sparsity: float
    hidden_sizes: List[int]
    activation: str
    dataset_size: int
    random_seed: int
    priority: int = 1  # Higher priority = run first
    
    def __lt__(self, other):
        """Less than comparison for priority queue."""
        return self.experiment_id < other.experiment_id
    
    def __eq__(self, other):
        """Equality comparison."""
        return self.experiment_id == other.experiment_id

@dataclass
class GPUInfo:
    """Information about a GPU."""
    gpu_id: int
    name: str
    total_memory: int  # MB
    available_memory: int  # MB
    max_batch_size: int
    is_available: bool

class ResourceMonitor:
    """Monitors GPU resources and determines optimal configurations."""
    
    def __init__(self):
        self.gpu_info = {}
        self._update_gpu_info()
    
    def _update_gpu_info(self):
        """Update GPU information."""
        if not torch.cuda.is_available():
            return
        
        for gpu_id in range(torch.cuda.device_count()):
            try:
                # Test if GPU is usable
                torch.cuda.set_device(gpu_id)
                test_tensor = torch.randn(10, 10).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                
                # Get memory info
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory // (1024**2)  # MB
                allocated = torch.cuda.memory_allocated(gpu_id) // (1024**2)
                available = total_memory - allocated
                
                # Estimate max batch size based on memory
                # RTX 2060 SUPER (8GB) can handle ~128 batch size
                # RTX 2060 (6GB) can handle ~96 batch size
                if total_memory > 7000:  # 8GB GPU
                    max_batch_size = 128
                elif total_memory > 5000:  # 6GB GPU
                    max_batch_size = 96
                else:
                    max_batch_size = 64
                
                self.gpu_info[gpu_id] = GPUInfo(
                    gpu_id=gpu_id,
                    name=torch.cuda.get_device_name(gpu_id),
                    total_memory=total_memory,
                    available_memory=available,
                    max_batch_size=max_batch_size,
                    is_available=True
                )
                
            except Exception as e:
                print(f"⚠️  GPU {gpu_id} not available: {e}")
                self.gpu_info[gpu_id] = GPUInfo(
                    gpu_id=gpu_id,
                    name=f"GPU {gpu_id} (unavailable)",
                    total_memory=0,
                    available_memory=0,
                    max_batch_size=32,
                    is_available=False
                )
    
    def get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs."""
        return [gpu_id for gpu_id, info in self.gpu_info.items() if info.is_available]
    
    def get_optimal_batch_size(self, gpu_id: int, base_batch_size: int = 64) -> int:
        """Get optimal batch size for a GPU."""
        if gpu_id not in self.gpu_info:
            return base_batch_size
        
        gpu_info = self.gpu_info[gpu_id]
        if not gpu_info.is_available:
            return base_batch_size
        
        # Use 80% of max batch size to leave room for other processes
        optimal = int(gpu_info.max_batch_size * 0.8)
        return max(min(optimal, base_batch_size * 2), base_batch_size // 2)
    
    def print_gpu_status(self):
        """Print current GPU status."""
        print("🖥️  GPU Status:")
        for gpu_id, info in self.gpu_info.items():
            status = "✅" if info.is_available else "❌"
            print(f"   {status} GPU {gpu_id}: {info.name}")
            if info.is_available:
                print(f"      Memory: {info.available_memory}/{info.total_memory} MB")
                print(f"      Max batch size: {info.max_batch_size}")

class ExperimentGenerator:
    """Generates experiment configurations for parameter sweeps."""
    
    def __init__(self, base_config: Dict):
        self.base_config = base_config
    
    def generate_parameter_sweep(self, num_experiments: int = 20) -> List[ExperimentConfig]:
        """Generate a parameter sweep of experiments."""
        experiments = []
        
        # Define parameter ranges
        learning_rates = [0.001, 0.002, 0.003, 0.005]
        sparsities = [0.0001, 0.0002, 0.0005, 0.001]
        hidden_architectures = [
            [256, 128],
            [512, 256, 128],
            [384, 192],
            [768, 384, 192]
        ]
        activations = ['tanh', 'sigmoid']
        dataset_sizes = [2000, 3000, 4000]
        epochs_options = [30, 50, 75]
        
        # Generate all combinations
        param_combinations = list(itertools.product(
            learning_rates, sparsities, hidden_architectures, 
            activations, dataset_sizes, epochs_options
        ))
        
        # Shuffle and take requested number
        np.random.shuffle(param_combinations)
        param_combinations = param_combinations[:num_experiments]
        
        for i, (lr, sparsity, hidden, activation, dataset_size, epochs) in enumerate(param_combinations):
            config = ExperimentConfig(
                gpu_id=-1,  # Will be assigned later
                experiment_id=i,
                epochs=epochs,
                batch_size=64,  # Will be optimized per GPU
                learning_rate=lr,
                sparsity=sparsity,
                hidden_sizes=hidden,
                activation=activation,
                dataset_size=dataset_size,
                random_seed=42 + i * 100,
                priority=1
            )
            experiments.append(config)
        
        return experiments
    
    def generate_focused_sweep(self, focus_area: str = "growth") -> List[ExperimentConfig]:
        """Generate experiments focused on specific aspects."""
        experiments = []
        
        if focus_area == "growth":
            # Focus on growth-related parameters
            learning_rates = [0.001, 0.002, 0.004]
            sparsities = [0.0001, 0.0005, 0.001]  # Different initial connectivity
            epochs_options = [50, 75, 100]  # Longer for more growth
            
        elif focus_area == "architecture":
            # Focus on architecture variations
            learning_rates = [0.002]  # Fixed LR
            sparsities = [0.0001]  # Fixed sparsity
            epochs_options = [50]  # Fixed epochs
            
        elif focus_area == "performance":
            # Focus on performance optimization
            learning_rates = [0.001, 0.002, 0.003, 0.005, 0.01]
            sparsities = [0.0001]  # Fixed sparsity
            epochs_options = [30, 50]  # Shorter for quick feedback
        
        # Generate combinations based on focus
        hidden_architectures = [
            [256, 128],
            [512, 256, 128],
            [384, 192],
            [768, 384, 192],
            [1024, 512, 256]
        ]
        
        param_combinations = list(itertools.product(
            learning_rates, sparsities, hidden_architectures, epochs_options
        ))
        
        for i, (lr, sparsity, hidden, epochs) in enumerate(param_combinations):
            config = ExperimentConfig(
                gpu_id=-1,
                experiment_id=i,
                epochs=epochs,
                batch_size=64,
                learning_rate=lr,
                sparsity=sparsity,
                hidden_sizes=hidden,
                activation='tanh',
                dataset_size=3000,
                random_seed=42 + i * 100,
                priority=2 if focus_area == "performance" else 1
            )
            experiments.append(config)
        
        return experiments

class ExperimentQueue:
    """Manages experiment queue and scheduling."""
    
    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self.experiment_queue = queue.PriorityQueue()
        self.running_experiments = {}
        self.completed_experiments = []
        self.failed_experiments = []
        self.total_experiments = 0
        self.lock = threading.Lock()
    
    def add_experiments(self, experiments: List[ExperimentConfig]):
        """Add experiments to the queue."""
        for exp in experiments:
            # Priority queue uses (priority, item) tuples
            # Lower priority number = higher priority
            self.experiment_queue.put((-exp.priority, exp))
            self.total_experiments += 1
        
        print(f"📋 Added {len(experiments)} experiments to queue")
        print(f"📊 Total experiments in queue: {self.experiment_queue.qsize()}")
    
    def get_next_experiment(self, gpu_id: int) -> Optional[ExperimentConfig]:
        """Get next experiment for a specific GPU."""
        try:
            if self.experiment_queue.empty():
                return None
            
            priority, config = self.experiment_queue.get_nowait()
            
            # Assign GPU and optimize batch size
            config.gpu_id = gpu_id
            config.batch_size = self.resource_monitor.get_optimal_batch_size(
                gpu_id, config.batch_size
            )
            
            with self.lock:
                self.running_experiments[f"{gpu_id}_{config.experiment_id}"] = config
            
            return config
            
        except queue.Empty:
            return None
    
    def mark_completed(self, config: ExperimentConfig, result: Dict):
        """Mark experiment as completed."""
        key = f"{config.gpu_id}_{config.experiment_id}"
        
        with self.lock:
            if key in self.running_experiments:
                del self.running_experiments[key]
            
            result['config'] = config
            self.completed_experiments.append(result)
        
        print(f"✅ Completed experiment {config.experiment_id} on GPU {config.gpu_id}")
        self._print_progress()
    
    def mark_failed(self, config: ExperimentConfig, error: str):
        """Mark experiment as failed."""
        key = f"{config.gpu_id}_{config.experiment_id}"
        
        with self.lock:
            if key in self.running_experiments:
                del self.running_experiments[key]
            
            self.failed_experiments.append({
                'config': config,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
        
        print(f"❌ Failed experiment {config.experiment_id} on GPU {config.gpu_id}: {error}")
        self._print_progress()
    
    def _print_progress(self):
        """Print current progress."""
        completed = len(self.completed_experiments)
        failed = len(self.failed_experiments)
        running = len(self.running_experiments)
        remaining = self.experiment_queue.qsize()
        
        print(f"📈 Progress: {completed} completed, {failed} failed, {running} running, {remaining} queued")
    
    def get_status(self) -> Dict:
        """Get current queue status."""
        return {
            'total_experiments': self.total_experiments,
            'completed': len(self.completed_experiments),
            'failed': len(self.failed_experiments),
            'running': len(self.running_experiments),
            'queued': self.experiment_queue.qsize(),
            'completion_rate': len(self.completed_experiments) / self.total_experiments if self.total_experiments > 0 else 0
        }

class MaxUtilizationExperiment:
    """Single experiment runner optimized for maximum utilization."""
    
    def __init__(self, config: ExperimentConfig, base_save_dir: str = "max_utilization_results"):
        self.config = config
        self.save_dir = os.path.join(base_save_dir, f"gpu_{config.gpu_id}_exp_{config.experiment_id}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Set device and random seed
        self.device = torch.device(f'cuda:{config.gpu_id}')
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Experiment tracking
        self.training_log = []
        self.growth_events = []
        self.performance_history = []
    
    def create_optimized_dataset(self):
        """Create dataset optimized for this experiment configuration."""
        patterns = []
        labels = []
        n_classes = 10
        samples_per_class = self.config.dataset_size // n_classes
        
        # Use config-specific random seed for reproducible but unique datasets
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        for class_idx in range(n_classes):
            for i in range(samples_per_class):
                # Create patterns with complexity based on experiment config
                complexity_factor = len(self.config.hidden_sizes) / 3.0  # More layers = more complexity
                base_pattern = torch.randn(784) * (0.8 + 0.2 * complexity_factor)
                
                # Add class-specific structure
                freq = (class_idx + 1) * 0.1 * (1 + self.config.learning_rate * 100)
                indices = torch.arange(784).float()
                
                if class_idx < 3:
                    base_pattern += torch.sin(indices * freq) * 2
                elif class_idx < 6:
                    base_pattern += torch.cos(indices * freq) * 2
                else:
                    base_pattern += torch.sin(indices * freq * 0.5) * torch.cos(indices * freq * 2) * 1.5
                
                # Add noise based on sparsity (lower sparsity = more noise)
                noise_scale = 0.5 + self.config.sparsity * 1000
                noise = torch.randn(784) * noise_scale
                pattern = base_pattern + noise
                
                patterns.append(pattern)
                labels.append(class_idx)
        
        X = torch.stack(patterns).to(self.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Shuffle
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        
        return X, y
    
    def run_experiment(self) -> Dict:
        """Run the experiment with the given configuration."""
        start_time = time.time()
        
        try:
            print(f"🚀 GPU {self.config.gpu_id} Exp {self.config.experiment_id}: Starting")
            print(f"   Config: LR={self.config.learning_rate}, Sparsity={self.config.sparsity}, "
                  f"Arch={self.config.hidden_sizes}, Batch={self.config.batch_size}")
            
            # Create network
            network = create_multi_scale_network(
                784, self.config.hidden_sizes, 10,
                sparsity=self.config.sparsity,
                activation=self.config.activation,
                device=self.device,
                snapshot_dir=os.path.join(self.save_dir, "snapshots")
            )
            
            initial_stats = network.network.get_connectivity_stats()
            
            # Create dataset
            X, y = self.create_optimized_dataset()
            
            # Create data loaders
            dataset = torch.utils.data.TensorDataset(X, y)
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, test_size],
                generator=torch.Generator().manual_seed(self.config.random_seed)
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.config.batch_size, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.config.batch_size, shuffle=False
            )
            
            # Training setup
            optimizer = optim.Adam(network.parameters(), lr=self.config.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            best_performance = 0
            total_growth_events = 0
            
            for epoch in range(self.config.epochs):
                # Training epoch
                epoch_stats = network.train_epoch(train_loader, optimizer, criterion, epoch)
                
                # Evaluation
                eval_stats = network.evaluate(test_loader, criterion)
                
                # Track performance
                self.performance_history.append({
                    'epoch': epoch,
                    'train_loss': epoch_stats['loss'],
                    'train_acc': epoch_stats['performance'],
                    'test_loss': eval_stats['loss'],
                    'test_acc': eval_stats['performance'],
                    'connections': epoch_stats['total_connections']
                })
                
                # Monitor growth
                if epoch_stats.get('growth_events', 0) > 0:
                    total_growth_events += 1
                    self.growth_events.append({
                        'epoch': epoch,
                        'connections_added': epoch_stats.get('connections_added', 0),
                        'total_connections': epoch_stats.get('total_connections', 0),
                        'performance': eval_stats['performance']
                    })
                
                # Update best performance
                if eval_stats['performance'] > best_performance:
                    best_performance = eval_stats['performance']
                
                # Progress reporting (every 10 epochs)
                if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                    print(f"   GPU {self.config.gpu_id} Exp {self.config.experiment_id} "
                          f"Epoch {epoch:3d}/{self.config.epochs}: "
                          f"Acc={eval_stats['performance']:.3f}, "
                          f"Conn={epoch_stats['total_connections']}")
            
            total_time = time.time() - start_time
            final_stats = network.network.get_connectivity_stats()
            
            # Save results
            self._save_results(network, total_time, best_performance)
            
            # Return summary
            return {
                'experiment_id': self.config.experiment_id,
                'gpu_id': self.config.gpu_id,
                'config': self.config,
                'total_time': total_time,
                'best_performance': best_performance,
                'total_growth_events': total_growth_events,
                'final_connections': final_stats['total_active_connections'],
                'connection_growth': final_stats['total_active_connections'] - initial_stats['total_active_connections'],
                'success': True
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ GPU {self.config.gpu_id} Exp {self.config.experiment_id} failed: {e}")
            return {
                'experiment_id': self.config.experiment_id,
                'gpu_id': self.config.gpu_id,
                'config': self.config,
                'total_time': total_time,
                'error': str(e),
                'success': False
            }
    
    def _save_results(self, network, total_time, best_performance):
        """Save experiment results."""
        results = {
            'config': {
                'experiment_id': self.config.experiment_id,
                'gpu_id': self.config.gpu_id,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'sparsity': self.config.sparsity,
                'hidden_sizes': self.config.hidden_sizes,
                'activation': self.config.activation,
                'dataset_size': self.config.dataset_size,
                'random_seed': self.config.random_seed
            },
            'results': {
                'total_time': total_time,
                'best_performance': best_performance,
                'total_growth_events': len(self.growth_events),
                'final_connections': network.network.get_connectivity_stats()['total_active_connections']
            },
            'performance_history': self.performance_history,
            'growth_events': self.growth_events,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.save_dir, "experiment_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)

def run_single_experiment_max_util(config: ExperimentConfig) -> Dict:
    """Run a single experiment with maximum utilization."""
    try:
        torch.cuda.set_device(config.gpu_id)
        experiment = MaxUtilizationExperiment(config)
        return experiment.run_experiment()
    except Exception as e:
        return {
            'experiment_id': config.experiment_id,
            'gpu_id': config.gpu_id,
            'config': config,
            'error': str(e),
            'success': False
        }

class MaxUtilizationManager:
    """Main manager for maximum GPU utilization experiments."""
    
    def __init__(self, base_save_dir: str = "max_utilization_results"):
        self.base_save_dir = base_save_dir
        os.makedirs(base_save_dir, exist_ok=True)
        
        self.resource_monitor = ResourceMonitor()
        self.experiment_queue = ExperimentQueue(self.resource_monitor)
        self.experiment_generator = ExperimentGenerator({})
        
        self.results = []
        self.start_time = None
    
    def run_maximum_utilization(
        self,
        experiment_type: str = "parameter_sweep",
        num_experiments: int = 50,
        max_workers: int = None
    ):
        """Run experiments with maximum GPU utilization."""
        print("🚀 Starting Maximum GPU Utilization System")
        print("=" * 80)
        
        # Check available GPUs
        self.resource_monitor.print_gpu_status()
        available_gpus = self.resource_monitor.get_available_gpus()
        
        if not available_gpus:
            print("❌ No available GPUs found!")
            return
        
        # Filter to only use RTX 2060s (avoid RTX 6000)
        rtx_2060_gpus = []
        for gpu_id in available_gpus:
            gpu_name = self.resource_monitor.gpu_info[gpu_id].name
            if "RTX 2060" in gpu_name and "6000" not in gpu_name:
                rtx_2060_gpus.append(gpu_id)
        
        if not rtx_2060_gpus:
            print("❌ No RTX 2060 GPUs found!")
            return
        
        print(f"🎯 Using RTX 2060 GPUs: {rtx_2060_gpus}")
        
        # Generate experiments
        if experiment_type == "parameter_sweep":
            experiments = self.experiment_generator.generate_parameter_sweep(num_experiments)
        elif experiment_type == "growth_focused":
            experiments = self.experiment_generator.generate_focused_sweep("growth")
        elif experiment_type == "architecture_focused":
            experiments = self.experiment_generator.generate_focused_sweep("architecture")
        elif experiment_type == "performance_focused":
            experiments = self.experiment_generator.generate_focused_sweep("performance")
        else:
            experiments = self.experiment_generator.generate_parameter_sweep(num_experiments)
        
        print(f"📋 Generated {len(experiments)} experiments")
        
        # Add to queue
        self.experiment_queue.add_experiments(experiments)
        
        # Set max workers (one per GPU)
        if max_workers is None:
            max_workers = len(rtx_2060_gpus)
        
        print(f"🔄 Starting {max_workers} parallel workers")
        print("=" * 80)
        
        self.start_time = time.time()
        
        # Run experiments with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit initial experiments
            futures = {}
            
            # Keep submitting experiments until queue is empty
            while True:
                # Submit new experiments for available workers
                for gpu_id in rtx_2060_gpus:
                    if len([f for f in futures.keys() if not f.done()]) < max_workers:
                        config = self.experiment_queue.get_next_experiment(gpu_id)
                        if config:
                            future = executor.submit(run_single_experiment_max_util, config)
                            futures[future] = config
                
                # Check for completed experiments
                completed_futures = [f for f in futures.keys() if f.done()]
                
                for future in completed_futures:
                    config = futures[future]
                    try:
                        result = future.result()
                        if result['success']:
                            self.experiment_queue.mark_completed(config, result)
                        else:
                            self.experiment_queue.mark_failed(config, result.get('error', 'Unknown error'))
                        
                        self.results.append(result)
                        
                    except Exception as e:
                        self.experiment_queue.mark_failed(config, str(e))
                    
                    del futures[future]
                
                # Check if we're done
                status = self.experiment_queue.get_status()
                if status['queued'] == 0 and len(futures) == 0:
                    break
                
                # Brief pause to avoid busy waiting
                time.sleep(1)
        
        total_time = time.time() - self.start_time
        
        # Final analysis
        self._analyze_results(total_time)
    
    def _analyze_results(self, total_time: float):
        """Analyze and save final results."""
        print("\n" + "=" * 80)
        print("📊 MAXIMUM UTILIZATION EXPERIMENT ANALYSIS")
        print("=" * 80)
        
        successful_results = [r for r in self.results if r['success']]
        failed_results = [r for r in self.results if not r['success']]
        
        print(f"✅ Successful experiments: {len(successful_results)}")
        print(f"❌ Failed experiments: {len(failed_results)}")
        print(f"⏱️  Total wall-clock time: {total_time:.2f} seconds")
        
        if successful_results:
            # Calculate statistics
            performances = [r['best_performance'] for r in successful_results]
            times = [r['total_time'] for r in successful_results]
            growth_events = [r['total_growth_events'] for r in successful_results]
            
            total_compute_time = sum(times)
            avg_performance = np.mean(performances)
            std_performance = np.std(performances)
            max_performance = max(performances)
            
            speedup = total_compute_time / total_time
            
            print(f"⚡ Speedup achieved: {speedup:.2f}x")
            print(f"📈 Performance statistics:")
            print(f"   Average accuracy: {avg_performance:.4f} ± {std_performance:.4f}")
            print(f"   Best accuracy: {max_performance:.4f}")
            print(f"   Total compute time: {total_compute_time:.2f} seconds")
            print(f"🌱 Average growth events: {np.mean(growth_events):.1f}")
            
            # Find best configurations
            best_result = max(successful_results, key=lambda x: x['best_performance'])
            print(f"\n🏆 Best performing configuration:")
            print(f"   Experiment ID: {best_result['experiment_id']}")
            print(f"   Performance: {best_result['best_performance']:.4f}")
            print(f"   Learning rate: {best_result['config'].learning_rate}")
            print(f"   Sparsity: {best_result['config'].sparsity}")
            print(f"   Architecture: {best_result['config'].hidden_sizes}")
        
        # Save aggregated results
        aggregated_results = {
            'summary': {
                'total_experiments': len(self.results),
                'successful_experiments': len(successful_results),
                'failed_experiments': len(failed_results),
                'total_wall_clock_time': total_time,
                'total_compute_time': sum(r['total_time'] for r in successful_results),
                'speedup_achieved': sum(r['total_time'] for r in successful_results) / total_time if total_time > 0 else 0,
                'performance_stats': {
                    'mean': np.mean([r['best_performance'] for r in successful_results]) if successful_results else 0,
                    'std': np.std([r['best_performance'] for r in successful_results]) if successful_results else 0,
                    'max': max([r['best_performance'] for r in successful_results]) if successful_results else 0,
                    'min': min([r['best_performance'] for r in successful_results]) if successful_results else 0
                }
            },
            'all_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.base_save_dir, "aggregated_results.json"), 'w') as f:
            json.dump(aggregated_results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {self.base_save_dir}/")
        print("🎉 Maximum utilization experiment completed!")

def main():
    """Main function for maximum utilization experiments."""
    parser = argparse.ArgumentParser(description='Maximum GPU Utilization Experiment System')
    parser.add_argument('--experiment-type', type=str, default='parameter_sweep',
                       choices=['parameter_sweep', 'growth_focused', 'architecture_focused', 'performance_focused'],
                       help='Type of experiment to run (default: parameter_sweep)')
    parser.add_argument('--num-experiments', type=int, default=50,
                       help='Number of experiments to generate (default: 50)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: number of RTX 2060 GPUs)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Maximum utilization requires CUDA.")
        return
    
    print(f"🚀 Starting Maximum GPU Utilization System")
    print(f"📊 Configuration:")
    print(f"   Experiment type: {args.experiment_type}")
    print(f"   Number of experiments: {args.num_experiments}")
    print(f"   Max workers: {args.max_workers or 'auto'}")
    
    # Create and run maximum utilization manager
    manager = MaxUtilizationManager()
    manager.run_maximum_utilization(
        experiment_type=args.experiment_type,
        num_experiments=args.num_experiments,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()
