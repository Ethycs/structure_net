#!/usr/bin/env python3
"""
Experiment 1: Multi-Scale Snapshots - Parallel GPU Implementation

This implementation runs independent experiments on both RTX 2060 SUPER GPUs simultaneously
to accelerate the overall job completion. Each GPU runs a separate experiment with:

1. Independent datasets and random seeds
2. Separate growth trajectories and snapshots
3. Individual result tracking and saving
4. Parallel execution for 2x speedup
5. Aggregated final results and comparisons

This allows for faster experimentation and statistical analysis across multiple runs.
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
from datetime import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor
import threading

sys.path.append('.')
from src.structure_net import create_multi_scale_network

class ParallelExperiment1:
    """
    Parallel implementation of Experiment 1: Multi-Scale Snapshots
    
    Runs independent experiments on multiple GPUs simultaneously
    for accelerated job completion and statistical analysis.
    """
    
    def __init__(self, gpu_id, experiment_id, base_save_dir="experiment_1_parallel_results"):
        self.gpu_id = gpu_id
        self.experiment_id = experiment_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.save_dir = os.path.join(base_save_dir, f"gpu_{gpu_id}_exp_{experiment_id}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Experiment tracking
        self.training_log = []
        self.growth_events = []
        self.extrema_evolution = []
        self.snapshot_timeline = []
        self.performance_history = []
        self.last_snapshot_epoch = -1
        
        # Set unique random seed for this GPU/experiment
        self.random_seed = 42 + gpu_id * 1000 + experiment_id
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        print(f"🚀 GPU {gpu_id} Experiment {experiment_id}: Initialized on device {self.device}")
        print(f"   Random seed: {self.random_seed}")
        print(f"   Save directory: {self.save_dir}")
    
    def create_challenging_dataset(self, n_samples=3000, input_dim=784, n_classes=10):
        """Create a unique challenging synthetic dataset for this experiment."""
        print(f"📦 GPU {self.gpu_id}: Creating unique synthetic dataset...")
        
        patterns = []
        labels = []
        samples_per_class = n_samples // n_classes
        
        # Use experiment-specific random seed for unique datasets
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        for class_idx in range(n_classes):
            for i in range(samples_per_class):
                # Create experiment-specific patterns
                base_pattern = torch.randn(input_dim) * (0.8 + 0.2 * self.experiment_id)
                
                # Add class-specific complex structure with experiment variation
                freq_modifier = 1.0 + 0.1 * self.experiment_id
                amplitude_modifier = 1.0 + 0.2 * self.gpu_id
                
                if class_idx < 2:
                    # High-frequency sinusoidal patterns
                    freq = (class_idx + 1) * 0.2 * freq_modifier
                    indices = torch.arange(input_dim).float()
                    base_pattern += torch.sin(indices * freq) * 3 * amplitude_modifier
                    base_pattern += torch.cos(indices * freq * 0.5) * 2 * amplitude_modifier
                elif class_idx < 4:
                    # Polynomial patterns with experiment-specific powers
                    power = class_idx + self.experiment_id * 0.5
                    base_pattern += (torch.randn(input_dim) ** power) * 2.5 * amplitude_modifier
                elif class_idx < 6:
                    # Mixed frequency patterns
                    base_pattern += torch.cos(torch.arange(input_dim).float() * 0.1 * class_idx * freq_modifier) * 2.5
                    base_pattern += torch.sin(torch.arange(input_dim).float() * 0.05 * class_idx * freq_modifier) * 1.5
                elif class_idx < 8:
                    # Exponential decay patterns
                    decay_rate = 100 + class_idx * 50 + self.experiment_id * 20
                    decay = torch.exp(-torch.arange(input_dim).float() / decay_rate)
                    base_pattern += decay * torch.randn(input_dim) * 4 * amplitude_modifier
                else:
                    # Complex multi-modal patterns
                    mode1 = torch.sin(torch.arange(input_dim).float() * 0.1 * freq_modifier) * 2
                    mode2 = torch.cos(torch.arange(input_dim).float() * 0.05 * freq_modifier) * 2
                    mode3 = torch.randn(input_dim) * 1.5 * amplitude_modifier
                    base_pattern += mode1 + mode2 + mode3
                
                # Add structured noise with experiment variation
                noise_scale = 0.5 + 0.1 * self.experiment_id
                noise = torch.randn(input_dim) * noise_scale
                pattern = base_pattern + noise
                
                # Add some extreme values to trigger extrema detection
                extreme_prob = 0.1 + 0.05 * self.gpu_id  # Different extreme probabilities per GPU
                if np.random.random() < extreme_prob:
                    extreme_indices = torch.randperm(input_dim)[:50]
                    pattern[extreme_indices] *= (3 + self.experiment_id * 0.5)
                
                patterns.append(pattern)
                labels.append(class_idx)
        
        X = torch.stack(patterns).to(self.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Shuffle the dataset
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        
        print(f"✅ GPU {self.gpu_id}: Dataset created with {X.shape[0]} samples")
        print(f"   Input range: [{X.min():.2f}, {X.max():.2f}]")
        print(f"   Input std: {X.std():.2f}")
        print(f"   Class distribution: {torch.bincount(y).tolist()}")
        
        return X, y
    
    def analyze_extrema_patterns(self, network, epoch):
        """Enhanced extrema analysis for this experiment."""
        # Run a forward pass to populate activations
        test_input = torch.randn(128, 784).to(self.device)
        _ = network(test_input)
        
        # Detect extrema with multiple methods
        extrema_adaptive = network.network.detect_extrema(use_adaptive=True, epoch=epoch)
        extrema_lenient = network.network.detect_extrema(use_adaptive=False, threshold_high=0.6, threshold_low=-0.6)
        extrema_strict = network.network.detect_extrema(use_adaptive=False, threshold_high=0.8, threshold_low=-0.8)
        
        # Analyze activation statistics in detail
        activation_stats = []
        for i, activations in enumerate(network.network.activation_history[:-1]):
            mean_acts = activations.mean(dim=0)
            stats = {
                'layer': i,
                'mean': mean_acts.mean().item(),
                'std': mean_acts.std().item(),
                'min': mean_acts.min().item(),
                'max': mean_acts.max().item(),
                'range': (mean_acts.max() - mean_acts.min()).item(),
                'high_count_08': (mean_acts > 0.8).sum().item(),
                'low_count_08': (mean_acts < -0.8).sum().item(),
                'high_count_06': (mean_acts > 0.6).sum().item(),
                'low_count_06': (mean_acts < -0.6).sum().item(),
                'saturation_ratio': ((mean_acts.abs() > 0.8).sum().float() / len(mean_acts)).item()
            }
            activation_stats.append(stats)
        
        # Count extrema for different methods
        adaptive_count = sum(len(layer['high']) + len(layer['low']) for layer in extrema_adaptive.values())
        lenient_count = sum(len(layer['high']) + len(layer['low']) for layer in extrema_lenient.values())
        strict_count = sum(len(layer['high']) + len(layer['low']) for layer in extrema_strict.values())
        
        extrema_analysis = {
            'epoch': epoch,
            'gpu_id': self.gpu_id,
            'experiment_id': self.experiment_id,
            'adaptive_extrema_count': adaptive_count,
            'lenient_extrema_count': lenient_count,
            'strict_extrema_count': strict_count,
            'adaptive_extrema': extrema_adaptive,
            'activation_stats': activation_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        self.extrema_evolution.append(extrema_analysis)
        
        # Print detailed extrema info every 10 epochs
        if epoch % 10 == 0:
            print(f"   📊 GPU {self.gpu_id} Exp {self.experiment_id}: Adaptive={adaptive_count}, Lenient={lenient_count}, Strict={strict_count}")
        
        return extrema_analysis
    
    def monitor_growth_event(self, network, epoch, epoch_stats):
        """Enhanced growth event monitoring for this experiment."""
        if epoch_stats.get('growth_events', 0) > 0:
            # Get detailed growth information
            growth_stats = network.get_growth_stats()
            
            growth_info = {
                'epoch': epoch,
                'gpu_id': self.gpu_id,
                'experiment_id': self.experiment_id,
                'connections_added': epoch_stats.get('connections_added', 0),
                'total_connections': epoch_stats.get('total_connections', 0),
                'phase': epoch_stats.get('phase', 'unknown'),
                'performance': epoch_stats.get('performance', 0),
                'loss': epoch_stats.get('loss', 0),
                'growth_stats': growth_stats,
                'scheduler_stats': growth_stats.get('scheduler_stats', {}),
                'routing_stats': growth_stats.get('routing_stats', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            self.growth_events.append(growth_info)
            
            print(f"\n🌱 GPU {self.gpu_id} Exp {self.experiment_id}: GROWTH EVENT #{len(self.growth_events)} at epoch {epoch}!")
            print(f"   💪 Connections added: {growth_info['connections_added']}")
            print(f"   🔗 Total connections: {growth_info['total_connections']}")
            print(f"   📈 Performance: {growth_info['performance']:.4f}")
            print(f"   🎯 Phase: {growth_info['phase']}")
            
            return True
        return False
    
    def track_snapshots(self, network, epoch):
        """Enhanced snapshot tracking for this experiment."""
        snapshots = network.get_snapshots()
        
        # Only process new snapshots
        new_snapshots = snapshots[len(self.snapshot_timeline):]
        
        for snapshot in new_snapshots:
            # Prevent duplicate snapshots at the same epoch
            if epoch != self.last_snapshot_epoch:
                snapshot_info = {
                    'epoch': epoch,
                    'gpu_id': self.gpu_id,
                    'experiment_id': self.experiment_id,
                    'snapshot_id': snapshot['snapshot_id'],
                    'phase': snapshot['phase'],
                    'performance': snapshot.get('performance', 0),
                    'growth_occurred': snapshot.get('growth_info', {}).get('growth_occurred', False),
                    'timestamp': datetime.now().isoformat()
                }
                self.snapshot_timeline.append(snapshot_info)
                self.last_snapshot_epoch = epoch
                print(f"📸 GPU {self.gpu_id} Exp {self.experiment_id}: Snapshot {snapshot['snapshot_id']} (phase: {snapshot['phase']})")
    
    def run_experiment(self, epochs=100, batch_size=64, learning_rate=0.002):
        """Run independent experiment on this GPU."""
        print(f"🚀 GPU {self.gpu_id} Experiment {self.experiment_id}: Starting")
        print(f"🎯 Device: {self.device}")
        print(f"📊 Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        print("=" * 60)
        
        # Create network with experiment-specific variations
        sparsity_variation = 0.0001 * (1.0 + 0.1 * self.experiment_id)  # Slight sparsity variations
        network = create_multi_scale_network(
            784, [512, 256, 128], 10,
            sparsity=sparsity_variation,
            device=self.device,
            snapshot_dir=os.path.join(self.save_dir, "snapshots")
        )
        
        print(f"🔧 GPU {self.gpu_id}: Growth scheduler threshold: {network.growth_scheduler.growth_threshold}")
        print(f"🔧 GPU {self.gpu_id}: Connection router max fan-out: {network.connection_router.max_fan_out}")
        print(f"🏗️  GPU {self.gpu_id}: Network created:")
        initial_stats = network.network.get_connectivity_stats()
        print(f"   Architecture: {network.network.layer_sizes}")
        print(f"   Initial connections: {initial_stats['total_active_connections']}")
        print(f"   Initial sparsity: {initial_stats['sparsity']:.6f}")
        
        # Create unique dataset for this experiment
        X, y = self.create_challenging_dataset(n_samples=3000)
        
        # Create data loaders
        dataset = torch.utils.data.TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Training setup with experiment-specific learning rate variation
        lr_variation = learning_rate * (1.0 + 0.1 * self.experiment_id)
        optimizer = optim.Adam(network.parameters(), lr=lr_variation)
        criterion = nn.CrossEntropyLoss()
        
        print(f"📚 GPU {self.gpu_id}: Training data: {len(train_dataset)} samples")
        print(f"🧪 GPU {self.gpu_id}: Test data: {len(test_dataset)} samples")
        print(f"📈 GPU {self.gpu_id}: Learning rate: {lr_variation:.6f}")
        print("=" * 60)
        
        # Training loop
        start_time = time.time()
        best_performance = 0
        epochs_since_growth = 0
        total_growth_events = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training epoch
            epoch_stats = network.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Evaluation
            eval_stats = network.evaluate(test_loader, criterion)
            
            # Combine stats
            combined_stats = {
                **epoch_stats,
                'test_loss': eval_stats['loss'],
                'test_performance': eval_stats['performance'],
                'gpu_id': self.gpu_id,
                'experiment_id': self.experiment_id
            }
            
            # Track performance
            self.performance_history.append({
                'epoch': epoch,
                'train_loss': epoch_stats['loss'],
                'train_acc': epoch_stats['performance'],
                'test_loss': eval_stats['loss'],
                'test_acc': eval_stats['performance'],
                'connections': epoch_stats['total_connections'],
                'gpu_id': self.gpu_id,
                'experiment_id': self.experiment_id
            })
            
            # Monitor growth events
            growth_occurred = self.monitor_growth_event(network, epoch, epoch_stats)
            if growth_occurred:
                epochs_since_growth = 0
                total_growth_events += 1
            else:
                epochs_since_growth += 1
            
            # Analyze extrema patterns (every 5 epochs or after growth)
            if epoch % 5 == 0 or growth_occurred:
                extrema_analysis = self.analyze_extrema_patterns(network, epoch)
            
            # Track snapshots
            self.track_snapshots(network, epoch)
            
            # Log training progress
            self.training_log.append(combined_stats)
            
            # Update best performance
            if eval_stats['performance'] > best_performance:
                best_performance = eval_stats['performance']
            
            # Progress reporting (less frequent to avoid spam)
            epoch_time = time.time() - epoch_start
            if epoch % 10 == 0 or growth_occurred:
                print(f"GPU {self.gpu_id} Exp {self.experiment_id} Epoch {epoch:3d}/{epochs}: "
                      f"Loss={epoch_stats['loss']:.4f}, "
                      f"Train Acc={epoch_stats['performance']:.3f}, "
                      f"Test Acc={eval_stats['performance']:.3f}, "
                      f"Connections={epoch_stats['total_connections']}, "
                      f"Time={epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Final analysis
        print(f"\n🎉 GPU {self.gpu_id} Experiment {self.experiment_id} Completed!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"🏆 Best test accuracy: {best_performance:.4f}")
        print(f"🌱 Total growth events: {total_growth_events}")
        print(f"📸 Total snapshots: {len(self.snapshot_timeline)}")
        
        final_stats = network.network.get_connectivity_stats()
        connection_growth = final_stats['total_active_connections'] - initial_stats['total_active_connections']
        growth_percentage = (connection_growth / initial_stats['total_active_connections']) * 100 if initial_stats['total_active_connections'] > 0 else 0
        
        print(f"🔗 Final connections: {final_stats['total_active_connections']} "
              f"(+{connection_growth}, +{growth_percentage:.1f}%)")
        
        # Save results
        self.save_results(network, total_time, best_performance)
        
        # Return summary for aggregation
        return {
            'gpu_id': self.gpu_id,
            'experiment_id': self.experiment_id,
            'total_time': total_time,
            'best_performance': best_performance,
            'total_growth_events': total_growth_events,
            'total_snapshots': len(self.snapshot_timeline),
            'final_connections': final_stats['total_active_connections'],
            'connection_growth': connection_growth,
            'growth_percentage': growth_percentage,
            'random_seed': self.random_seed,
            'save_dir': self.save_dir
        }
    
    def save_results(self, network, total_time, best_performance):
        """Save experimental results for this experiment."""
        print(f"💾 GPU {self.gpu_id} Exp {self.experiment_id}: Saving results...")
        
        # Save training log
        with open(os.path.join(self.save_dir, "training_log.json"), 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        # Save growth events
        with open(os.path.join(self.save_dir, "growth_events.json"), 'w') as f:
            json.dump(self.growth_events, f, indent=2, default=str)
        
        # Save extrema evolution
        extrema_json = []
        for analysis in self.extrema_evolution:
            json_analysis = analysis.copy()
            # Convert tensor data to serializable format
            if 'adaptive_extrema' in json_analysis:
                json_analysis['adaptive_extrema'] = {
                    str(k): {
                        'high': v['high'] if isinstance(v['high'], list) else v['high'].tolist(),
                        'low': v['low'] if isinstance(v['low'], list) else v['low'].tolist()
                    } for k, v in json_analysis['adaptive_extrema'].items()
                }
            extrema_json.append(json_analysis)
        
        with open(os.path.join(self.save_dir, "extrema_evolution.json"), 'w') as f:
            json.dump(extrema_json, f, indent=2)
        
        # Save snapshot timeline
        with open(os.path.join(self.save_dir, "snapshot_timeline.json"), 'w') as f:
            json.dump(self.snapshot_timeline, f, indent=2)
        
        # Save performance history
        with open(os.path.join(self.save_dir, "performance_history.json"), 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        # Save experiment summary
        experiment_summary = {
            'gpu_id': self.gpu_id,
            'experiment_id': self.experiment_id,
            'random_seed': self.random_seed,
            'network_stats': network.network.get_connectivity_stats(),
            'growth_stats': network.get_growth_stats(),
            'total_training_time': total_time,
            'best_performance': best_performance,
            'experiment_summary': {
                'total_epochs': len(self.training_log),
                'growth_events': len(self.growth_events),
                'snapshots_created': len(self.snapshot_timeline),
                'final_performance': self.performance_history[-1] if self.performance_history else None,
                'parallel_execution': True,
                'independent_dataset': True,
                'unique_random_seed': self.random_seed
            }
        }
        
        with open(os.path.join(self.save_dir, "experiment_summary.json"), 'w') as f:
            json.dump(experiment_summary, f, indent=2, default=str)
        
        print(f"✅ GPU {self.gpu_id} Exp {self.experiment_id}: Results saved to {self.save_dir}/")


def run_single_experiment(gpu_id, experiment_id, epochs, batch_size, learning_rate):
    """Run a single experiment on a specific GPU."""
    try:
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_id)
        
        # Create and run experiment
        experiment = ParallelExperiment1(gpu_id, experiment_id)
        result = experiment.run_experiment(epochs, batch_size, learning_rate)
        
        return result
        
    except Exception as e:
        print(f"❌ Error in GPU {gpu_id} Experiment {experiment_id}: {e}")
        return None


def aggregate_results(results, base_save_dir):
    """Aggregate results from all parallel experiments."""
    print("\n" + "=" * 80)
    print("📊 AGGREGATING PARALLEL EXPERIMENT RESULTS")
    print("=" * 80)
    
    # Filter out failed experiments
    successful_results = [r for r in results if r is not None]
    
    if not successful_results:
        print("❌ No successful experiments to aggregate!")
        return
    
    # Calculate statistics
    total_experiments = len(successful_results)
    total_time = sum(r['total_time'] for r in successful_results)
    avg_time = total_time / total_experiments
    max_time = max(r['total_time'] for r in successful_results)
    min_time = min(r['total_time'] for r in successful_results)
    
    performances = [r['best_performance'] for r in successful_results]
    avg_performance = np.mean(performances)
    std_performance = np.std(performances)
    max_performance = max(performances)
    min_performance = min(performances)
    
    growth_events = [r['total_growth_events'] for r in successful_results]
    avg_growth_events = np.mean(growth_events)
    
    connections = [r['final_connections'] for r in successful_results]
    avg_connections = np.mean(connections)
    
    # Print summary
    print(f"🎉 PARALLEL EXPERIMENT SUMMARY")
    print(f"📊 Total experiments completed: {total_experiments}")
    print(f"⏱️  Total wall-clock time: {max_time:.2f} seconds (fastest GPU)")
    print(f"⏱️  Total compute time: {total_time:.2f} seconds (all GPUs combined)")
    print(f"⚡ Speedup achieved: {total_time/max_time:.2f}x")
    print(f"📈 Performance statistics:")
    print(f"   Average accuracy: {avg_performance:.4f} ± {std_performance:.4f}")
    print(f"   Best accuracy: {max_performance:.4f}")
    print(f"   Worst accuracy: {min_performance:.4f}")
    print(f"🌱 Average growth events: {avg_growth_events:.1f}")
    print(f"🔗 Average final connections: {avg_connections:.0f}")
    
    # Save aggregated results
    aggregated_data = {
        'summary': {
            'total_experiments': total_experiments,
            'total_wall_clock_time': max_time,
            'total_compute_time': total_time,
            'speedup_achieved': total_time / max_time,
            'avg_time_per_experiment': avg_time,
            'performance_stats': {
                'mean': avg_performance,
                'std': std_performance,
                'min': min_performance,
                'max': max_performance
            },
            'avg_growth_events': avg_growth_events,
            'avg_final_connections': avg_connections
        },
        'individual_results': successful_results,
        'timestamp': datetime.now().isoformat()
    }
    
    aggregated_file = os.path.join(base_save_dir, "aggregated_results.json")
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_data, f, indent=2)
    
    print(f"📁 Aggregated results saved to: {aggregated_file}")
    
    # Create comparison visualization
    create_comparison_visualization(successful_results, base_save_dir)


def create_comparison_visualization(results, base_save_dir):
    """Create visualization comparing all parallel experiments."""
    print("📊 Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parallel Experiment 1 Results Comparison', fontsize=16)
    
    # Extract data
    gpu_ids = [r['gpu_id'] for r in results]
    exp_ids = [r['experiment_id'] for r in results]
    performances = [r['best_performance'] for r in results]
    times = [r['total_time'] for r in results]
    growth_events = [r['total_growth_events'] for r in results]
    connections = [r['final_connections'] for r in results]
    growth_percentages = [r['growth_percentage'] for r in results]
    
    # Create labels
    labels = [f"GPU{gpu}_Exp{exp}" for gpu, exp in zip(gpu_ids, exp_ids)]
    
    # 1. Performance comparison
    bars1 = axes[0, 0].bar(labels, performances, alpha=0.7, color='blue')
    axes[0, 0].set_title('Best Performance by Experiment')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, performances):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Training time comparison
    bars2 = axes[0, 1].bar(labels, times, alpha=0.7, color='green')
    axes[0, 1].set_title('Training Time by Experiment')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Growth events comparison
    bars3 = axes[0, 2].bar(labels, growth_events, alpha=0.7, color='orange')
    axes[0, 2].set_title('Growth Events by Experiment')
    axes[0, 2].set_ylabel('Number of Growth Events')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Final connections comparison
    bars4 = axes[1, 0].bar(labels, connections, alpha=0.7, color='red')
    axes[1, 0].set_title('Final Connections by Experiment')
    axes[1, 0].set_ylabel('Number of Connections')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Growth percentage comparison
    bars5 = axes[1, 1].bar(labels, growth_percentages, alpha=0.7, color='purple')
    axes[1, 1].set_title('Connection Growth % by Experiment')
    axes[1, 1].set_ylabel('Growth Percentage')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Performance vs Time scatter
    colors = ['blue' if gpu == 0 else 'red' for gpu in gpu_ids]
    axes[1, 2].scatter(times, performances, c=colors, alpha=0.7, s=100)
    axes[1, 2].set_title('Performance vs Training Time')
    axes[1, 2].set_xlabel('Training Time (seconds)')
    axes[1, 2].set_ylabel('Best Accuracy')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, (time, perf, label) in enumerate(zip(times, performances, labels)):
        axes[1, 2].annotate(label, (time, perf), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_save_dir, 'parallel_experiments_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparison visualization saved to {base_save_dir}/parallel_experiments_comparison.png")


def main():
    """Main function to run parallel experiments on multiple GPUs."""
    parser = argparse.ArgumentParser(description='Parallel Experiment 1')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per experiment (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.002, help='Learning rate (default: 0.002)')
    parser.add_argument('--experiments-per-gpu', type=int, default=1, help='Number of experiments per GPU (default: 1)')
    parser.add_argument('--gpus', type=str, default='0,1', help='GPU IDs to use (default: 0,1)')
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Parallel training requires CUDA.")
        return
    
    available_gpus = torch.cuda.device_count()
    for gpu_id in gpu_ids:
        if gpu_id >= available_gpus:
            print(f"❌ GPU {gpu_id} not available! Only {available_gpus} GPUs detected.")
            return
    
    print(f"🚀 Starting Parallel Experiment 1")
    print(f"🎯 Using GPUs: {gpu_ids}")
    print(f"📊 Configuration: {args.epochs} epochs, {args.batch_size} batch size, {args.learning_rate} LR")
    print(f"🔄 Running {args.experiments_per_gpu} experiment(s) per GPU")
    print(f"📈 Total experiments: {len(gpu_ids) * args.experiments_per_gpu}")
    
    # Create base save directory
    base_save_dir = "experiment_1_parallel_results"
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Prepare experiment configurations
    experiment_configs = []
    for gpu_id in gpu_ids:
        for exp_id in range(args.experiments_per_gpu):
            experiment_configs.append((gpu_id, exp_id, args.epochs, args.batch_size, args.learning_rate))
    
    print(f"🚀 Launching {len(experiment_configs)} parallel experiments...")
    start_time = time.time()
    
    # Run experiments in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        # Submit all experiments
        futures = [
            executor.submit(run_single_experiment, gpu_id, exp_id, epochs, batch_size, learning_rate)
            for gpu_id, exp_id, epochs, batch_size, learning_rate in experiment_configs
        ]
        
        # Collect results as they complete
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=3600)  # 1 hour timeout per experiment
                if result:
                    results.append(result)
                    print(f"✅ Experiment {i+1}/{len(experiment_configs)} completed successfully")
                else:
                    print(f"❌ Experiment {i+1}/{len(experiment_configs)} failed")
            except Exception as e:
                print(f"❌ Experiment {i+1}/{len(experiment_configs)} failed with error: {e}")
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 All parallel experiments completed in {total_time:.2f} seconds!")
    
    # Aggregate and analyze results
    if results:
        aggregate_results(results, base_save_dir)
    else:
        print("❌ No experiments completed successfully!")


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()
