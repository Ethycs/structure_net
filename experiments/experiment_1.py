#!/usr/bin/env python3
"""
Experiment 1: Multi-Scale Snapshots - Neural Network Communication Theory

This is the main implementation of Experiment 1 from the specification.
Implements the complete multi-scale network with dynamic growth based on:

1. Gradient variance spike detection for growth triggers
2. Extrema-based connection routing (high → low extrema)
3. Credit-based growth economy system
4. Multi-scale preservation (coarse → medium → fine phases)
5. Snapshot management with performance thresholds
6. Load balancing and structural limits

This implementation closely follows the 13 rules specified in experiment 1.md
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import json
from datetime import datetime

sys.path.append('.')
from src.structure_net import create_multi_scale_network

class Experiment1:
    """
    Main implementation of Experiment 1: Multi-Scale Snapshots
    
    Implements the Neural Network Communication Theory with dynamic growth
    based on extrema detection and credit-based growth economy.
    """
    
    def __init__(self, device=None, save_dir="experiment_1_results"):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Experiment tracking
        self.training_log = []
        self.growth_events = []
        self.extrema_evolution = []
        self.snapshot_timeline = []
        self.performance_history = []
        
        # Track snapshots to prevent duplicates
        self.last_snapshot_epoch = -1
        
    def create_challenging_dataset(self, n_samples=3000, input_dim=784, n_classes=10):
        """Create a more challenging synthetic dataset."""
        print("📦 Creating challenging synthetic dataset...")
        
        patterns = []
        labels = []
        samples_per_class = n_samples // n_classes
        
        for class_idx in range(n_classes):
            for i in range(samples_per_class):
                # Create more complex patterns that should trigger growth
                base_pattern = torch.randn(input_dim) * 0.8
                
                # Add class-specific complex structure
                if class_idx < 2:
                    # High-frequency sinusoidal patterns
                    freq = (class_idx + 1) * 0.2
                    indices = torch.arange(input_dim).float()
                    base_pattern += torch.sin(indices * freq) * 3
                    base_pattern += torch.cos(indices * freq * 0.5) * 2
                elif class_idx < 4:
                    # Polynomial patterns with higher powers
                    power = class_idx
                    base_pattern += (torch.randn(input_dim) ** power) * 2.5
                elif class_idx < 6:
                    # Mixed frequency patterns
                    base_pattern += torch.cos(torch.arange(input_dim).float() * 0.1 * class_idx) * 2.5
                    base_pattern += torch.sin(torch.arange(input_dim).float() * 0.05 * class_idx) * 1.5
                elif class_idx < 8:
                    # Exponential decay patterns
                    decay = torch.exp(-torch.arange(input_dim).float() / (100 + class_idx * 50))
                    base_pattern += decay * torch.randn(input_dim) * 4
                else:
                    # Complex multi-modal patterns
                    mode1 = torch.sin(torch.arange(input_dim).float() * 0.1) * 2
                    mode2 = torch.cos(torch.arange(input_dim).float() * 0.05) * 2
                    mode3 = torch.randn(input_dim) * 1.5
                    base_pattern += mode1 + mode2 + mode3
                
                # Add structured noise
                noise = torch.randn(input_dim) * 0.5
                pattern = base_pattern + noise
                
                # Add some extreme values to trigger extrema detection
                if i % 10 == 0:  # 10% of samples get extreme values
                    extreme_indices = torch.randperm(input_dim)[:50]
                    pattern[extreme_indices] *= 3  # Make some values extreme
                
                patterns.append(pattern)
                labels.append(class_idx)
        
        X = torch.stack(patterns).to(self.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Shuffle the dataset
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        
        print(f"✅ Dataset created: {X.shape[0]} samples, {n_classes} classes")
        print(f"   Input range: [{X.min():.2f}, {X.max():.2f}]")
        print(f"   Input std: {X.std():.2f}")
        print(f"   Class distribution: {torch.bincount(y).tolist()}")
        
        return X, y
    
    def analyze_extrema_patterns(self, network, epoch):
        """Enhanced extrema analysis with multiple detection methods."""
        # Run a forward pass to populate activations
        test_input = torch.randn(128, 784).to(self.device)  # Larger batch for better statistics
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
            print(f"   📊 Extrema analysis: Adaptive={adaptive_count}, Lenient={lenient_count}, Strict={strict_count}")
            for i, stats in enumerate(activation_stats):
                print(f"      Layer {i}: saturation={stats['saturation_ratio']:.3f}, range={stats['range']:.3f}")
        
        return extrema_analysis
    
    def monitor_growth_event(self, network, epoch, epoch_stats):
        """Enhanced growth event monitoring."""
        if epoch_stats.get('growth_events', 0) > 0:
            # Get detailed growth information
            growth_stats = network.get_growth_stats()
            
            growth_info = {
                'epoch': epoch,
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
            
            print(f"\n🌱 GROWTH EVENT #{len(self.growth_events)} at epoch {epoch}!")
            print(f"   💪 Connections added: {growth_info['connections_added']}")
            print(f"   🔗 Total connections: {growth_info['total_connections']}")
            print(f"   📈 Performance: {growth_info['performance']:.4f}")
            print(f"   🎯 Phase: {growth_info['phase']}")
            
            # Print scheduler details
            scheduler_stats = growth_info['scheduler_stats']
            if scheduler_stats:
                print(f"   💳 Credits spent: {scheduler_stats.get('current_credits', 0)}")
                print(f"   📊 Total spikes: {scheduler_stats.get('total_spikes', 0)}")
            
            return True
        return False
    
    def track_snapshots(self, network, epoch):
        """Enhanced snapshot tracking with duplicate prevention."""
        snapshots = network.get_snapshots()
        
        # Only process new snapshots
        new_snapshots = snapshots[len(self.snapshot_timeline):]
        
        for snapshot in new_snapshots:
            # Prevent duplicate snapshots at the same epoch
            if epoch != self.last_snapshot_epoch:
                snapshot_info = {
                    'epoch': epoch,
                    'snapshot_id': snapshot['snapshot_id'],
                    'phase': snapshot['phase'],
                    'performance': snapshot.get('performance', 0),
                    'growth_occurred': snapshot.get('growth_info', {}).get('growth_occurred', False),
                    'timestamp': datetime.now().isoformat()
                }
                self.snapshot_timeline.append(snapshot_info)
                self.last_snapshot_epoch = epoch
                print(f"📸 Snapshot saved: {snapshot['snapshot_id']} (phase: {snapshot['phase']})")
    
    def run_improved_experiment(self, epochs=100, batch_size=64, learning_rate=0.002):
        """Run improved extended training experiment."""
        print("🚀 Starting Improved Extended Training Experiment")
        print(f"🎯 Device: {self.device}")
        print(f"📊 Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        print("=" * 80)
        
        # Create network with specification-compliant settings
        network = create_multi_scale_network(
            784, [512, 256, 128], 10,
            sparsity=0.0001,  # 0.01% initial connectivity (back to specification)
            device=self.device,
            snapshot_dir=os.path.join(self.save_dir, "snapshots")
        )
        
        # Manually configure more aggressive growth settings
        # The improved parameters are already set in the updated classes
        print(f"🔧 Growth scheduler threshold: {network.growth_scheduler.growth_threshold}")
        print(f"🔧 Connection router max fan-out: {network.connection_router.max_fan_out}")
        
        print(f"🏗️  Network created with aggressive growth settings:")
        initial_stats = network.network.get_connectivity_stats()
        print(f"   Architecture: {network.network.layer_sizes}")
        print(f"   Initial connections: {initial_stats['total_active_connections']}")
        print(f"   Initial sparsity: {initial_stats['sparsity']:.4f}")
        
        # Create more challenging dataset
        X, y = self.create_challenging_dataset(n_samples=3000)
        
        # Create data loaders
        dataset = torch.utils.data.TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Training setup with higher learning rate
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print(f"📚 Training data: {len(train_dataset)} samples")
        print(f"🧪 Test data: {len(test_dataset)} samples")
        print("=" * 80)
        
        # Training loop with enhanced monitoring
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
                'test_performance': eval_stats['performance']
            }
            
            # Track performance
            self.performance_history.append({
                'epoch': epoch,
                'train_loss': epoch_stats['loss'],
                'train_acc': epoch_stats['performance'],
                'test_loss': eval_stats['loss'],
                'test_acc': eval_stats['performance'],
                'connections': epoch_stats['total_connections']
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
                
            # Track snapshots (with duplicate prevention)
            self.track_snapshots(network, epoch)
            
            # Log training progress
            self.training_log.append(combined_stats)
            
            # Update best performance
            if eval_stats['performance'] > best_performance:
                best_performance = eval_stats['performance']
            
            # Enhanced progress reporting
            epoch_time = time.time() - epoch_start
            if epoch % 5 == 0 or growth_occurred:
                print(f"Epoch {epoch:3d}/{epochs}: "
                      f"Loss={epoch_stats['loss']:.4f}, "
                      f"Train Acc={epoch_stats['performance']:.3f}, "
                      f"Test Acc={eval_stats['performance']:.3f}, "
                      f"Connections={epoch_stats['total_connections']}, "
                      f"Phase={epoch_stats['phase']}, "
                      f"Time={epoch_time:.2f}s")
                
                if epochs_since_growth > 15:
                    print(f"   ⏳ {epochs_since_growth} epochs since last growth")
                
                # Show scheduler status
                scheduler_stats = network.growth_scheduler.get_stats()
                print(f"   💳 Credits: {scheduler_stats['current_credits']}, "
                      f"Spikes: {scheduler_stats['total_spikes']}")
        
        total_time = time.time() - start_time
        
        # Final analysis
        print("\n" + "=" * 80)
        print("🎉 Improved Extended Training Completed!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"🏆 Best test accuracy: {best_performance:.4f}")
        print(f"🌱 Total growth events: {total_growth_events}")
        print(f"📸 Total snapshots: {len(self.snapshot_timeline)}")
        
        final_stats = network.network.get_connectivity_stats()
        connection_growth = final_stats['total_active_connections'] - initial_stats['total_active_connections']
        growth_percentage = (connection_growth / initial_stats['total_active_connections']) * 100
        
        print(f"🔗 Final connections: {final_stats['total_active_connections']} "
              f"(+{connection_growth}, +{growth_percentage:.1f}%)")
        
        # Save results
        self.save_results(network, total_time)
        
        # Generate visualizations
        self.create_visualizations()
        
        return network, self.training_log, self.growth_events
    
    def save_results(self, network, total_time):
        """Save all experimental results."""
        print("\n💾 Saving experimental results...")
        
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
        
        # Save final network state
        final_state = {
            'network_stats': network.network.get_connectivity_stats(),
            'growth_stats': network.get_growth_stats(),
            'total_training_time': total_time,
            'experiment_summary': {
                'total_epochs': len(self.training_log),
                'growth_events': len(self.growth_events),
                'snapshots_created': len(self.snapshot_timeline),
                'final_performance': self.performance_history[-1] if self.performance_history else None,
                'improvements': {
                    'fixed_snapshot_explosion': True,
                    'aggressive_growth_params': True,
                    'enhanced_extrema_detection': True,
                    'larger_growth_impact': True
                }
            }
        }
        
        with open(os.path.join(self.save_dir, "experiment_summary.json"), 'w') as f:
            json.dump(final_state, f, indent=2, default=str)
        
        print(f"✅ Results saved to {self.save_dir}/")
    
    def create_visualizations(self):
        """Create enhanced visualizations."""
        print("📊 Creating enhanced visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Improved Extended Training Experiment Results', fontsize=16)
        
        # 1. Training curves
        epochs = [p['epoch'] for p in self.performance_history]
        train_acc = [p['train_acc'] for p in self.performance_history]
        test_acc = [p['test_acc'] for p in self.performance_history]
        train_loss = [p['train_loss'] for p in self.performance_history]
        test_loss = [p['test_loss'] for p in self.performance_history]
        
        axes[0, 0].plot(epochs, train_acc, label='Train Accuracy', alpha=0.8)
        axes[0, 0].plot(epochs, test_acc, label='Test Accuracy', alpha=0.8)
        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark growth events
        for growth in self.growth_events:
            axes[0, 0].axvline(x=growth['epoch'], color='red', linestyle='--', alpha=0.7)
        
        # 2. Loss curves
        axes[0, 1].plot(epochs, train_loss, label='Train Loss', alpha=0.8)
        axes[0, 1].plot(epochs, test_loss, label='Test Loss', alpha=0.8)
        axes[0, 1].set_title('Loss Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mark growth events
        for growth in self.growth_events:
            axes[0, 1].axvline(x=growth['epoch'], color='red', linestyle='--', alpha=0.7)
        
        # 3. Connection growth
        connections = [p['connections'] for p in self.performance_history]
        axes[0, 2].plot(epochs, connections, color='green', linewidth=2)
        axes[0, 2].set_title('Network Growth (Connections)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Total Connections')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Mark growth events with connection counts
        for growth in self.growth_events:
            axes[0, 2].axvline(x=growth['epoch'], color='red', linestyle='--', alpha=0.7)
            axes[0, 2].annotate(f"+{growth['connections_added']}", 
                              xy=(growth['epoch'], growth['total_connections']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Enhanced extrema evolution
        if self.extrema_evolution:
            extrema_epochs = [e['epoch'] for e in self.extrema_evolution]
            adaptive_counts = [e['adaptive_extrema_count'] for e in self.extrema_evolution]
            lenient_counts = [e['lenient_extrema_count'] for e in self.extrema_evolution]
            strict_counts = [e['strict_extrema_count'] for e in self.extrema_evolution]
            
            axes[1, 0].plot(extrema_epochs, adaptive_counts, label='Adaptive', marker='o', alpha=0.8)
            axes[1, 0].plot(extrema_epochs, lenient_counts, label='Lenient (0.6)', marker='s', alpha=0.8)
            axes[1, 0].plot(extrema_epochs, strict_counts, label='Strict (0.8)', marker='^', alpha=0.8)
            axes[1, 0].set_title('Extrema Detection Comparison')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Extrema Count')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Growth events timeline
        if self.growth_events:
            growth_epochs = [g['epoch'] for g in self.growth_events]
            growth_connections = [g['connections_added'] for g in self.growth_events]
            
            bars = axes[1, 1].bar(growth_epochs, growth_connections, alpha=0.7, color='orange')
            axes[1, 1].set_title('Growth Events (Connections Added)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Connections Added')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, growth_connections):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value}', ha='center', va='bottom', fontsize=8)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Growth Events', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Growth Events')
        
        # 6. Saturation analysis
        if self.extrema_evolution:
            saturation_data = []
            for analysis in self.extrema_evolution:
                for layer_stats in analysis['activation_stats']:
                    saturation_data.append({
                        'epoch': analysis['epoch'],
                        'layer': layer_stats['layer'],
                        'saturation': layer_stats['saturation_ratio']
                    })
            
            # Plot saturation by layer
            layers = set(d['layer'] for d in saturation_data)
            for layer in layers:
                layer_data = [d for d in saturation_data if d['layer'] == layer]
                layer_epochs = [d['epoch'] for d in layer_data]
                layer_saturations = [d['saturation'] for d in layer_data]
                axes[1, 2].plot(layer_epochs, layer_saturations, label=f'Layer {layer}', alpha=0.8)
            
            axes[1, 2].set_title('Activation Saturation by Layer')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Saturation Ratio')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Performance vs Growth correlation
        if self.growth_events:
            growth_performance = [g['performance'] for g in self.growth_events]
            growth_epochs_scatter = [g['epoch'] for g in self.growth_events]
            
            axes[2, 0].scatter(growth_epochs_scatter, growth_performance, 
                             c='red', s=100, alpha=0.7, label='Growth Events')
            axes[2, 0].plot(epochs, test_acc, alpha=0.5, color='blue', label='Test Accuracy')
            axes[2, 0].set_title('Growth Events vs Performance')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Performance')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Connection growth rate
        if len(connections) > 1:
            growth_rate = np.diff(connections)
            axes[2, 1].plot(epochs[1:], growth_rate, color='purple', alpha=0.8)
            axes[2, 1].set_title('Connection Growth Rate')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Connections Added per Epoch')
            axes[2, 1].grid(True, alpha=0.3)
            
            # Mark significant growth
            for i, rate in enumerate(growth_rate):
                if rate > 10:  # Significant growth
                    axes[2, 1].axvline(x=epochs[i+1], color='red', linestyle='--', alpha=0.5)
        
        # 9. Summary statistics
        axes[2, 2].axis('off')
        summary_text = f"""
Experiment Summary:
• Total Epochs: {len(epochs)}
• Growth Events: {len(self.growth_events)}
• Snapshots: {len(self.snapshot_timeline)}
• Final Accuracy: {test_acc[-1]:.3f}
• Best Accuracy: {max(test_acc):.3f}
• Connection Growth: {connections[-1] - connections[0]:,}
• Growth Rate: {((connections[-1] - connections[0]) / connections[0] * 100):.1f}%

Improvements Applied:
✓ Fixed snapshot explosion
✓ Aggressive growth parameters
✓ Enhanced extrema detection
✓ Larger growth impact
        """
        axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'improved_training_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Enhanced visualizations saved to {self.save_dir}/improved_training_results.png")

def main():
    """Main function to run improved extended training experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Extended Training Experiment')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.002, help='Learning rate (default: 0.002)')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, cpu (default: auto)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Run experiment
    experiment = Experiment1(device=device)
    network, training_log, growth_events = experiment.run_improved_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print(f"\n🎯 Experiment 1 completed! Check {experiment.save_dir}/ for detailed results.")
    
    return experiment

if __name__ == "__main__":
    main()
