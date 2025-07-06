#!/usr/bin/env python3
"""
Aggressive Growth Test

The cliff rescue failed because growth was too conservative:
- Only 1 growth event in 50 epochs
- Only +5 connections added
- Performance got worse instead of better

Test aggressive growth parameters:
- Lower thresholds for more frequent growth
- More connections per growth event
- Earlier and more aggressive triggering

Goal: Rescue 0.002 sparsity from 24% → 35%+ with aggressive growth
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
from datetime import datetime
import argparse

sys.path.append('.')
from src.structure_net import create_multi_scale_network
from src.structure_net.core.minimal_network import MinimalNetwork

class AggressiveGrowthTest:
    """Test aggressive growth parameters to rescue cliff performance."""
    
    def __init__(self, save_dir="aggressive_growth_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Use RTX 2060 SUPER for consistency
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Test parameters
        self.cliff_sparsity = 0.002  # 24% static performance
        self.target_performance = 0.356  # 35.6% target
        
        # Architecture: single layer [256]
        self.input_size = 784
        self.hidden_size = 256
        self.output_size = 10
        
        # Results storage
        self.results = {}
    
    def create_dataset(self, n_samples=3000):
        """Create the same challenging dataset for consistency."""
        print("📦 Creating challenging synthetic dataset...")
        
        patterns = []
        labels = []
        samples_per_class = n_samples // self.output_size
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        for class_idx in range(self.output_size):
            for i in range(samples_per_class):
                # Create complex patterns (same as previous experiments)
                base_pattern = torch.randn(self.input_size) * 0.8
                
                # Add class-specific structure
                if class_idx < 2:
                    freq = (class_idx + 1) * 0.2
                    indices = torch.arange(self.input_size).float()
                    base_pattern += torch.sin(indices * freq) * 3
                    base_pattern += torch.cos(indices * freq * 0.5) * 2
                elif class_idx < 4:
                    power = class_idx
                    base_pattern += (torch.randn(self.input_size) ** power) * 2.5
                elif class_idx < 6:
                    base_pattern += torch.cos(torch.arange(self.input_size).float() * 0.1 * class_idx) * 2.5
                    base_pattern += torch.sin(torch.arange(self.input_size).float() * 0.05 * class_idx) * 1.5
                elif class_idx < 8:
                    decay = torch.exp(-torch.arange(self.input_size).float() / (100 + class_idx * 50))
                    base_pattern += decay * torch.randn(self.input_size) * 4
                else:
                    mode1 = torch.sin(torch.arange(self.input_size).float() * 0.1) * 2
                    mode2 = torch.cos(torch.arange(self.input_size).float() * 0.05) * 2
                    mode3 = torch.randn(self.input_size) * 1.5
                    base_pattern += mode1 + mode2 + mode3
                
                # Add structured noise
                noise = torch.randn(self.input_size) * 0.5
                pattern = base_pattern + noise
                
                patterns.append(pattern)
                labels.append(class_idx)
        
        X = torch.stack(patterns).to(self.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Shuffle
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        
        print(f"✅ Dataset created: {X.shape[0]} samples, {self.output_size} classes")
        return X, y
    
    def create_aggressive_growth_network(self, sparsity, growth_config):
        """Create growth-enabled network with aggressive parameters."""
        print(f"🌱 Creating aggressive growth network:")
        print(f"   Growth threshold: {growth_config['threshold']}")
        print(f"   Connections per event: {growth_config['connections_per_event']}")
        print(f"   Growth frequency: {growth_config['frequency']}")
        
        # Create the network
        network = create_multi_scale_network(
            self.input_size, [self.hidden_size], self.output_size,
            sparsity=sparsity,
            activation='tanh',
            device=self.device,
            snapshot_dir=os.path.join(self.save_dir, f"growth_snapshots_{growth_config['name']}")
        )
        
        # Modify growth scheduler for aggressive growth
        if hasattr(network, 'growth_scheduler'):
            # Lower threshold for more frequent growth
            network.growth_scheduler.threshold = growth_config['threshold']
            print(f"   Modified growth threshold to: {network.growth_scheduler.threshold}")
        
        # Modify connection router for more connections per event
        if hasattr(network, 'connection_router'):
            # Increase connections added per growth event
            network.connection_router.connections_per_event = growth_config['connections_per_event']
            print(f"   Modified connections per event to: {network.connection_router.connections_per_event}")
        
        stats = network.network.get_connectivity_stats()
        print(f"   Initial connections: {stats['total_active_connections']}")
        
        return network
    
    def train_aggressive_growth_network(self, network, train_loader, test_loader, epochs=50, lr=0.002, name="Aggressive Growth"):
        """Train aggressive growth network."""
        print(f"🚀 Training {name}...")
        
        optimizer = optim.Adam(network.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        connections_history = []
        growth_events = []
        
        best_test_acc = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training epoch using growth network's method
            epoch_stats = network.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Evaluation using growth network's method
            eval_stats = network.evaluate(test_loader, criterion)
            
            train_losses.append(epoch_stats['loss'])
            train_accs.append(epoch_stats['performance'])
            test_losses.append(eval_stats['loss'])
            test_accs.append(eval_stats['performance'])
            connections_history.append(epoch_stats['total_connections'])
            
            # Track growth events
            if epoch_stats.get('growth_events', 0) > 0:
                growth_events.append({
                    'epoch': epoch,
                    'connections_added': epoch_stats.get('connections_added', 0),
                    'total_connections': epoch_stats['total_connections'],
                    'performance': eval_stats['performance']
                })
                print(f"   🌱 GROWTH EVENT at epoch {epoch}! "
                      f"Added {epoch_stats.get('connections_added', 0)} connections "
                      f"(total: {epoch_stats['total_connections']}, performance: {eval_stats['performance']:.3f})")
            
            if eval_stats['performance'] > best_test_acc:
                best_test_acc = eval_stats['performance']
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch:2d}/{epochs}: "
                      f"Train Acc={epoch_stats['performance']:.3f}, Test Acc={eval_stats['performance']:.3f}, "
                      f"Connections={epoch_stats['total_connections']}")
        
        total_time = time.time() - start_time
        final_stats = network.network.get_connectivity_stats()
        
        print(f"✅ {name} completed!")
        print(f"   Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.1f}%)")
        print(f"   Final connections: {final_stats['total_active_connections']}")
        print(f"   Growth events: {len(growth_events)}")
        print(f"   Training time: {total_time:.2f} seconds")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'connections_history': connections_history,
            'best_test_acc': best_test_acc,
            'final_test_acc': test_accs[-1],
            'initial_connections': connections_history[0],
            'final_connections': final_stats['total_active_connections'],
            'connection_growth': final_stats['total_active_connections'] - connections_history[0],
            'training_time': total_time,
            'growth_events': len(growth_events),
            'growth_event_details': growth_events
        }
    
    def run_aggressive_growth_test(self, epochs=50):
        """Run aggressive growth parameter test."""
        print("🔬 AGGRESSIVE GROWTH TEST")
        print("=" * 60)
        print(f"🎯 GOAL: Rescue cliff performance with aggressive growth")
        print(f"📉 Cliff sparsity: {self.cliff_sparsity} (static: 24%)")
        print(f"🎯 Target: {self.target_performance:.1%}")
        print("=" * 60)
        
        # Create dataset
        X, y = self.create_dataset()
        
        # Create data loaders
        dataset = torch.utils.data.TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        print(f"📚 Training data: {len(train_dataset)} samples")
        print(f"🧪 Test data: {len(test_dataset)} samples")
        
        # Define aggressive growth configurations
        growth_configs = [
            {
                'name': 'moderate',
                'threshold': 50,  # Default is 100, lower = more frequent
                'connections_per_event': 20,  # Default is 10, more per event
                'frequency': 'every_5_epochs'
            },
            {
                'name': 'aggressive',
                'threshold': 25,  # Much lower threshold
                'connections_per_event': 50,  # Much more per event
                'frequency': 'every_3_epochs'
            },
            {
                'name': 'very_aggressive',
                'threshold': 10,  # Very low threshold
                'connections_per_event': 100,  # Lots per event
                'frequency': 'every_2_epochs'
            }
        ]
        
        # Test each aggressive growth configuration
        for config in growth_configs:
            print(f"\n🌱 TESTING {config['name'].upper()} GROWTH")
            print("-" * 50)
            
            # Create aggressive growth network
            network = self.create_aggressive_growth_network(self.cliff_sparsity, config)
            
            # Train with aggressive growth
            results = self.train_aggressive_growth_network(
                network, train_loader, test_loader,
                epochs=epochs, name=f"{config['name'].title()} Growth"
            )
            
            # Store results
            self.results[config['name']] = {
                'config': config,
                'sparsity': self.cliff_sparsity,
                **results
            }
            
            # Analyze this configuration
            performance = results['best_test_acc']
            growth_events = results['growth_events']
            connection_growth = results['connection_growth']
            target_achieved = performance >= self.target_performance
            
            print(f"\n📊 {config['name'].upper()} RESULTS:")
            print(f"   Performance: {performance:.1%} (target: {self.target_performance:.1%})")
            print(f"   Growth events: {growth_events}")
            print(f"   Connection growth: +{connection_growth}")
            print(f"   Target achieved: {'✅ YES' if target_achieved else '❌ NO'}")
            
            if target_achieved:
                print(f"   🎉 SUCCESS! Aggressive growth rescued cliff performance!")
                break
            else:
                improvement_needed = self.target_performance - performance
                print(f"   📈 Need {improvement_needed:.1%} more improvement")
        
        # Final analysis
        self.analyze_aggressive_results()
        
        # Save and visualize
        self.save_results()
        self.create_visualizations()
        
        return self.results
    
    def analyze_aggressive_results(self):
        """Analyze aggressive growth results."""
        print(f"\n📊 AGGRESSIVE GROWTH ANALYSIS")
        print("=" * 60)
        
        # Find best performing configuration
        best_config = None
        best_performance = 0
        
        for name, result in self.results.items():
            performance = result['best_test_acc']
            if performance > best_performance:
                best_performance = performance
                best_config = name
        
        print(f"🏆 BEST CONFIGURATION: {best_config.upper()}")
        if best_config:
            best_result = self.results[best_config]
            print(f"   Performance: {best_result['best_test_acc']:.1%}")
            print(f"   Growth events: {best_result['growth_events']}")
            print(f"   Connection growth: +{best_result['connection_growth']}")
            print(f"   Target achieved: {'✅ YES' if best_result['best_test_acc'] >= self.target_performance else '❌ NO'}")
        
        # Compare all configurations
        print(f"\n📈 CONFIGURATION COMPARISON:")
        for name, result in self.results.items():
            config = result['config']
            performance = result['best_test_acc']
            growth_events = result['growth_events']
            connection_growth = result['connection_growth']
            
            print(f"   {name.upper()}:")
            print(f"      Threshold: {config['threshold']}, Connections/event: {config['connections_per_event']}")
            print(f"      Performance: {performance:.1%}, Growth events: {growth_events}, +{connection_growth} connections")
        
        # Determine if aggressive growth worked
        any_success = any(result['best_test_acc'] >= self.target_performance for result in self.results.values())
        
        if any_success:
            print(f"\n🎉 AGGRESSIVE GROWTH SUCCESS!")
            print(f"   ✅ Found configuration that rescues cliff performance")
            print(f"   🚀 Ready to scale to multi-layer networks")
        else:
            max_performance = max(result['best_test_acc'] for result in self.results.values())
            print(f"\n📈 AGGRESSIVE GROWTH PARTIAL SUCCESS")
            print(f"   📊 Best performance: {max_performance:.1%} (target: {self.target_performance:.1%})")
            print(f"   🔧 May need even more aggressive parameters or different approach")
    
    def save_results(self):
        """Save experiment results."""
        print(f"\n💾 Saving results...")
        
        # Save detailed results
        with open(os.path.join(self.save_dir, "aggressive_growth_results.json"), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'cliff_sparsity': self.cliff_sparsity,
            'target_performance': self.target_performance,
            'configurations_tested': len(self.results),
            'best_performance': max(result['best_test_acc'] for result in self.results.values()) if self.results else 0,
            'target_achieved': any(result['best_test_acc'] >= self.target_performance for result in self.results.values()),
            'results_summary': {
                name: {
                    'performance': result['best_test_acc'],
                    'growth_events': result['growth_events'],
                    'connection_growth': result['connection_growth'],
                    'config': result['config']
                }
                for name, result in self.results.items()
            }
        }
        
        with open(os.path.join(self.save_dir, "aggressive_growth_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved to {self.save_dir}/")
    
    def create_visualizations(self):
        """Create aggressive growth visualizations."""
        print("📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Aggressive Growth Test Results', fontsize=16)
        
        # 1. Performance comparison
        configs = list(self.results.keys())
        performances = [self.results[name]['best_test_acc'] for name in configs]
        
        bars = axes[0, 0].bar(configs, performances, alpha=0.7)
        axes[0, 0].axhline(y=self.target_performance, color='red', linestyle='--', 
                          label=f'Target ({self.target_performance:.1%})')
        axes[0, 0].set_title('Performance by Configuration')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Color bars based on target achievement
        for i, (bar, perf) in enumerate(zip(bars, performances)):
            if perf >= self.target_performance:
                bar.set_color('green')
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{perf:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Growth events comparison
        growth_events = [self.results[name]['growth_events'] for name in configs]
        axes[0, 1].bar(configs, growth_events, alpha=0.7, color='orange')
        axes[0, 1].set_title('Growth Events by Configuration')
        axes[0, 1].set_ylabel('Number of Growth Events')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Connection growth comparison
        connection_growth = [self.results[name]['connection_growth'] for name in configs]
        axes[0, 2].bar(configs, connection_growth, alpha=0.7, color='purple')
        axes[0, 2].set_title('Connection Growth by Configuration')
        axes[0, 2].set_ylabel('Connections Added')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Training curves
        colors = ['blue', 'green', 'red', 'purple']
        for i, name in enumerate(configs):
            if i < len(colors):
                result = self.results[name]
                epochs = range(len(result['test_accs']))
                axes[1, 0].plot(epochs, result['test_accs'], color=colors[i], 
                               label=name.title(), linewidth=2)
        
        axes[1, 0].axhline(y=self.target_performance, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Training Curves')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Test Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Connection evolution
        for i, name in enumerate(configs):
            if i < len(colors):
                result = self.results[name]
                epochs = range(len(result['connections_history']))
                axes[1, 1].plot(epochs, result['connections_history'], color=colors[i], 
                               label=name.title(), linewidth=2)
                
                # Mark growth events
                for event in result['growth_event_details']:
                    axes[1, 1].axvline(x=event['epoch'], color=colors[i], linestyle=':', alpha=0.7)
        
        axes[1, 1].set_title('Connection Evolution')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Active Connections')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Configuration parameters
        thresholds = [self.results[name]['config']['threshold'] for name in configs]
        connections_per_event = [self.results[name]['config']['connections_per_event'] for name in configs]
        
        x = np.arange(len(configs))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, thresholds, width, label='Threshold', alpha=0.7)
        axes[1, 2].bar(x + width/2, connections_per_event, width, label='Connections/Event', alpha=0.7)
        axes[1, 2].set_title('Configuration Parameters')
        axes[1, 2].set_xlabel('Configuration')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(configs)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'aggressive_growth_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to {self.save_dir}/aggressive_growth_analysis.png")


def main():
    """Main function for aggressive growth test."""
    parser = argparse.ArgumentParser(description='Aggressive Growth Test')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--save-dir', type=str, default='aggressive_growth_results',
                       help='Directory to save results (default: aggressive_growth_results)')
    
    args = parser.parse_args()
    
    print("🔬 AGGRESSIVE GROWTH TEST")
    print("=" * 60)
    print("🎯 GOAL: Rescue cliff performance with aggressive growth")
    print("📉 Previous growth was too conservative (1 event, +5 connections)")
    print("🌱 Testing aggressive parameters (lower thresholds, more connections)")
    print("=" * 60)
    
    test = AggressiveGrowthTest(save_dir=args.save_dir)
    results = test.run_aggressive_growth_test(epochs=args.epochs)
    
    print(f"\n🎉 Aggressive growth test completed!")
    print(f"📁 Results saved to {args.save_dir}/")
    
    # Check if any configuration succeeded
    any_success = any(result['best_test_acc'] >= test.target_performance for result in results.values())
    if any_success:
        print(f"\n🎉 AGGRESSIVE GROWTH WORKS! Found rescue configuration.")
    else:
        best_perf = max(result['best_test_acc'] for result in results.values())
        print(f"\n📈 PARTIAL SUCCESS! Best: {best_perf:.1%} (target: {test.target_performance:.1%})")


if __name__ == "__main__":
    main()
