#!/usr/bin/env python3
"""
Debug Growth Mechanism

Found potential issues:
1. Extrema detection thresholds too strict (0.85/0.15 for tanh)
2. Gradient variance threshold too high (0.5 = 50% change)
3. Growth threshold too high (100 credits)
4. Bootstrap mechanism might not be working

Test with much more lenient parameters to see if growth triggers properly.
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

sys.path.append('.')
from src.structure_net import create_multi_scale_network
from src.structure_net.core.minimal_network import MinimalNetwork

class GrowthMechanismDebugger:
    """Debug the growth mechanism with detailed logging."""
    
    def __init__(self, save_dir="debug_growth_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Test parameters
        self.cliff_sparsity = 0.002
        self.input_size = 784
        self.hidden_size = 256
        self.output_size = 10
        
        # Debug configurations to test
        self.debug_configs = [
            {
                'name': 'original',
                'extrema_high': 0.85,
                'extrema_low': 0.15,
                'variance_threshold': 0.5,
                'growth_threshold': 100,
                'bootstrap_epochs': 15
            },
            {
                'name': 'lenient_extrema',
                'extrema_high': 0.6,   # Much more lenient
                'extrema_low': 0.4,    # Much more lenient
                'variance_threshold': 0.5,
                'growth_threshold': 100,
                'bootstrap_epochs': 15
            },
            {
                'name': 'lenient_variance',
                'extrema_high': 0.85,
                'extrema_low': 0.15,
                'variance_threshold': 0.1,  # Much more sensitive
                'growth_threshold': 100,
                'bootstrap_epochs': 15
            },
            {
                'name': 'low_threshold',
                'extrema_high': 0.85,
                'extrema_low': 0.15,
                'variance_threshold': 0.5,
                'growth_threshold': 20,  # Much lower threshold
                'bootstrap_epochs': 15
            },
            {
                'name': 'aggressive_bootstrap',
                'extrema_high': 0.85,
                'extrema_low': 0.15,
                'variance_threshold': 0.5,
                'growth_threshold': 100,
                'bootstrap_epochs': 30  # Longer bootstrap
            },
            {
                'name': 'super_lenient',
                'extrema_high': 0.55,   # Very lenient
                'extrema_low': 0.45,    # Very lenient
                'variance_threshold': 0.05,  # Very sensitive
                'growth_threshold': 10,  # Very low threshold
                'bootstrap_epochs': 25
            }
        ]
    
    def create_dataset(self, n_samples=1000):
        """Create smaller dataset for faster debugging."""
        print("📦 Creating debug dataset...")
        
        patterns = []
        labels = []
        samples_per_class = n_samples // self.output_size
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        for class_idx in range(self.output_size):
            for i in range(samples_per_class):
                base_pattern = torch.randn(self.input_size) * 0.8
                
                if class_idx < 2:
                    freq = (class_idx + 1) * 0.2
                    indices = torch.arange(self.input_size).float()
                    base_pattern += torch.sin(indices * freq) * 3
                elif class_idx < 4:
                    power = class_idx
                    base_pattern += (torch.randn(self.input_size) ** power) * 2.5
                else:
                    base_pattern += torch.cos(torch.arange(self.input_size).float() * 0.1 * class_idx) * 2.5
                
                noise = torch.randn(self.input_size) * 0.5
                pattern = base_pattern + noise
                
                patterns.append(pattern)
                labels.append(class_idx)
        
        X = torch.stack(patterns).to(self.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        
        print(f"✅ Debug dataset: {X.shape[0]} samples")
        return X, y
    
    def create_debug_network(self, config):
        """Create network with debug configuration."""
        print(f"🔧 Creating debug network: {config['name']}")
        print(f"   Extrema thresholds: {config['extrema_high']:.2f}/{config['extrema_low']:.2f}")
        print(f"   Variance threshold: {config['variance_threshold']:.2f}")
        print(f"   Growth threshold: {config['growth_threshold']}")
        print(f"   Bootstrap epochs: {config['bootstrap_epochs']}")
        
        # Create network
        network = create_multi_scale_network(
            self.input_size, [self.hidden_size], self.output_size,
            sparsity=self.cliff_sparsity,
            activation='tanh',
            device=self.device,
            snapshot_dir=os.path.join(self.save_dir, f"debug_snapshots_{config['name']}")
        )
        
        # Modify growth scheduler parameters
        if hasattr(network, 'growth_scheduler'):
            network.growth_scheduler.variance_threshold = config['variance_threshold']
            network.growth_scheduler.growth_threshold = config['growth_threshold']
            network.growth_scheduler.bootstrap_epochs = config['bootstrap_epochs']
            print(f"   Modified growth scheduler parameters")
        
        return network
    
    def train_debug_network(self, network, train_loader, test_loader, config, epochs=30):
        """Train network with detailed growth debugging."""
        print(f"🚀 Training debug network: {config['name']}")
        
        optimizer = optim.Adam(network.parameters(), lr=0.002)
        criterion = nn.CrossEntropyLoss()
        
        debug_log = {
            'config': config,
            'epochs': [],
            'growth_events': [],
            'extrema_counts': [],
            'gradient_norms': [],
            'credits': [],
            'variance_spikes': []
        }
        
        for epoch in range(epochs):
            # Training epoch
            epoch_stats = network.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Evaluation
            eval_stats = network.evaluate(test_loader, criterion)
            
            # Debug extrema detection with custom thresholds
            if hasattr(network.network, 'detect_extrema'):
                extrema = network.network.detect_extrema(
                    threshold_high=config['extrema_high'],
                    threshold_low=config['extrema_low'],
                    use_adaptive=False,  # Use fixed thresholds for debugging
                    epoch=epoch
                )
                
                # Count extrema
                total_high = sum(len(layer_extrema.get('high', [])) for layer_extrema in extrema.values())
                total_low = sum(len(layer_extrema.get('low', [])) for layer_extrema in extrema.values())
                
                debug_log['extrema_counts'].append({
                    'epoch': epoch,
                    'high': total_high,
                    'low': total_low,
                    'extrema': extrema
                })
                
                print(f"   Epoch {epoch}: High extrema: {total_high}, Low extrema: {total_low}")
            
            # Get growth scheduler stats
            if hasattr(network, 'growth_scheduler'):
                scheduler_stats = network.growth_scheduler.get_stats()
                debug_log['credits'].append({
                    'epoch': epoch,
                    'credits': scheduler_stats['current_credits'],
                    'spikes': scheduler_stats['total_spikes'],
                    'growth_events': scheduler_stats['total_growth_events']
                })
                
                print(f"   Epoch {epoch}: Credits: {scheduler_stats['current_credits']}, "
                      f"Spikes: {scheduler_stats['total_spikes']}, "
                      f"Growth events: {scheduler_stats['total_growth_events']}")
            
            # Track growth events
            if epoch_stats.get('growth_events', 0) > 0:
                debug_log['growth_events'].append({
                    'epoch': epoch,
                    'connections_added': epoch_stats.get('connections_added', 0),
                    'total_connections': epoch_stats['total_connections'],
                    'performance': eval_stats['performance']
                })
                print(f"   🌱 GROWTH EVENT! Added {epoch_stats.get('connections_added', 0)} connections")
            
            # Store epoch data
            debug_log['epochs'].append({
                'epoch': epoch,
                'train_acc': epoch_stats['performance'],
                'test_acc': eval_stats['performance'],
                'connections': epoch_stats['total_connections']
            })
            
            # Progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch:2d}/{epochs}: "
                      f"Test Acc={eval_stats['performance']:.3f}, "
                      f"Connections={epoch_stats['total_connections']}")
        
        return debug_log
    
    def run_debug_test(self, epochs=30):
        """Run complete debug test."""
        print("🔍 GROWTH MECHANISM DEBUG TEST")
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
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print(f"📚 Training data: {len(train_dataset)} samples")
        print(f"🧪 Test data: {len(test_dataset)} samples")
        
        # Test each configuration
        results = {}
        
        for config in self.debug_configs:
            print(f"\n🔧 TESTING CONFIGURATION: {config['name'].upper()}")
            print("-" * 50)
            
            # Create and train network
            network = self.create_debug_network(config)
            debug_log = self.train_debug_network(network, train_loader, test_loader, config, epochs)
            
            results[config['name']] = debug_log
            
            # Quick analysis
            growth_events = len(debug_log['growth_events'])
            final_performance = debug_log['epochs'][-1]['test_acc'] if debug_log['epochs'] else 0
            final_extrema = debug_log['extrema_counts'][-1] if debug_log['extrema_counts'] else {'high': 0, 'low': 0}
            
            print(f"\n📊 {config['name'].upper()} RESULTS:")
            print(f"   Growth events: {growth_events}")
            print(f"   Final performance: {final_performance:.3f}")
            print(f"   Final extrema: {final_extrema['high']} high, {final_extrema['low']} low")
            
            if growth_events > 1:
                print(f"   ✅ SUCCESS! Multiple growth events detected")
            elif growth_events == 1:
                print(f"   📈 PARTIAL! One growth event detected")
            else:
                print(f"   ❌ FAILED! No growth events detected")
        
        # Save results
        self.save_debug_results(results)
        self.create_debug_visualizations(results)
        
        return results
    
    def save_debug_results(self, results):
        """Save debug results."""
        print(f"\n💾 Saving debug results...")
        
        with open(os.path.join(self.save_dir, "debug_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'configurations_tested': len(results),
            'summary': {}
        }
        
        for name, result in results.items():
            growth_events = len(result['growth_events'])
            final_performance = result['epochs'][-1]['test_acc'] if result['epochs'] else 0
            
            summary['summary'][name] = {
                'growth_events': growth_events,
                'final_performance': final_performance,
                'config': result['config'],
                'success': growth_events > 1
            }
        
        with open(os.path.join(self.save_dir, "debug_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Debug results saved to {self.save_dir}/")
    
    def create_debug_visualizations(self, results):
        """Create debug visualizations."""
        print("📊 Creating debug visualizations...")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Growth Mechanism Debug Results', fontsize=16)
        
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        
        # 1. Growth events over time
        for i, (name, result) in enumerate(results.items()):
            if i < len(colors):
                epochs = [e['epoch'] for e in result['epochs']]
                growth_epochs = [e['epoch'] for e in result['growth_events']]
                
                axes[0, 0].plot(epochs, [0] * len(epochs), color=colors[i], alpha=0.3, linewidth=2)
                for ge in growth_epochs:
                    axes[0, 0].scatter(ge, i, color=colors[i], s=100, label=name if ge == growth_epochs[0] else "")
        
        axes[0, 0].set_title('Growth Events Timeline')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Configuration')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Performance curves
        for i, (name, result) in enumerate(results.items()):
            if i < len(colors):
                epochs = [e['epoch'] for e in result['epochs']]
                test_accs = [e['test_acc'] for e in result['epochs']]
                axes[0, 1].plot(epochs, test_accs, color=colors[i], label=name, linewidth=2)
        
        axes[0, 1].set_title('Performance Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Credits accumulation
        for i, (name, result) in enumerate(results.items()):
            if i < len(colors) and result['credits']:
                epochs = [c['epoch'] for c in result['credits']]
                credits = [c['credits'] for c in result['credits']]
                axes[1, 0].plot(epochs, credits, color=colors[i], label=name, linewidth=2)
        
        axes[1, 0].set_title('Credits Accumulation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Credits')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Extrema counts
        for i, (name, result) in enumerate(results.items()):
            if i < len(colors) and result['extrema_counts']:
                epochs = [e['epoch'] for e in result['extrema_counts']]
                high_counts = [e['high'] for e in result['extrema_counts']]
                low_counts = [e['low'] for e in result['extrema_counts']]
                
                axes[1, 1].plot(epochs, high_counts, color=colors[i], linestyle='-', 
                               label=f'{name} (high)', alpha=0.7)
                axes[1, 1].plot(epochs, low_counts, color=colors[i], linestyle='--', 
                               label=f'{name} (low)', alpha=0.7)
        
        axes[1, 1].set_title('Extrema Detection Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Extrema Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Growth events summary
        config_names = list(results.keys())
        growth_counts = [len(results[name]['growth_events']) for name in config_names]
        
        bars = axes[2, 0].bar(config_names, growth_counts, color=colors[:len(config_names)])
        axes[2, 0].set_title('Total Growth Events by Configuration')
        axes[2, 0].set_ylabel('Growth Events')
        axes[2, 0].tick_params(axis='x', rotation=45)
        axes[2, 0].grid(True, alpha=0.3)
        
        # Color bars based on success
        for i, (bar, count) in enumerate(zip(bars, growth_counts)):
            if count > 1:
                bar.set_color('green')
            elif count == 1:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # 6. Configuration parameters heatmap
        param_names = ['extrema_high', 'extrema_low', 'variance_threshold', 'growth_threshold', 'bootstrap_epochs']
        config_matrix = []
        
        for name in config_names:
            config = results[name]['config']
            row = [config[param] for param in param_names]
            config_matrix.append(row)
        
        config_matrix = np.array(config_matrix)
        
        # Normalize each column for better visualization
        for j in range(config_matrix.shape[1]):
            col = config_matrix[:, j]
            config_matrix[:, j] = (col - col.min()) / (col.max() - col.min()) if col.max() > col.min() else col
        
        im = axes[2, 1].imshow(config_matrix, cmap='viridis', aspect='auto')
        axes[2, 1].set_title('Configuration Parameters (Normalized)')
        axes[2, 1].set_xticks(range(len(param_names)))
        axes[2, 1].set_xticklabels(param_names, rotation=45)
        axes[2, 1].set_yticks(range(len(config_names)))
        axes[2, 1].set_yticklabels(config_names)
        
        plt.colorbar(im, ax=axes[2, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'debug_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Debug visualizations saved to {self.save_dir}/debug_analysis.png")


def main():
    """Main debug function."""
    print("🔍 GROWTH MECHANISM DEBUGGER")
    print("=" * 60)
    print("Testing different parameter configurations to find growth issues")
    print("=" * 60)
    
    debugger = GrowthMechanismDebugger()
    results = debugger.run_debug_test(epochs=30)
    
    print(f"\n🎉 Debug test completed!")
    
    # Find successful configurations
    successful_configs = []
    for name, result in results.items():
        if len(result['growth_events']) > 1:
            successful_configs.append(name)
    
    if successful_configs:
        print(f"\n✅ SUCCESSFUL CONFIGURATIONS FOUND:")
        for config in successful_configs:
            print(f"   - {config}")
        print(f"\n🚀 Use these parameters for cliff rescue!")
    else:
        print(f"\n⚠️  NO FULLY SUCCESSFUL CONFIGURATIONS")
        print(f"🔧 Growth mechanism needs deeper debugging")


if __name__ == "__main__":
    main()
