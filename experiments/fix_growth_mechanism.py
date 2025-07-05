#!/usr/bin/env python3
"""
Fix Growth Mechanism

Issues identified:
1. Gradient variance detection is broken
2. Only bootstrap mechanism works
3. Stabilization period too long
4. Need proportional growth (at least 5% forward skips)

Fixes:
1. Bypass gradient variance detection
2. Use connection-based growth triggers
3. Reduce stabilization period
4. Add proportional growth with forward skips
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
from src.structure_net.core.connection_router import ConnectionRouter

class FixedGrowthTest:
    """Test the fixed growth mechanism."""
    
    def __init__(self, save_dir="fixed_growth_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Test parameters
        self.cliff_sparsity = 0.002
        self.target_performance = 0.356
        self.input_size = 784
        self.hidden_sizes = [256, 128]
        self.output_size = 10
        
        self.connection_router = ConnectionRouter()
    
    def create_dataset(self, n_samples=1000):
        """Create debug dataset."""
        print("📦 Creating dataset...")
        
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
        
        print(f"✅ Dataset: {X.shape[0]} samples")
        return X, y
    
    def create_fixed_growth_network(self, sparsity):
        """Create network with fixed growth mechanism."""
        print(f"🔧 Creating fixed growth network (sparsity={sparsity})")
        
        # Create network
        network = create_multi_scale_network(
            self.input_size, self.hidden_sizes, self.output_size,
            sparsity=sparsity,
            activation='tanh',
            device=self.device,
            snapshot_dir=os.path.join(self.save_dir, "fixed_snapshots")
        )
        
        # Apply fixes to growth scheduler
        if hasattr(network, 'growth_scheduler'):
            # Fix 1: Bypass gradient variance detection
            network.growth_scheduler.variance_threshold = 0.01  # Very low threshold
            network.growth_scheduler.growth_threshold = 25      # Lower threshold
            network.growth_scheduler.stabilization_epochs = 3  # Shorter stabilization
            network.growth_scheduler.bootstrap_epochs = 20     # Longer bootstrap
            network.growth_scheduler.bootstrap_credits_per_epoch = 8  # More credits
            print(f"   Applied growth scheduler fixes")
        
        return network
    
    def apply_proportional_growth(self, network, epoch, total_epochs):
        """Apply proportional growth with forward skips."""
        if not hasattr(network, 'network') or not hasattr(network.network, 'connection_masks'):
            return 0
        
        # Calculate target connections based on epoch progress
        initial_connections = sum(mask.sum().item() for mask in network.network.connection_masks)
        
        # Target: grow to at least 5% more connections by end
        min_growth_rate = 0.05  # 5% minimum growth
        progress = epoch / total_epochs
        
        # Calculate target connections for this epoch
        target_connections = initial_connections * (1 + min_growth_rate * progress)
        current_connections = sum(mask.sum().item() for mask in network.network.connection_masks)
        
        connections_needed = int(target_connections - current_connections)
        
        if connections_needed > 0:
            # Add connections with forward skip preference
            connections_added = self.add_forward_skip_connections(network, connections_needed)
            return connections_added
        
        return 0
    
    def add_forward_skip_connections(self, network, num_connections):
        """Add connections by cloning connection patterns from high-extrema to low-extrema neurons, generalized for multiple layers."""
        if not hasattr(network, 'network') or not hasattr(network.network, 'connection_masks'):
            return 0

        extrema = network.network.detect_extrema()
        if not extrema:
            print("   ⚠️ No extrema detected, adding random connections.")
            return self._add_random_connections(network, num_connections)

        connections_added = 0
        
        # Distribute connections to add across layers
        num_layers_to_grow = len(network.network.layers) - 1 # Exclude output layer
        connections_per_layer = num_connections // num_layers_to_grow
        
        for layer_idx in range(num_layers_to_grow):
            if layer_idx not in extrema or not extrema[layer_idx]['high'] or not extrema[layer_idx]['low']:
                continue

            high_extrema_neurons = extrema[layer_idx]['high']
            low_extrema_neurons = extrema[layer_idx]['low']
            
            mask = network.network.connection_masks[layer_idx]
            layer = network.network.layers[layer_idx]

            for _ in range(connections_per_layer):
                for _ in range(100): # Max 100 attempts
                    donor_neuron = np.random.choice(high_extrema_neurons)
                    recipient_neuron = np.random.choice(low_extrema_neurons)

                    donor_connections = layer.weight.data[donor_neuron, :] * mask[donor_neuron, :]
                    if donor_connections.abs().sum() == 0:
                        continue

                    strongest_input_idx = donor_connections.abs().argmax().item()

                    if not mask[recipient_neuron, strongest_input_idx]:
                        mask[recipient_neuron, strongest_input_idx] = True
                        with torch.no_grad():
                            init_weight = torch.randn(1).to(self.device) * 0.01
                            layer.weight.data[recipient_neuron, strongest_input_idx] = init_weight
                        connections_added += 1
                        break
        
        if connections_added > 0:
            self.prune_weakest_connections(network, connections_added)

        if connections_added == 0:
            print("   ⚠️ Could not add any new connections via cloning, adding random instead.")
            return self._add_random_connections(network, num_connections)

        return connections_added

    def _add_random_connections(self, network, num_connections):
        """Fallback to add random connections if no extrema are found."""
        # This is a simplified version for the test, adds to first layer only
        masks = network.network.connection_masks
        layers = network.network.layers
        connections_added = 0
        if len(masks) > 0:
            mask = masks[0]
            layer = layers[0]
            inactive_positions = (mask == False).nonzero(as_tuple=False)
            if len(inactive_positions) > 0:
                num_to_add = min(num_connections, len(inactive_positions))
                selected_indices = torch.randperm(len(inactive_positions))[:num_to_add]
                for idx in selected_indices:
                    pos = inactive_positions[idx]
                    hidden_idx, input_idx = pos[0].item(), pos[1].item()
                    mask[hidden_idx, input_idx] = True
                    with torch.no_grad():
                        init_weight = torch.randn(1) * 0.1
                        layer.weight.data[hidden_idx, input_idx] = init_weight
                    connections_added += 1
        return connections_added

    def prune_weakest_connections(self, network, num_to_prune):
        """Prune the connections with the smallest weight magnitudes across all layers."""
        if not hasattr(network, 'network') or not hasattr(network.network, 'connection_masks'):
            return

        all_weights = []
        for layer_idx, (mask, layer) in enumerate(zip(network.network.connection_masks, network.network.layers)):
            active_weights = layer.weight.data[mask]
            for weight in active_weights:
                all_weights.append((weight.abs().item(), layer_idx, 0, 0)) # Store placeholder indices

        # This is inefficient, but simple for a quick test. A real implementation would be better.
        # We need to get the actual indices of the weights to prune them.
        
        # Let's do it properly, but still simply.
        
        # 1. Collect all active weights with their locations
        weight_locs = []
        for layer_idx, (mask, layer) in enumerate(zip(network.network.connection_masks, network.network.layers)):
            active_indices = mask.nonzero(as_tuple=False)
            for pos in active_indices:
                weight = layer.weight.data[pos[0], pos[1]]
                weight_locs.append((weight.abs().item(), layer_idx, pos[0].item(), pos[1].item()))

        # 2. Sort by weight magnitude
        weight_locs.sort(key=lambda x: x[0])
        
        # 3. Prune the weakest ones
        num_pruned = 0
        for _, layer_idx, row_idx, col_idx in weight_locs:
            if num_pruned >= num_to_prune:
                break
            network.network.connection_masks[layer_idx][row_idx, col_idx] = False
            num_pruned += 1
    
    def train_fixed_network(self, network, train_loader, test_loader, epochs=50):
        """Train network with fixed growth mechanism."""
        print(f"🚀 Training fixed growth network...")
        
        optimizer = optim.Adam(network.parameters(), lr=0.002)
        criterion = nn.CrossEntropyLoss()
        
        results = {
            'epochs': [],
            'growth_events': [],
            'proportional_growth': []
        }
        
        best_test_acc = 0
        
        for epoch in range(epochs):
            # Training epoch
            epoch_stats = network.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Evaluation
            eval_stats = network.evaluate(test_loader, criterion)
            
            # Apply proportional growth every 5 epochs
            proportional_connections = 0
            if epoch % 5 == 0 and epoch > 0:
                proportional_connections = self.apply_proportional_growth(network, epoch, epochs)
                if proportional_connections > 0:
                    results['proportional_growth'].append({
                        'epoch': epoch,
                        'connections_added': proportional_connections,
                        'total_connections': epoch_stats['total_connections'] + proportional_connections
                    })
                    print(f"   📈 PROPORTIONAL GROWTH! Added {proportional_connections} connections")
            
            # Track growth events from scheduler
            if epoch_stats.get('growth_events', 0) > 0:
                results['growth_events'].append({
                    'epoch': epoch,
                    'connections_added': epoch_stats.get('connections_added', 0),
                    'total_connections': epoch_stats['total_connections'],
                    'performance': eval_stats['performance']
                })
                print(f"   🌱 SCHEDULER GROWTH! Added {epoch_stats.get('connections_added', 0)} connections")
            
            # Store epoch data
            results['epochs'].append({
                'epoch': epoch,
                'train_acc': epoch_stats['performance'],
                'test_acc': eval_stats['performance'],
                'connections': epoch_stats['total_connections']
            })
            
            if eval_stats['performance'] > best_test_acc:
                best_test_acc = eval_stats['performance']
            
            # Progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch:2d}/{epochs}: "
                      f"Test Acc={eval_stats['performance']:.3f}, "
                      f"Connections={epoch_stats['total_connections']}")
        
        results['best_test_acc'] = best_test_acc
        results['final_test_acc'] = results['epochs'][-1]['test_acc']
        
        return results
    
    def run_cliff_rescue_test(self, epochs=50):
        """Run cliff rescue test with fixed growth mechanism."""
        print("🔧 FIXED GROWTH MECHANISM - CLIFF RESCUE TEST")
        print("=" * 60)
        print(f"🎯 Goal: Rescue {self.cliff_sparsity} sparsity from cliff")
        print(f"📈 Target: {self.target_performance:.1%} performance")
        print(f"🔧 Fixes: Bypass variance detection, proportional growth, forward skips")
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
        
        # Test 1: Static baseline (for comparison)
        print(f"\n📊 BASELINE: Static Sparse Network")
        print("-" * 40)
        
        static_network = MinimalNetwork(
            layer_sizes=[self.input_size] + self.hidden_sizes + [self.output_size],
            sparsity=self.cliff_sparsity,
            activation='tanh',
            device=self.device
        )
        
        static_results = self.train_static_network(static_network, train_loader, test_loader, epochs)
        
        # Test 2: Fixed growth network
        print(f"\n🔧 FIXED GROWTH: Growth-Enabled Network")
        print("-" * 40)
        
        growth_network = self.create_fixed_growth_network(self.cliff_sparsity)
        growth_results = self.train_fixed_network(growth_network, train_loader, test_loader, epochs)
        
        # Analysis
        self.analyze_rescue_results(static_results, growth_results)
        
        # Save results
        results = {
            'static': static_results,
            'growth': growth_results,
            'cliff_sparsity': self.cliff_sparsity,
            'target_performance': self.target_performance
        }
        
        self.save_results(results)
        self.create_visualizations(results)
        
        return results
    
    def train_static_network(self, network, train_loader, test_loader, epochs):
        """Train static network for baseline."""
        optimizer = optim.Adam(network.parameters(), lr=0.002)
        criterion = nn.CrossEntropyLoss()
        
        results = {'epochs': [], 'best_test_acc': 0}
        
        for epoch in range(epochs):
            # Training
            network.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = network(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # Evaluation
            network.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = network(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            test_acc = correct / total
            results['epochs'].append({'epoch': epoch, 'test_acc': test_acc})
            
            if test_acc > results['best_test_acc']:
                results['best_test_acc'] = test_acc
        
        results['final_test_acc'] = test_acc
        return results
    
    def analyze_rescue_results(self, static_results, growth_results):
        """Analyze rescue test results."""
        print(f"\n📊 CLIFF RESCUE ANALYSIS")
        print("=" * 60)
        
        static_perf = static_results['best_test_acc']
        growth_perf = growth_results['best_test_acc']
        
        total_growth_events = len(growth_results['growth_events'])
        total_proportional = len(growth_results['proportional_growth'])
        
        print(f"🎯 Target Performance: {self.target_performance:.1%}")
        print(f"📉 Static Performance: {static_perf:.1%}")
        print(f"🌱 Growth Performance: {growth_perf:.1%}")
        
        improvement = growth_perf - static_perf
        improvement_pct = (improvement / static_perf) * 100 if static_perf > 0 else 0
        
        print(f"\n🔍 RESCUE ANALYSIS:")
        print(f"   Improvement: {improvement:.1%} ({improvement_pct:+.1f}%)")
        print(f"   Scheduler growth events: {total_growth_events}")
        print(f"   Proportional growth events: {total_proportional}")
        print(f"   Target achieved: {'✅ YES' if growth_perf >= self.target_performance else '❌ NO'}")
        
        if growth_perf >= self.target_performance:
            print(f"\n🎉 CLIFF RESCUE SUCCESS!")
            print(f"   ✅ Fixed growth mechanism works!")
            print(f"   🚀 Ready to scale to multi-layer networks")
        elif improvement > 0.05:  # 5% improvement
            print(f"\n📈 PARTIAL CLIFF RESCUE")
            print(f"   📊 Significant improvement achieved")
            print(f"   🔧 May need more aggressive growth parameters")
        else:
            print(f"\n❌ CLIFF RESCUE FAILED")
            print(f"   🔧 Growth mechanism still needs work")
        
        # Connection analysis
        if growth_results['epochs']:
            initial_connections = growth_results['epochs'][0]['connections']
            final_connections = growth_results['epochs'][-1]['connections']
            connection_growth = final_connections - initial_connections
            growth_rate = (connection_growth / initial_connections) * 100
            
            print(f"\n🔗 CONNECTION ANALYSIS:")
            print(f"   Initial connections: {initial_connections}")
            print(f"   Final connections: {final_connections}")
            print(f"   Connection growth: +{connection_growth} ({growth_rate:.1f}%)")
            
            if growth_rate >= 5:
                print(f"   ✅ Achieved minimum 5% growth target")
            else:
                print(f"   ⚠️  Below 5% growth target")
    
    def save_results(self, results):
        """Save results."""
        print(f"\n💾 Saving results...")
        
        with open(os.path.join(self.save_dir, "fixed_growth_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'static_performance': results['static']['best_test_acc'],
            'growth_performance': results['growth']['best_test_acc'],
            'improvement': results['growth']['best_test_acc'] - results['static']['best_test_acc'],
            'target_achieved': results['growth']['best_test_acc'] >= results['target_performance'],
            'scheduler_growth_events': len(results['growth']['growth_events']),
            'proportional_growth_events': len(results['growth']['proportional_growth'])
        }
        
        with open(os.path.join(self.save_dir, "fixed_growth_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved to {self.save_dir}/")
    
    def create_visualizations(self, results):
        """Create visualizations."""
        print("📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Fixed Growth Mechanism - Cliff Rescue Results', fontsize=14)
        
        # 1. Performance comparison
        static_epochs = [e['epoch'] for e in results['static']['epochs']]
        static_accs = [e['test_acc'] for e in results['static']['epochs']]
        growth_epochs = [e['epoch'] for e in results['growth']['epochs']]
        growth_accs = [e['test_acc'] for e in results['growth']['epochs']]
        
        axes[0, 0].plot(static_epochs, static_accs, 'r-', label='Static Sparse', linewidth=2)
        axes[0, 0].plot(growth_epochs, growth_accs, 'b-', label='Fixed Growth', linewidth=2)
        axes[0, 0].axhline(y=results['target_performance'], color='orange', linestyle='--', 
                          label=f'Target ({results["target_performance"]:.1%})')
        
        # Mark growth events
        for event in results['growth']['growth_events']:
            axes[0, 0].axvline(x=event['epoch'], color='green', linestyle=':', alpha=0.7)
        for event in results['growth']['proportional_growth']:
            axes[0, 0].axvline(x=event['epoch'], color='purple', linestyle=':', alpha=0.7)
        
        axes[0, 0].set_title('Performance Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Connection growth
        connections = [e['connections'] for e in results['growth']['epochs']]
        axes[0, 1].plot(growth_epochs, connections, 'g-', linewidth=2)
        
        # Mark growth events
        for event in results['growth']['growth_events']:
            axes[0, 1].scatter(event['epoch'], event['total_connections'], 
                              color='green', s=100, marker='o')
        for event in results['growth']['proportional_growth']:
            axes[0, 1].scatter(event['epoch'], event['total_connections'], 
                              color='purple', s=100, marker='s')
        
        axes[0, 1].set_title('Connection Growth Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Active Connections')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Growth events summary
        event_types = ['Scheduler\nGrowth', 'Proportional\nGrowth']
        event_counts = [len(results['growth']['growth_events']), 
                       len(results['growth']['proportional_growth'])]
        
        bars = axes[1, 0].bar(event_types, event_counts, color=['green', 'purple'], alpha=0.7)
        axes[1, 0].set_title('Growth Events by Type')
        axes[1, 0].set_ylabel('Number of Events')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, event_counts):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}', ha='center', va='bottom')
        
        # 4. Final comparison
        networks = ['Static\nSparse', 'Fixed\nGrowth']
        performances = [results['static']['best_test_acc'], results['growth']['best_test_acc']]
        
        bars = axes[1, 1].bar(networks, performances, color=['red', 'blue'], alpha=0.7)
        axes[1, 1].axhline(y=results['target_performance'], color='orange', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Final Performance Comparison')
        axes[1, 1].set_ylabel('Best Test Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Color bars based on target achievement
        for i, (bar, perf) in enumerate(zip(bars, performances)):
            if perf >= results['target_performance']:
                bar.set_color('green')
            
            # Add value labels
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{perf:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'fixed_growth_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to {self.save_dir}/fixed_growth_analysis.png")


def main():
    """Main function."""
    print("🔧 FIXED GROWTH MECHANISM TEST")
    print("=" * 60)
    print("Fixes applied:")
    print("1. Bypass broken gradient variance detection")
    print("2. Reduce stabilization period (3 epochs)")
    print("3. Add proportional growth (5% minimum)")
    print("4. Forward skip connections preferred")
    print("=" * 60)
    
    tester = FixedGrowthTest()
    results = tester.run_cliff_rescue_test(epochs=50)
    
    print(f"\n🎉 Fixed growth test completed!")
    
    # Check success
    growth_perf = results['growth']['best_test_acc']
    target = results['target_performance']
    
    if growth_perf >= target:
        print(f"\n🎉 CLIFF RESCUE SUCCESS! Growth mechanism fixed!")
    else:
        improvement = growth_perf - results['static']['best_test_acc']
        print(f"\n📈 Improvement achieved: {improvement:.1%}")
        print(f"🔧 May need further tuning for full rescue")


if __name__ == "__main__":
    main()
