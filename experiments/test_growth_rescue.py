#!/usr/bin/env python3
"""
Growth Rescue Test - The Most Diagnostic Experiment

Test the core hypothesis: Can growth rescue performance from the cliff?

Cliff Test:
- Static 0.002 sparsity: 18% accuracy (below 35.6% target)
- Growth-enabled 0.002 sparsity: Can it reach 35%+ ?

This is the most diagnostic test of the growth mechanism.
If growth rescues performance from cliff → mechanism works
If growth fails to rescue → mechanism needs work
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

class GrowthRescueTest:
    """Test if growth can rescue performance from the sparsity cliff."""
    
    def __init__(self, save_dir="growth_rescue_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Use RTX 2060 SUPER for consistency
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Test parameters from cliff analysis
        self.cliff_sparsity = 0.002  # 18% performance - below target
        self.target_performance = 0.356  # 35.6% - 50% of dense baseline
        self.safe_sparsity = 0.005  # 39.3% performance - above target
        
        # Architecture: single layer [256] for focused test
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
    
    def create_static_sparse_network(self, sparsity):
        """Create static sparse network (no growth)."""
        layer_sizes = [self.input_size, self.hidden_size, self.output_size]
        
        network = MinimalNetwork(
            layer_sizes=layer_sizes,
            sparsity=sparsity,
            activation='tanh',
            device=self.device
        )
        
        stats = network.get_connectivity_stats()
        print(f"🕸️  Static sparse network (sparsity={sparsity}): {stats['total_active_connections']} connections")
        
        return network
    
    def create_growth_enabled_network(self, sparsity):
        """Create growth-enabled network."""
        network = create_multi_scale_network(
            self.input_size, [self.hidden_size], self.output_size,
            sparsity=sparsity,
            activation='tanh',
            device=self.device,
            snapshot_dir=os.path.join(self.save_dir, "growth_snapshots")
        )
        
        stats = network.network.get_connectivity_stats()
        print(f"🌱 Growth-enabled network (sparsity={sparsity}): {stats['total_active_connections']} connections")
        
        return network
    
    def train_static_network(self, network, train_loader, test_loader, epochs=50, lr=0.002, name="Static"):
        """Train static sparse network."""
        print(f"🚀 Training {name}...")
        
        optimizer = optim.Adam(network.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        connections_history = []
        
        best_test_acc = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            network.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for data, target in train_loader:
                optimizer.zero_grad()
                output = network(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Testing
            network.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    output = network(data)
                    loss = criterion(output, target)
                    
                    test_loss += loss.item()
                    pred = output.argmax(dim=1)
                    test_correct += pred.eq(target).sum().item()
                    test_total += target.size(0)
            
            test_loss /= len(test_loader)
            test_acc = test_correct / test_total
            
            # Track connections (static - should stay constant)
            stats = network.get_connectivity_stats()
            connections = stats['total_active_connections']
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            connections_history.append(connections)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch:2d}/{epochs}: "
                      f"Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, "
                      f"Connections={connections}")
        
        total_time = time.time() - start_time
        final_stats = network.get_connectivity_stats()
        
        print(f"✅ {name} completed!")
        print(f"   Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.1f}%)")
        print(f"   Final connections: {final_stats['total_active_connections']}")
        print(f"   Training time: {total_time:.2f} seconds")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'connections_history': connections_history,
            'best_test_acc': best_test_acc,
            'final_test_acc': test_acc,
            'initial_connections': connections_history[0],
            'final_connections': final_stats['total_active_connections'],
            'connection_growth': final_stats['total_active_connections'] - connections_history[0],
            'training_time': total_time,
            'growth_events': 0  # Static network has no growth
        }
    
    def train_growth_network(self, network, train_loader, test_loader, epochs=50, lr=0.002, name="Growth"):
        """Train growth-enabled network."""
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
                      f"(total: {epoch_stats['total_connections']})")
            
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
    
    def run_cliff_rescue_test(self, epochs=50):
        """Run the complete cliff rescue test."""
        print("🔬 GROWTH RESCUE TEST - THE DIAGNOSTIC EXPERIMENT")
        print("=" * 70)
        print(f"🎯 HYPOTHESIS: Can growth rescue performance from the cliff?")
        print(f"📉 Cliff sparsity: {self.cliff_sparsity} (static performance: ~18%)")
        print(f"🎯 Target performance: {self.target_performance:.1%} (50% of dense)")
        print(f"🧪 Test: Static vs Growth-enabled at cliff sparsity")
        print("=" * 70)
        
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
        
        # Test 1: Static sparse network at cliff sparsity
        print(f"\n🧪 TEST 1: STATIC SPARSE NETWORK (CLIFF BASELINE)")
        print("-" * 50)
        
        static_network = self.create_static_sparse_network(self.cliff_sparsity)
        static_results = self.train_static_network(
            static_network, train_loader, test_loader,
            epochs=epochs, name=f"Static {self.cliff_sparsity}"
        )
        
        self.results['static_cliff'] = {
            'sparsity': self.cliff_sparsity,
            'network_type': 'static',
            **static_results
        }
        
        # Test 2: Growth-enabled network at cliff sparsity
        print(f"\n🌱 TEST 2: GROWTH-ENABLED NETWORK (RESCUE ATTEMPT)")
        print("-" * 50)
        
        growth_network = self.create_growth_enabled_network(self.cliff_sparsity)
        growth_results = self.train_growth_network(
            growth_network, train_loader, test_loader,
            epochs=epochs, name=f"Growth {self.cliff_sparsity}"
        )
        
        self.results['growth_cliff'] = {
            'sparsity': self.cliff_sparsity,
            'network_type': 'growth',
            **growth_results
        }
        
        # Test 3: Static sparse network at safe sparsity (reference)
        print(f"\n📊 TEST 3: STATIC SPARSE NETWORK (SAFE REFERENCE)")
        print("-" * 50)
        
        safe_network = self.create_static_sparse_network(self.safe_sparsity)
        safe_results = self.train_static_network(
            safe_network, train_loader, test_loader,
            epochs=epochs, name=f"Static {self.safe_sparsity}"
        )
        
        self.results['static_safe'] = {
            'sparsity': self.safe_sparsity,
            'network_type': 'static',
            **safe_results
        }
        
        # Analysis
        self.analyze_rescue_results()
        
        # Save and visualize
        self.save_results()
        self.create_visualizations()
        
        return self.results
    
    def analyze_rescue_results(self):
        """Analyze the cliff rescue test results."""
        print(f"\n📊 CLIFF RESCUE ANALYSIS")
        print("=" * 70)
        
        static_cliff_perf = self.results['static_cliff']['best_test_acc']
        growth_cliff_perf = self.results['growth_cliff']['best_test_acc']
        safe_perf = self.results['static_safe']['best_test_acc']
        
        growth_events = self.results['growth_cliff']['growth_events']
        connection_growth = self.results['growth_cliff']['connection_growth']
        
        print(f"🎯 Target Performance: {self.target_performance:.1%}")
        print(f"📊 Reference (Safe Sparsity): {safe_perf:.1%}")
        print(f"📉 Static Cliff Performance: {static_cliff_perf:.1%}")
        print(f"🌱 Growth Cliff Performance: {growth_cliff_perf:.1%}")
        
        # Calculate rescue metrics
        rescue_improvement = growth_cliff_perf - static_cliff_perf
        rescue_percentage = (rescue_improvement / static_cliff_perf) * 100 if static_cliff_perf > 0 else 0
        target_achieved = growth_cliff_perf >= self.target_performance
        
        print(f"\n🔍 RESCUE ANALYSIS:")
        print(f"   Improvement: {rescue_improvement:.1%} ({rescue_percentage:+.1f}%)")
        print(f"   Growth events: {growth_events}")
        print(f"   Connection growth: +{connection_growth} connections")
        print(f"   Target achieved: {'✅ YES' if target_achieved else '❌ NO'}")
        
        # Determine verdict
        if target_achieved:
            print(f"\n🎉 GROWTH RESCUE SUCCESS!")
            print(f"   ✅ Growth mechanism WORKS - rescued performance from cliff")
            print(f"   ✅ Achieved target performance: {growth_cliff_perf:.1%} ≥ {self.target_performance:.1%}")
            print(f"   🌱 Growth added {connection_growth} connections in {growth_events} events")
            verdict = "SUCCESS"
        elif rescue_improvement > 0.05:  # 5% improvement
            print(f"\n📈 PARTIAL GROWTH RESCUE")
            print(f"   📊 Growth mechanism shows promise - significant improvement")
            print(f"   ⚠️  Didn't reach target but improved by {rescue_improvement:.1%}")
            print(f"   🔧 Recommendation: Tune growth parameters (thresholds, rates)")
            verdict = "PARTIAL"
        else:
            print(f"\n❌ GROWTH RESCUE FAILED")
            print(f"   ❌ Growth mechanism needs work - minimal improvement")
            print(f"   🔧 Recommendation: Debug growth triggers and connection routing")
            print(f"   🔍 Check: gradient variance detection, extrema thresholds")
            verdict = "FAILED"
        
        # Connection efficiency analysis
        static_efficiency = static_cliff_perf / self.results['static_cliff']['final_connections']
        growth_efficiency = growth_cliff_perf / self.results['growth_cliff']['final_connections']
        
        print(f"\n⚡ EFFICIENCY ANALYSIS:")
        print(f"   Static efficiency: {static_efficiency:.6f} performance/connection")
        print(f"   Growth efficiency: {growth_efficiency:.6f} performance/connection")
        
        if growth_efficiency > static_efficiency:
            print(f"   ✅ Growth is more efficient - better performance per connection")
        else:
            print(f"   ⚠️  Growth is less efficient - added connections didn't help enough")
        
        # Store verdict
        self.results['verdict'] = {
            'rescue_success': target_achieved,
            'rescue_improvement': rescue_improvement,
            'rescue_percentage': rescue_percentage,
            'growth_events': growth_events,
            'connection_growth': connection_growth,
            'verdict': verdict,
            'static_efficiency': static_efficiency,
            'growth_efficiency': growth_efficiency
        }
        
        print(f"\n💡 NEXT STEPS:")
        if verdict == "SUCCESS":
            print(f"   🚀 Scale to multi-layer networks with confidence")
            print(f"   📊 Test growth on higher sparsity levels")
            print(f"   🔬 Optimize growth parameters for even better performance")
        elif verdict == "PARTIAL":
            print(f"   🔧 Tune growth scheduler thresholds")
            print(f"   ⚡ Increase growth rate (more connections per event)")
            print(f"   🎯 Lower growth triggers for more frequent growth")
        else:
            print(f"   🔍 Debug growth mechanism fundamentals")
            print(f"   📊 Verify gradient variance spike detection")
            print(f"   🎯 Check extrema detection and connection routing")
    
    def save_results(self):
        """Save experiment results."""
        print(f"\n💾 Saving results...")
        
        # Save detailed results
        with open(os.path.join(self.save_dir, "cliff_rescue_results.json"), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'cliff_sparsity': self.cliff_sparsity,
            'target_performance': self.target_performance,
            'static_cliff_performance': self.results['static_cliff']['best_test_acc'],
            'growth_cliff_performance': self.results['growth_cliff']['best_test_acc'],
            'safe_reference_performance': self.results['static_safe']['best_test_acc'],
            'verdict': self.results['verdict']
        }
        
        with open(os.path.join(self.save_dir, "cliff_rescue_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved to {self.save_dir}/")
    
    def create_visualizations(self):
        """Create cliff rescue visualizations."""
        print("📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Growth Rescue Test - Cliff Diagnostic Results', fontsize=16)
        
        # 1. Performance comparison
        networks = ['Static Cliff', 'Growth Cliff', 'Static Safe']
        performances = [
            self.results['static_cliff']['best_test_acc'],
            self.results['growth_cliff']['best_test_acc'],
            self.results['static_safe']['best_test_acc']
        ]
        colors = ['red', 'blue', 'green']
        
        bars = axes[0, 0].bar(networks, performances, color=colors, alpha=0.7)
        axes[0, 0].axhline(y=self.target_performance, color='orange', linestyle='--', 
                          label=f'Target ({self.target_performance:.1%})')
        axes[0, 0].set_title('Performance Comparison')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, perf in zip(bars, performances):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{perf:.3f}', ha='center', va='bottom')
        
        # 2. Training curves
        epochs_static = range(len(self.results['static_cliff']['test_accs']))
        epochs_growth = range(len(self.results['growth_cliff']['test_accs']))
        epochs_safe = range(len(self.results['static_safe']['test_accs']))
        
        axes[0, 1].plot(epochs_static, self.results['static_cliff']['test_accs'], 
                       'r-', label='Static Cliff', linewidth=2)
        axes[0, 1].plot(epochs_growth, self.results['growth_cliff']['test_accs'], 
                       'b-', label='Growth Cliff', linewidth=2)
        axes[0, 1].plot(epochs_safe, self.results['static_safe']['test_accs'], 
                       'g-', label='Static Safe', linewidth=2)
        axes[0, 1].axhline(y=self.target_performance, color='orange', linestyle='--', alpha=0.5)
        
        # Mark growth events
        if 'growth_event_details' in self.results['growth_cliff']:
            for event in self.results['growth_cliff']['growth_event_details']:
                axes[0, 1].axvline(x=event['epoch'], color='blue', linestyle=':', alpha=0.7)
                axes[0, 1].annotate(f"+{event['connections_added']}", 
                                  xy=(event['epoch'], event['performance']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[0, 1].set_title('Training Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Connection growth
        axes[0, 2].plot(epochs_static, self.results['static_cliff']['connections_history'], 
                       'r-', label='Static Cliff', linewidth=2)
        axes[0, 2].plot(epochs_growth, self.results['growth_cliff']['connections_history'], 
                       'b-', label='Growth Cliff', linewidth=2)
        axes[0, 2].plot(epochs_safe, self.results['static_safe']['connections_history'], 
                       'g-', label='Static Safe', linewidth=2)
        
        axes[0, 2].set_title('Connection Evolution')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Active Connections')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Rescue metrics
        rescue_improvement = self.results['verdict']['rescue_improvement']
        rescue_percentage = self.results['verdict']['rescue_percentage']
        
        metrics = ['Improvement', 'Percentage', 'Growth Events']
        values = [rescue_improvement * 100, rescue_percentage, self.results['verdict']['growth_events']]
        
        axes[1, 0].bar(metrics, values, alpha=0.7, color='purple')
        axes[1, 0].set_title('Rescue Metrics')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Efficiency comparison
        static_eff = self.results['verdict']['static_efficiency']
        growth_eff = self.results['verdict']['growth_efficiency']
        
        eff_networks = ['Static Cliff', 'Growth Cliff']
        efficiencies = [static_eff, growth_eff]
        
        axes[1, 1].bar(eff_networks, efficiencies, alpha=0.7, color=['red', 'blue'])
        axes[1, 1].set_title('Efficiency (Performance/Connection)')
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Verdict summary
        verdict = self.results['verdict']['verdict']
        verdict_color = {'SUCCESS': 'green', 'PARTIAL': 'orange', 'FAILED': 'red'}
        
        axes[1, 2].text(0.5, 0.7, f"VERDICT", ha='center', va='center', 
                       transform=axes[1, 2].transAxes, fontsize=16, fontweight='bold')
        axes[1, 2].text(0.5, 0.5, verdict, ha='center', va='center', 
                       transform=axes[1, 2].transAxes, fontsize=20, fontweight='bold',
                       color=verdict_color.get(verdict, 'black'))
        axes[1, 2].text(0.5, 0.3, f"Growth: {rescue_improvement:.1%}", ha='center', va='center', 
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Growth Rescue Test Result')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'cliff_rescue_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to {self.save_dir}/cliff_rescue_analysis.png")


def main():
    """Main function for cliff rescue test."""
    parser = argparse.ArgumentParser(description='Growth Rescue Test - Cliff Diagnostic')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--save-dir', type=str, default='growth_rescue_results',
                       help='Directory to save results (default: growth_rescue_results)')
    
    args = parser.parse_args()
    
    print("🔬 GROWTH RESCUE TEST - THE DIAGNOSTIC EXPERIMENT")
    print("=" * 70)
    print("🎯 HYPOTHESIS: Can growth rescue performance from the cliff?")
    print("📉 Testing at 0.002 sparsity (cliff point: ~18% performance)")
    print("🎯 Target: Rescue to 35.6%+ (50% of dense baseline)")
    print("=" * 70)
    
    test = GrowthRescueTest(save_dir=args.save_dir)
    results = test.run_cliff_rescue_test(epochs=args.epochs)
    
    print(f"\n🎉 Growth rescue test completed!")
    print(f"📁 Results saved to {args.save_dir}/")
    
    # Print final verdict
    verdict = results['verdict']['verdict']
    if verdict == "SUCCESS":
        print(f"\n🎉 GROWTH MECHANISM WORKS! Ready to scale up.")
    elif verdict == "PARTIAL":
        print(f"\n📈 GROWTH SHOWS PROMISE! Tune parameters and retry.")
    else:
        print(f"\n🔧 GROWTH NEEDS WORK! Debug mechanism before scaling.")


if __name__ == "__main__":
    main()
