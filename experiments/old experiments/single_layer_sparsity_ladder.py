#!/usr/bin/env python3
"""
Single Layer Sparsity Ladder Experiment

Phase 1: Coarse-to-Fine Validation
- Start with simple single layer [256] architecture
- Test sparsity ladder to find the "Goldilocks zone"
- Fast iteration, clear signal, easier debugging
- Target: Find sparsity where performance ≥ 50% of dense baseline

This gives us sparsity intuition before scaling to multi-layer complexity.
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
from src.structure_net.core.minimal_network import MinimalNetwork

class SingleLayerSparsityExperiment:
    """Single layer sparsity ladder experiment for fast validation."""
    
    def __init__(self, save_dir="single_layer_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Use RTX 2060 SUPER for consistency
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Results storage
        self.results = {}
        
        # Sparsity ladder - from coarse to fine
        self.sparsity_ladder = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0001]
        
        # Architecture: single layer [256]
        self.input_size = 784
        self.hidden_size = 256
        self.output_size = 10
    
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
                # Create complex patterns
                base_pattern = torch.randn(self.input_size) * 0.8
                
                # Add class-specific structure (same as validation)
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
        print(f"   Input range: [{X.min():.2f}, {X.max():.2f}]")
        print(f"   Input std: {X.std():.2f}")
        
        return X, y
    
    def create_dense_baseline(self):
        """Create dense single layer network for baseline."""
        network = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.output_size)
        ).to(self.device)
        
        total_params = sum(p.numel() for p in network.parameters())
        print(f"📐 Dense baseline: {total_params:,} parameters")
        
        return network
    
    def create_sparse_network(self, sparsity):
        """Create sparse single layer network."""
        # Create minimal network with single hidden layer
        layer_sizes = [self.input_size, self.hidden_size, self.output_size]
        
        network = MinimalNetwork(
            layer_sizes=layer_sizes,
            sparsity=sparsity,
            activation='tanh',
            device=self.device
        )
        
        stats = network.get_connectivity_stats()
        print(f"🕸️  Sparse network (sparsity={sparsity}): {stats['total_active_connections']} connections")
        print(f"   Connectivity ratio: {stats['connectivity_ratio']:.6f}")
        
        return network
    
    def train_network(self, network, train_loader, test_loader, epochs=30, lr=0.002, name="network"):
        """Train and evaluate a network."""
        print(f"🚀 Training {name}...")
        
        optimizer = optim.Adam(network.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        
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
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            # Progress reporting (every 10 epochs)
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch:2d}/{epochs}: "
                      f"Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
        
        total_time = time.time() - start_time
        
        print(f"✅ {name} completed!")
        print(f"   Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.1f}%)")
        print(f"   Training time: {total_time:.2f} seconds")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'best_test_acc': best_test_acc,
            'final_test_acc': test_acc,
            'training_time': total_time
        }
    
    def run_sparsity_ladder(self, epochs=30):
        """Run the complete sparsity ladder experiment."""
        print("🔬 SINGLE LAYER SPARSITY LADDER EXPERIMENT")
        print("=" * 60)
        print(f"Architecture: [{self.input_size}] → [{self.hidden_size}] → [{self.output_size}]")
        print(f"Sparsity ladder: {self.sparsity_ladder}")
        print(f"Target: Find sparsity where performance ≥ 50% of dense baseline")
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
        
        # 1. Dense baseline
        print(f"\n🏗️  PHASE 1: DENSE BASELINE")
        print("-" * 40)
        
        dense_network = self.create_dense_baseline()
        dense_results = self.train_network(
            dense_network, train_loader, test_loader, 
            epochs=epochs, name="Dense Baseline"
        )
        
        dense_baseline = dense_results['best_test_acc']
        target_performance = dense_baseline * 0.5  # 50% of dense
        
        self.results['dense_baseline'] = {
            'architecture': [self.hidden_size],
            'parameters': sum(p.numel() for p in dense_network.parameters()),
            **dense_results
        }
        
        print(f"\n📊 Dense baseline: {dense_baseline:.4f} ({dense_baseline*100:.1f}%)")
        print(f"🎯 Target performance (50% of dense): {target_performance:.4f} ({target_performance*100:.1f}%)")
        
        # 2. Sparsity ladder
        print(f"\n🪜 PHASE 2: SPARSITY LADDER")
        print("-" * 40)
        
        sparsity_results = []
        goldilocks_candidates = []
        
        for i, sparsity in enumerate(self.sparsity_ladder):
            print(f"\n🔍 Testing sparsity {i+1}/{len(self.sparsity_ladder)}: {sparsity} ({sparsity*100:.1f}%)")
            
            # Create sparse network
            sparse_network = self.create_sparse_network(sparsity)
            
            # Get connection count
            stats = sparse_network.get_connectivity_stats()
            connections = stats['total_active_connections']
            
            # Train sparse network
            sparse_results = self.train_network(
                sparse_network, train_loader, test_loader,
                epochs=epochs, name=f"Sparse {sparsity}"
            )
            
            # Calculate performance metrics
            performance = sparse_results['best_test_acc']
            dense_ratio = performance / dense_baseline
            efficiency = performance / connections  # Performance per connection
            
            result = {
                'sparsity': sparsity,
                'connections': connections,
                'connectivity_ratio': stats['connectivity_ratio'],
                'performance': performance,
                'dense_ratio': dense_ratio,
                'efficiency': efficiency,
                'meets_target': performance >= target_performance,
                **sparse_results
            }
            
            sparsity_results.append(result)
            self.results[f'sparsity_{sparsity}'] = result
            
            # Check if this is a Goldilocks candidate
            if performance >= target_performance:
                goldilocks_candidates.append((sparsity, performance, connections))
                print(f"   🎯 GOLDILOCKS CANDIDATE! Performance: {performance:.4f} ({dense_ratio*100:.1f}% of dense)")
            else:
                print(f"   📉 Below target: {performance:.4f} ({dense_ratio*100:.1f}% of dense)")
            
            print(f"   Efficiency: {efficiency:.6f} performance/connection")
        
        # 3. Analysis
        self.analyze_sparsity_ladder(sparsity_results, goldilocks_candidates, dense_baseline, target_performance)
        
        # 4. Save and visualize
        self.save_results()
        self.create_visualizations(sparsity_results, dense_baseline, target_performance)
        
        return sparsity_results, goldilocks_candidates
    
    def analyze_sparsity_ladder(self, results, goldilocks_candidates, dense_baseline, target_performance):
        """Analyze sparsity ladder results."""
        print(f"\n📊 SPARSITY LADDER ANALYSIS")
        print("=" * 60)
        
        # Find performance cliff
        performances = [r['performance'] for r in results]
        sparsities = [r['sparsity'] for r in results]
        
        # Find where performance drops below target
        cliff_index = None
        for i, perf in enumerate(performances):
            if perf < target_performance:
                cliff_index = i
                break
        
        if cliff_index is not None and cliff_index > 0:
            cliff_sparsity = sparsities[cliff_index]
            safe_sparsity = sparsities[cliff_index - 1]
            print(f"📉 Performance cliff detected:")
            print(f"   Safe sparsity: {safe_sparsity} ({performances[cliff_index-1]:.4f} accuracy)")
            print(f"   Cliff sparsity: {cliff_sparsity} ({performances[cliff_index]:.4f} accuracy)")
        else:
            print(f"📈 No performance cliff found in tested range")
        
        # Goldilocks analysis
        if goldilocks_candidates:
            print(f"\n🎯 GOLDILOCKS ZONE FOUND!")
            print(f"   {len(goldilocks_candidates)} viable sparsity levels:")
            
            for sparsity, performance, connections in goldilocks_candidates:
                dense_ratio = performance / dense_baseline
                print(f"   - Sparsity {sparsity}: {performance:.4f} ({dense_ratio*100:.1f}% of dense, {connections} connections)")
            
            # Find most efficient
            best_efficiency = max(goldilocks_candidates, key=lambda x: x[1] / x[2])
            print(f"\n⭐ MOST EFFICIENT: Sparsity {best_efficiency[0]}")
            print(f"   Performance: {best_efficiency[1]:.4f}")
            print(f"   Connections: {best_efficiency[2]}")
            print(f"   Efficiency: {best_efficiency[1] / best_efficiency[2]:.6f} performance/connection")
            
        else:
            print(f"\n❌ NO GOLDILOCKS ZONE FOUND")
            print(f"   All tested sparsities below target performance")
            print(f"   Recommendation: Test higher sparsity levels (0.2, 0.5, 1.0)")
        
        # Connection-performance relationship
        connections = [r['connections'] for r in results]
        correlation = np.corrcoef(connections, performances)[0, 1]
        
        print(f"\n🔗 Connection-Performance Relationship:")
        print(f"   Correlation: {correlation:.3f}")
        if correlation > 0.8:
            print(f"   ✅ Strong positive correlation - more connections = better performance")
        elif correlation > 0.5:
            print(f"   📊 Moderate correlation - connections help but other factors matter")
        else:
            print(f"   ⚠️  Weak correlation - connections may not be the limiting factor")
        
        # Efficiency sweet spot
        efficiencies = [r['efficiency'] for r in results]
        max_efficiency_idx = np.argmax(efficiencies)
        max_efficiency_sparsity = sparsities[max_efficiency_idx]
        
        print(f"\n⚡ EFFICIENCY SWEET SPOT:")
        print(f"   Sparsity: {max_efficiency_sparsity}")
        print(f"   Performance: {performances[max_efficiency_idx]:.4f}")
        print(f"   Connections: {connections[max_efficiency_idx]}")
        print(f"   Efficiency: {efficiencies[max_efficiency_idx]:.6f}")
    
    def save_results(self):
        """Save experiment results."""
        print(f"\n💾 Saving results...")
        
        # Save detailed results
        with open(os.path.join(self.save_dir, "sparsity_ladder_results.json"), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'architecture': [self.input_size, self.hidden_size, self.output_size],
            'sparsity_ladder': self.sparsity_ladder,
            'dense_baseline': self.results['dense_baseline']['best_test_acc'],
            'target_performance': self.results['dense_baseline']['best_test_acc'] * 0.5,
            'goldilocks_candidates': [],
            'efficiency_ranking': []
        }
        
        # Find Goldilocks candidates and efficiency ranking
        sparsity_results = []
        for key, result in self.results.items():
            if key.startswith('sparsity_'):
                sparsity_results.append(result)
                
                if result['meets_target']:
                    summary['goldilocks_candidates'].append({
                        'sparsity': result['sparsity'],
                        'performance': result['performance'],
                        'connections': result['connections'],
                        'dense_ratio': result['dense_ratio']
                    })
        
        # Sort by efficiency
        sparsity_results.sort(key=lambda x: x['efficiency'], reverse=True)
        summary['efficiency_ranking'] = [
            {
                'sparsity': r['sparsity'],
                'efficiency': r['efficiency'],
                'performance': r['performance'],
                'connections': r['connections']
            }
            for r in sparsity_results[:5]  # Top 5
        ]
        
        with open(os.path.join(self.save_dir, "sparsity_ladder_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved to {self.save_dir}/")
    
    def create_visualizations(self, results, dense_baseline, target_performance):
        """Create sparsity ladder visualizations."""
        print("📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Single Layer Sparsity Ladder Results', fontsize=16)
        
        sparsities = [r['sparsity'] for r in results]
        performances = [r['performance'] for r in results]
        connections = [r['connections'] for r in results]
        efficiencies = [r['efficiency'] for r in results]
        dense_ratios = [r['dense_ratio'] for r in results]
        
        # 1. Performance vs Sparsity
        axes[0, 0].semilogx(sparsities, performances, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].axhline(y=dense_baseline, color='green', linestyle='--', label=f'Dense Baseline ({dense_baseline:.3f})')
        axes[0, 0].axhline(y=target_performance, color='red', linestyle='--', label=f'Target (50% of dense)')
        axes[0, 0].set_xlabel('Sparsity')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_title('Performance vs Sparsity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark Goldilocks zone
        for i, (sparsity, performance) in enumerate(zip(sparsities, performances)):
            if performance >= target_performance:
                axes[0, 0].plot(sparsity, performance, 'go', markersize=12, alpha=0.7)
        
        # 2. Performance vs Connections
        axes[0, 1].plot(connections, performances, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].axhline(y=target_performance, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Active Connections')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].set_title('Performance vs Connections')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Efficiency (Performance per Connection)
        axes[0, 2].semilogx(sparsities, efficiencies, 'mo-', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Sparsity')
        axes[0, 2].set_ylabel('Efficiency (Performance/Connection)')
        axes[0, 2].set_title('Efficiency vs Sparsity')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Mark efficiency peak
        max_eff_idx = np.argmax(efficiencies)
        axes[0, 2].plot(sparsities[max_eff_idx], efficiencies[max_eff_idx], 'go', markersize=12)
        
        # 4. Dense Performance Ratio
        axes[1, 0].semilogx(sparsities, [r*100 for r in dense_ratios], 'co-', linewidth=2, markersize=8)
        axes[1, 0].axhline(y=50, color='red', linestyle='--', label='50% Target')
        axes[1, 0].set_xlabel('Sparsity')
        axes[1, 0].set_ylabel('% of Dense Performance')
        axes[1, 0].set_title('Relative Performance vs Sparsity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Training curves for key sparsities
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        key_indices = [0, len(results)//4, len(results)//2, 3*len(results)//4, -1]  # Sample across range
        
        for i, idx in enumerate(key_indices):
            if idx < len(results) and i < len(colors):
                result = results[idx]
                epochs = range(len(result['test_accs']))
                label = f"Sparsity {result['sparsity']}"
                axes[1, 1].plot(epochs, result['test_accs'], color=colors[i], label=label, alpha=0.8)
        
        axes[1, 1].axhline(y=target_performance, color='red', linestyle=':', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Test Accuracy')
        axes[1, 1].set_title('Training Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Sparsity ladder summary
        axes[1, 2].bar(range(len(sparsities)), performances, alpha=0.7)
        axes[1, 2].axhline(y=target_performance, color='red', linestyle='--', alpha=0.7)
        axes[1, 2].set_xlabel('Sparsity Level')
        axes[1, 2].set_ylabel('Test Accuracy')
        axes[1, 2].set_title('Sparsity Ladder Summary')
        axes[1, 2].set_xticks(range(len(sparsities)))
        axes[1, 2].set_xticklabels([f'{s:.3f}' for s in sparsities], rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Color bars based on target achievement
        for i, performance in enumerate(performances):
            if performance >= target_performance:
                axes[1, 2].patches[i].set_color('green')
                axes[1, 2].patches[i].set_alpha(0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'sparsity_ladder_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to {self.save_dir}/sparsity_ladder_analysis.png")


def main():
    """Main function for single layer sparsity ladder experiment."""
    parser = argparse.ArgumentParser(description='Single Layer Sparsity Ladder Experiment')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs (default: 30)')
    parser.add_argument('--save-dir', type=str, default='single_layer_results',
                       help='Directory to save results (default: single_layer_results)')
    
    args = parser.parse_args()
    
    print("🔬 SINGLE LAYER SPARSITY LADDER EXPERIMENT")
    print("=" * 60)
    print("Phase 1: Coarse-to-Fine Validation")
    print("- Fast iteration with single layer [256]")
    print("- Find Goldilocks zone where performance ≥ 50% of dense")
    print("- Build sparsity intuition before multi-layer complexity")
    print("=" * 60)
    
    experiment = SingleLayerSparsityExperiment(save_dir=args.save_dir)
    sparsity_results, goldilocks_candidates = experiment.run_sparsity_ladder(epochs=args.epochs)
    
    print(f"\n🎉 Single layer sparsity ladder completed!")
    print(f"📁 Results saved to {args.save_dir}/")
    
    if goldilocks_candidates:
        print(f"\n🎯 GOLDILOCKS ZONE FOUND!")
        print(f"Ready for Phase 2: Test growth mechanism on viable sparsity levels")
        print(f"Recommended next step: Test growth on sparsity {goldilocks_candidates[0][0]}")
    else:
        print(f"\n⚠️  NO GOLDILOCKS ZONE IN TESTED RANGE")
        print(f"Recommendation: Test higher sparsity levels (0.2, 0.5, 1.0)")


if __name__ == "__main__":
    main()
