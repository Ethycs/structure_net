#!/usr/bin/env python3
"""
CIFAR-10 Improved Experiment

This script demonstrates the full structure_net capabilities on CIFAR-10:
1. Uses proper multi-layer architecture for extrema detection
2. Integrates advanced growth mechanisms
3. Leverages the complete framework features
4. Includes comprehensive monitoring and analysis
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import time
import random
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.structure_net import create_multi_scale_network


class SimpleForwardPass:
    """Simple forward-pass strategy for network growth."""
    
    def __init__(self):
        self.dead_neuron_registry = {}
    
    def find_dead_neurons(self, layer, threshold=0.01):
        """Identify dead neurons based on activation threshold."""
        with torch.no_grad():
            # Get layer weights and check for near-zero activations
            weights = layer.weight.data
            # Consider neurons with very small weight magnitudes as "dead"
            weight_norms = torch.norm(weights, dim=1)
            dead_indices = torch.where(weight_norms < threshold)[0]
            return dead_indices.cpu().numpy().tolist()
    
    def add_connection(self, network, from_layer, from_neuron, to_layer, to_neuron, weight):
        """Add a new connection between layers."""
        if hasattr(network, 'connection_masks') and len(network.connection_masks) > from_layer:
            # For sparse networks with connection masks
            mask = network.connection_masks[from_layer]
            if to_neuron < mask.size(0) and from_neuron < mask.size(1):
                mask[to_neuron, from_neuron] = True
                network.layers[from_layer].weight.data[to_neuron, from_neuron] = weight
        elif hasattr(network, 'layers'):
            # For regular networks
            if from_layer < len(network.layers):
                layer = network.layers[from_layer]
                if hasattr(layer, 'weight') and to_neuron < layer.weight.size(0) and from_neuron < layer.weight.size(1):
                    layer.weight.data[to_neuron, from_neuron] = weight
    
    def grow_network_with_forward_pass(self, network):
        """Growth phase: forward dead neurons + randomly reinitialize."""
        
        # Step 1: Identify dead neurons in first hidden layer
        if hasattr(network, 'layers') and len(network.layers) > 0:
            first_layer = network.layers[0]
            dead_indices = self.find_dead_neurons(first_layer)
            
            if len(dead_indices) == 0:
                return network, 0  # No dead neurons found
            
            print(f"   🔍 Found {len(dead_indices)} dead neurons in first layer")
            
            # Step 2: Add connections from dead neurons to next layer
            connections_added = 0
            if len(network.layers) > 1:
                next_layer_size = network.layers[1].weight.size(0)
                
                for dead_idx in dead_indices[:min(len(dead_indices), 50)]:  # Limit to 50 for efficiency
                    # Connect to 5 random neurons in next layer
                    target_indices = random.sample(range(next_layer_size), k=min(5, next_layer_size))
                    
                    for target_idx in target_indices:
                        # Create NEW connections with fresh random weights
                        weight = torch.randn(1).item() * 0.1
                        self.add_connection(network, 0, dead_idx, 1, target_idx, weight)
                        connections_added += 1
            
            # Step 3: Randomly reinitialize incoming weights to dead neurons
            for dead_idx in dead_indices:
                if dead_idx < first_layer.weight.size(0):
                    # Give them a fresh start
                    first_layer.weight.data[dead_idx] = torch.randn_like(first_layer.weight.data[dead_idx]) * 0.1
                    if hasattr(first_layer, 'bias') and first_layer.bias is not None:
                        first_layer.bias.data[dead_idx] = 0.0  # Reset bias
            
            print(f"   🌱 Added {connections_added} new connections and reinitialized {len(dead_indices)} dead neurons")
            return network, connections_added
        
        return network, 0

class CIFAR10ImprovedExperiment:
    """Improved CIFAR-10 experiment using full structure_net capabilities."""
    
    def __init__(self, save_dir="data/cifar10_improved_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Experiment tracking
        self.training_log = []
        self.growth_events = []
        self.extrema_evolution = []
        self.snapshot_timeline = []
        self.performance_history = []
    
    def load_cifar10_data(self, batch_size=64):
        """Load CIFAR-10 with proper preprocessing."""
        print("📦 Loading CIFAR-10 dataset...")
        
        # Improved transforms for CIFAR-10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples.")
        return train_loader, test_loader
    
    def create_baseline_network(self, architecture, sparsity=0.02):
        """Create baseline sparse network for comparison."""
        from src.structure_net.core.minimal_network import MinimalNetwork
        
        network = MinimalNetwork(
            layer_sizes=architecture,
            sparsity=sparsity,
            activation='relu',
            device=self.device
        )
        
        return network
    
    def create_growth_network(self, architecture, sparsity=0.02):
        """Create growth-enabled network using full framework."""
        network = create_multi_scale_network(
            input_size=3072,  # 32*32*3
            hidden_sizes=architecture[1:-1],  # Extract hidden layers
            output_size=10,
            sparsity=sparsity,
            activation='relu',
            device=self.device,
            snapshot_dir=os.path.join(self.save_dir, "snapshots")
        )
        
        return network
    
    def train_baseline_network(self, network, train_loader, test_loader, epochs=50, lr=0.001):
        """Train baseline network without growth."""
        print("🚀 Training baseline network...")
        
        optimizer = optim.Adam(network.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        results = {
            'train_losses': [],
            'train_accs': [],
            'test_losses': [],
            'test_accs': [],
            'connections': []
        }
        
        best_test_acc = 0
        
        for epoch in range(epochs):
            # Training
            network.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)  # Flatten
                
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
                    data, target = data.to(self.device), target.to(self.device)
                    data = data.view(data.size(0), -1)
                    
                    output = network(data)
                    loss = criterion(output, target)
                    
                    test_loss += loss.item()
                    pred = output.argmax(dim=1)
                    test_correct += pred.eq(target).sum().item()
                    test_total += target.size(0)
            
            test_loss /= len(test_loader)
            test_acc = test_correct / test_total
            
            # Track results
            results['train_losses'].append(train_loss)
            results['train_accs'].append(train_acc)
            results['test_losses'].append(test_loss)
            results['test_accs'].append(test_acc)
            
            # Track connections (should be constant for baseline)
            if hasattr(network, 'get_connectivity_stats'):
                stats = network.get_connectivity_stats()
                results['connections'].append(stats['total_active_connections'])
            else:
                results['connections'].append(sum(p.numel() for p in network.parameters()))
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch:3d}/{epochs}: "
                      f"Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
        
        results['best_test_acc'] = best_test_acc
        results['final_test_acc'] = test_acc
        
        print(f"✅ Baseline training completed! Best accuracy: {best_test_acc:.4f}")
        return results
    
    def train_growth_network(self, network, train_loader, test_loader, epochs=50, lr=0.001):
        """Train growth-enabled network using framework capabilities with SimpleForwardPass."""
        print("🌱 Training growth-enabled network...")
        
        optimizer = optim.Adam(network.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Initialize SimpleForwardPass strategy
        forward_pass_strategy = SimpleForwardPass()
        
        results = {
            'train_losses': [],
            'train_accs': [],
            'test_losses': [],
            'test_accs': [],
            'connections': [],
            'growth_events': [],
            'extrema_counts': [],
            'forward_pass_events': []
        }
        
        best_test_acc = 0
        
        for epoch in range(epochs):
            # Training epoch using framework's method
            epoch_stats = network.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Evaluation using framework's method
            eval_stats = network.evaluate(test_loader, criterion)
            
            # Track results
            results['train_losses'].append(epoch_stats['loss'])
            results['train_accs'].append(epoch_stats['performance'])
            results['test_losses'].append(eval_stats['loss'])
            results['test_accs'].append(eval_stats['performance'])
            results['connections'].append(epoch_stats['total_connections'])
            
            # Track growth events from framework
            if epoch_stats.get('growth_events', 0) > 0:
                growth_info = {
                    'epoch': epoch,
                    'connections_added': epoch_stats.get('connections_added', 0),
                    'total_connections': epoch_stats['total_connections'],
                    'performance': eval_stats['performance']
                }
                results['growth_events'].append(growth_info)
                print(f"   🌱 FRAMEWORK GROWTH EVENT at epoch {epoch}! "
                      f"Added {growth_info['connections_added']} connections")
            
            # Apply SimpleForwardPass strategy every 10 epochs
            if epoch > 0 and epoch % 10 == 0:
                print(f"   🔄 Applying SimpleForwardPass strategy at epoch {epoch}")
                modified_network, connections_added = forward_pass_strategy.grow_network_with_forward_pass(network.network)
                
                if connections_added > 0:
                    forward_pass_info = {
                        'epoch': epoch,
                        'connections_added': connections_added,
                        'performance_before': eval_stats['performance']
                    }
                    results['forward_pass_events'].append(forward_pass_info)
                    print(f"   ✨ SIMPLE FORWARD PASS: Added {connections_added} connections")
            
            # Analyze extrema (every 5 epochs)
            if epoch % 5 == 0:
                extrema_analysis = self.analyze_extrema_patterns(network, epoch)
                results['extrema_counts'].append(extrema_analysis)
            
            if eval_stats['performance'] > best_test_acc:
                best_test_acc = eval_stats['performance']
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch:3d}/{epochs}: "
                      f"Train Acc={epoch_stats['performance']:.3f}, "
                      f"Test Acc={eval_stats['performance']:.3f}, "
                      f"Connections={epoch_stats['total_connections']}")
        
        results['best_test_acc'] = best_test_acc
        results['final_test_acc'] = results['test_accs'][-1]
        
        print(f"✅ Growth training completed! Best accuracy: {best_test_acc:.4f}")
        print(f"🌱 Framework growth events: {len(results['growth_events'])}")
        print(f"✨ SimpleForwardPass events: {len(results['forward_pass_events'])}")
        
        return results
    
    def analyze_extrema_patterns(self, network, epoch):
        """Analyze extrema patterns in the network."""
        # Run forward pass to populate activations
        test_input = torch.randn(64, 3072).to(self.device)
        _ = network(test_input)
        
        # Detect extrema
        extrema = network.network.detect_extrema(use_adaptive=True, epoch=epoch)
        
        # Count extrema
        total_high = sum(len(layer.get('high', [])) for layer in extrema.values())
        total_low = sum(len(layer.get('low', [])) for layer in extrema.values())
        
        extrema_analysis = {
            'epoch': epoch,
            'total_high': total_high,
            'total_low': total_low,
            'extrema_details': extrema
        }
        
        return extrema_analysis
    
    def run_comprehensive_experiment(self):
        """Run comprehensive CIFAR-10 experiment."""
        print("🔬 CIFAR-10 IMPROVED EXPERIMENT")
        print("=" * 60)
        
        # Load data
        train_loader, test_loader = self.load_cifar10_data(batch_size=128)
        
        # Define architectures to test
        architectures = [
            [3072, 512, 10],           # Simple 1-hidden
            [3072, 512, 256, 10],      # 2-hidden
            [3072, 768, 384, 192, 10], # 3-hidden
        ]
        
        sparsity_levels = [0.01, 0.005, 0.002]  # Different sparsity levels
        
        all_results = {}
        
        for arch_idx, architecture in enumerate(architectures):
            arch_name = f"arch_{len(architecture)-2}hidden"
            print(f"\n🏗️  Testing Architecture {arch_idx+1}/{len(architectures)}: {architecture}")
            
            for sparsity in sparsity_levels:
                print(f"\n📊 Testing sparsity: {sparsity}")
                
                # Test 1: Baseline (no growth)
                print(f"\n🔧 Baseline Network (sparsity={sparsity})")
                baseline_network = self.create_baseline_network(architecture, sparsity)
                baseline_results = self.train_baseline_network(
                    baseline_network, train_loader, test_loader, epochs=30
                )
                
                # Test 2: Growth-enabled
                print(f"\n🌱 Growth-enabled Network (sparsity={sparsity})")
                growth_network = self.create_growth_network(architecture, sparsity)
                growth_results = self.train_growth_network(
                    growth_network, train_loader, test_loader, epochs=30
                )
                
                # Store results
                result_key = f"{arch_name}_sparsity_{sparsity}"
                all_results[result_key] = {
                    'architecture': architecture,
                    'sparsity': sparsity,
                    'baseline': baseline_results,
                    'growth': growth_results,
                    'improvement': growth_results['best_test_acc'] - baseline_results['best_test_acc']
                }
                
                # Print comparison
                improvement = growth_results['best_test_acc'] - baseline_results['best_test_acc']
                print(f"\n📈 Results for {architecture} at {sparsity} sparsity:")
                print(f"   Baseline: {baseline_results['best_test_acc']:.4f}")
                print(f"   Growth:   {growth_results['best_test_acc']:.4f}")
                print(f"   Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
                print(f"   Growth events: {len(growth_results['growth_events'])}")
        
        # Save all results
        self.save_comprehensive_results(all_results)
        
        # Create comprehensive visualizations
        self.create_comprehensive_visualizations(all_results)
        
        # Final analysis
        self.analyze_comprehensive_results(all_results)
        
        return all_results
    
    def save_comprehensive_results(self, results):
        """Save comprehensive experiment results."""
        print("\n💾 Saving comprehensive results...")
        
        # Save detailed results
        with open(os.path.join(self.save_dir, "comprehensive_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'architectures_tested': len(set(r['architecture'] for r in results.values())),
            'sparsity_levels_tested': len(set(r['sparsity'] for r in results.values())),
            'best_results': {},
            'growth_effectiveness': {}
        }
        
        # Find best results
        best_baseline = max(results.values(), key=lambda x: x['baseline']['best_test_acc'])
        best_growth = max(results.values(), key=lambda x: x['growth']['best_test_acc'])
        best_improvement = max(results.values(), key=lambda x: x['improvement'])
        
        summary['best_results'] = {
            'best_baseline': {
                'accuracy': best_baseline['baseline']['best_test_acc'],
                'architecture': best_baseline['architecture'],
                'sparsity': best_baseline['sparsity']
            },
            'best_growth': {
                'accuracy': best_growth['growth']['best_test_acc'],
                'architecture': best_growth['architecture'],
                'sparsity': best_growth['sparsity']
            },
            'best_improvement': {
                'improvement': best_improvement['improvement'],
                'architecture': best_improvement['architecture'],
                'sparsity': best_improvement['sparsity']
            }
        }
        
        # Analyze growth effectiveness
        positive_improvements = [r for r in results.values() if r['improvement'] > 0]
        summary['growth_effectiveness'] = {
            'positive_improvements': len(positive_improvements),
            'total_experiments': len(results),
            'success_rate': len(positive_improvements) / len(results),
            'average_improvement': np.mean([r['improvement'] for r in results.values()])
        }
        
        with open(os.path.join(self.save_dir, "comprehensive_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved to {self.save_dir}/")
    
    def create_comprehensive_visualizations(self, results):
        """Create comprehensive visualizations."""
        print("📊 Creating comprehensive visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('CIFAR-10 Improved Experiment Results', fontsize=16)
        
        # Extract data for plotting
        architectures = []
        sparsities = []
        baseline_accs = []
        growth_accs = []
        improvements = []
        growth_events = []
        
        for key, result in results.items():
            arch_str = f"{len(result['architecture'])-2}H"  # e.g., "1H", "2H"
            architectures.append(arch_str)
            sparsities.append(result['sparsity'])
            baseline_accs.append(result['baseline']['best_test_acc'])
            growth_accs.append(result['growth']['best_test_acc'])
            improvements.append(result['improvement'])
            growth_events.append(len(result['growth']['growth_events']))
        
        # 1. Accuracy comparison
        x = np.arange(len(results))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, baseline_accs, width, label='Baseline', alpha=0.7)
        axes[0, 0].bar(x + width/2, growth_accs, width, label='Growth', alpha=0.7)
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f"{a}\n{s}" for a, s in zip(architectures, sparsities)], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Improvement analysis
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[0, 1].bar(x, improvements, color=colors, alpha=0.7)
        axes[0, 1].set_title('Growth Improvement')
        axes[0, 1].set_ylabel('Accuracy Improvement')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f"{a}\n{s}" for a, s in zip(architectures, sparsities)], rotation=45)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Growth events
        axes[0, 2].bar(x, growth_events, alpha=0.7, color='orange')
        axes[0, 2].set_title('Growth Events Count')
        axes[0, 2].set_ylabel('Number of Growth Events')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels([f"{a}\n{s}" for a, s in zip(architectures, sparsities)], rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4-6. Training curves for selected experiments
        selected_keys = list(results.keys())[:3]  # Show first 3 experiments
        colors = ['blue', 'green', 'red']
        
        for i, key in enumerate(selected_keys):
            if i < 3:
                result = results[key]
                epochs = range(len(result['baseline']['test_accs']))
                
                # Training curves
                axes[1, i].plot(epochs, result['baseline']['test_accs'], 
                               label='Baseline', color='red', alpha=0.8)
                axes[1, i].plot(epochs, result['growth']['test_accs'], 
                               label='Growth', color='blue', alpha=0.8)
                
                # Mark growth events
                for event in result['growth']['growth_events']:
                    axes[1, i].axvline(x=event['epoch'], color='green', linestyle='--', alpha=0.7)
                
                axes[1, i].set_title(f"{architectures[i]} Hidden, Sparsity {sparsities[i]}")
                axes[1, i].set_xlabel('Epoch')
                axes[1, i].set_ylabel('Test Accuracy')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
        
        # 7. Architecture comparison
        arch_groups = {}
        for i, arch in enumerate(architectures):
            if arch not in arch_groups:
                arch_groups[arch] = {'baseline': [], 'growth': [], 'improvement': []}
            arch_groups[arch]['baseline'].append(baseline_accs[i])
            arch_groups[arch]['growth'].append(growth_accs[i])
            arch_groups[arch]['improvement'].append(improvements[i])
        
        arch_names = list(arch_groups.keys())
        avg_baseline = [np.mean(arch_groups[arch]['baseline']) for arch in arch_names]
        avg_growth = [np.mean(arch_groups[arch]['growth']) for arch in arch_names]
        
        x_arch = np.arange(len(arch_names))
        axes[2, 0].bar(x_arch - width/2, avg_baseline, width, label='Avg Baseline', alpha=0.7)
        axes[2, 0].bar(x_arch + width/2, avg_growth, width, label='Avg Growth', alpha=0.7)
        axes[2, 0].set_title('Average Performance by Architecture')
        axes[2, 0].set_ylabel('Average Test Accuracy')
        axes[2, 0].set_xticks(x_arch)
        axes[2, 0].set_xticklabels(arch_names)
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Sparsity analysis
        sparsity_groups = {}
        for i, sparsity in enumerate(sparsities):
            if sparsity not in sparsity_groups:
                sparsity_groups[sparsity] = {'baseline': [], 'growth': [], 'improvement': []}
            sparsity_groups[sparsity]['baseline'].append(baseline_accs[i])
            sparsity_groups[sparsity]['growth'].append(growth_accs[i])
            sparsity_groups[sparsity]['improvement'].append(improvements[i])
        
        sparsity_names = list(sparsity_groups.keys())
        avg_improvement = [np.mean(sparsity_groups[s]['improvement']) for s in sparsity_names]
        
        axes[2, 1].bar(range(len(sparsity_names)), avg_improvement, alpha=0.7, color='purple')
        axes[2, 1].set_title('Average Improvement by Sparsity')
        axes[2, 1].set_ylabel('Average Improvement')
        axes[2, 1].set_xticks(range(len(sparsity_names)))
        axes[2, 1].set_xticklabels([f'{s:.3f}' for s in sparsity_names])
        axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Summary statistics
        axes[2, 2].axis('off')
        summary_text = f"""
Experiment Summary:
• Total Experiments: {len(results)}
• Architectures: {len(set(architectures))}
• Sparsity Levels: {len(set(sparsities))}

Best Results:
• Best Baseline: {max(baseline_accs):.3f}
• Best Growth: {max(growth_accs):.3f}
• Best Improvement: {max(improvements):.3f}

Growth Effectiveness:
• Positive Improvements: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}
• Success Rate: {sum(1 for imp in improvements if imp > 0)/len(improvements)*100:.1f}%
• Avg Improvement: {np.mean(improvements):.3f}
        """
        axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'comprehensive_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualizations saved to {self.save_dir}/comprehensive_results.png")
    
    def analyze_comprehensive_results(self, results):
        """Analyze comprehensive experiment results."""
        print("\n📊 COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        # Overall statistics
        baseline_accs = [r['baseline']['best_test_acc'] for r in results.values()]
        growth_accs = [r['growth']['best_test_acc'] for r in results.values()]
        improvements = [r['improvement'] for r in results.values()]
        
        print(f"📈 Overall Performance:")
        print(f"   Baseline range: {min(baseline_accs):.3f} - {max(baseline_accs):.3f}")
        print(f"   Growth range: {min(growth_accs):.3f} - {max(growth_accs):.3f}")
        print(f"   Average improvement: {np.mean(improvements):.3f}")
        
        # Success analysis
        positive_improvements = [imp for imp in improvements if imp > 0]
        success_rate = len(positive_improvements) / len(improvements)
        
        print(f"\n🎯 Growth Effectiveness:")
        print(f"   Successful experiments: {len(positive_improvements)}/{len(improvements)}")
        print(f"   Success rate: {success_rate*100:.1f}%")
        print(f"   Average positive improvement: {np.mean(positive_improvements):.3f}")
        
        # Best configurations
        best_result = max(results.values(), key=lambda x: x['growth']['best_test_acc'])
        best_improvement = max(results.values(), key=lambda x: x['improvement'])
        
        print(f"\n🏆 Best Configurations:")
        print(f"   Best absolute performance:")
        print(f"      Architecture: {best_result['architecture']}")
        print(f"      Sparsity: {best_result['sparsity']}")
        print(f"      Accuracy: {best_result['growth']['best_test_acc']:.3f}")
        
        print(f"   Best improvement:")
        print(f"      Architecture: {best_improvement['architecture']}")
        print(f"      Sparsity: {best_improvement['sparsity']}")
        print(f"      Improvement: {best_improvement['improvement']:.3f}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if success_rate > 0.7:
            print(f"   ✅ Growth mechanism is effective for CIFAR-10")
            print(f"   🚀 Consider scaling to larger architectures")
        elif success_rate > 0.4:
            print(f"   📊 Growth shows promise but needs optimization")
            print(f"   🔧 Focus on parameter tuning for better results")
        else:
            print(f"   ⚠️  Growth mechanism needs significant improvement")
            print(f"   🔧 Debug extrema detection and growth triggers")


def main():
    """Main function to run the improved CIFAR-10 experiment."""
    print("🔬 CIFAR-10 IMPROVED EXPERIMENT")
    print("=" * 60)
    print("This experiment demonstrates the full structure_net capabilities:")
    print("1. Proper multi-layer architectures for extrema detection")
    print("2. Advanced growth mechanisms integration")
    print("3. Comprehensive monitoring and analysis")
    print("4. Comparison between baseline and growth-enabled networks")
    print("=" * 60)
    
    # Run the comprehensive experiment
    experiment = CIFAR10ImprovedExperiment()
    results = experiment.run_comprehensive_experiment()
    
    print(f"\n🎉 CIFAR-10 improved experiment completed!")
    print(f"📁 Results saved to {experiment.save_dir}/")
    
    return results


if __name__ == "__main__":
    main()
