#!/usr/bin/env python3
"""
Architecture Validation Script

Before running sparsity experiments, this script validates:
1. Can this architecture learn anything when dense?
2. What's the random baseline for the task?
3. Are gradients flowing through all layers?

This ensures we have a solid foundation before exploring sparsity.
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

class ArchitectureValidator:
    """Validates the fundamental capabilities of the architecture."""
    
    def __init__(self, save_dir="validation_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Check device
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Results storage
        self.results = {}
    
    def create_challenging_dataset(self, n_samples=3000, input_dim=784, n_classes=10):
        """Create the same challenging synthetic dataset used in experiments."""
        print("📦 Creating challenging synthetic dataset...")
        
        patterns = []
        labels = []
        samples_per_class = n_samples // n_classes
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        for class_idx in range(n_classes):
            for i in range(samples_per_class):
                # Create complex patterns that should trigger growth
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
    
    def calculate_random_baseline(self, y):
        """Calculate the random baseline performance."""
        n_classes = len(torch.unique(y))
        random_accuracy = 1.0 / n_classes
        
        print(f"🎲 Random baseline accuracy: {random_accuracy:.4f} ({random_accuracy*100:.1f}%)")
        
        # Also calculate class distribution baseline (majority class)
        class_counts = torch.bincount(y)
        majority_class_accuracy = class_counts.max().float() / len(y)
        
        print(f"📊 Majority class baseline: {majority_class_accuracy:.4f} ({majority_class_accuracy*100:.1f}%)")
        
        self.results['random_baseline'] = {
            'random_accuracy': random_accuracy,
            'majority_class_accuracy': majority_class_accuracy.item(),
            'n_classes': n_classes,
            'class_distribution': class_counts.tolist()
        }
        
        return random_accuracy, majority_class_accuracy.item()
    
    def create_dense_network(self, input_size, hidden_sizes, output_size, activation='tanh'):
        """Create a fully dense version of the network architecture."""
        layers = []
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers).to(self.device)
    
    def analyze_gradient_flow(self, network, data_loader, criterion):
        """Analyze gradient flow through all layers."""
        print("🔍 Analyzing gradient flow...")
        
        network.train()
        
        # Run a few batches to get gradient statistics
        gradient_stats = {}
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 5:  # Only analyze first 5 batches
                break
            
            network.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Collect gradient statistics
            for name, param in network.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    grad_max = param.grad.max().item()
                    grad_min = param.grad.min().item()
                    
                    if name not in gradient_stats:
                        gradient_stats[name] = {
                            'norms': [],
                            'means': [],
                            'stds': [],
                            'maxs': [],
                            'mins': []
                        }
                    
                    gradient_stats[name]['norms'].append(grad_norm)
                    gradient_stats[name]['means'].append(grad_mean)
                    gradient_stats[name]['stds'].append(grad_std)
                    gradient_stats[name]['maxs'].append(grad_max)
                    gradient_stats[name]['mins'].append(grad_min)
        
        # Calculate average statistics
        avg_gradient_stats = {}
        for name, stats in gradient_stats.items():
            avg_gradient_stats[name] = {
                'avg_norm': np.mean(stats['norms']),
                'avg_mean': np.mean(stats['means']),
                'avg_std': np.mean(stats['stds']),
                'avg_max': np.mean(stats['maxs']),
                'avg_min': np.mean(stats['mins']),
                'norm_std': np.std(stats['norms'])
            }
        
        # Print gradient flow analysis
        print("📊 Gradient Flow Analysis:")
        for name, stats in avg_gradient_stats.items():
            print(f"   {name}:")
            print(f"      Norm: {stats['avg_norm']:.6f} ± {stats['norm_std']:.6f}")
            print(f"      Range: [{stats['avg_min']:.6f}, {stats['avg_max']:.6f}]")
            
            # Check for potential issues
            if stats['avg_norm'] < 1e-6:
                print(f"      ⚠️  Very small gradients - potential vanishing gradient problem")
            elif stats['avg_norm'] > 10:
                print(f"      ⚠️  Large gradients - potential exploding gradient problem")
            else:
                print(f"      ✅ Healthy gradient flow")
        
        self.results['gradient_flow'] = avg_gradient_stats
        return avg_gradient_stats
    
    def train_and_evaluate(self, network, train_loader, test_loader, epochs=50, lr=0.002, name="network"):
        """Train and evaluate a network."""
        print(f"\n🚀 Training {name}...")
        
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
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch:3d}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
                      f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.3f}")
        
        total_time = time.time() - start_time
        
        print(f"✅ {name} training completed!")
        print(f"   Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.1f}%)")
        print(f"   Final test accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
        print(f"   Training time: {total_time:.2f} seconds")
        
        # Analyze gradient flow at the end
        gradient_stats = self.analyze_gradient_flow(network, train_loader, criterion)
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'best_test_acc': best_test_acc,
            'final_test_acc': test_acc,
            'training_time': total_time,
            'gradient_stats': gradient_stats
        }
    
    def run_validation(self):
        """Run complete architecture validation."""
        print("🔬 ARCHITECTURE VALIDATION")
        print("=" * 60)
        
        # Create dataset
        X, y = self.create_challenging_dataset()
        
        # Calculate baselines
        random_baseline, majority_baseline = self.calculate_random_baseline(y)
        
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
        
        # Test different architectures
        architectures = [
            ([256, 128], "Small Dense"),
            ([512, 256, 128], "Medium Dense"),
            ([768, 384, 192], "Large Dense"),
        ]
        
        print("\n" + "=" * 60)
        print("🧪 TESTING DENSE ARCHITECTURES")
        print("=" * 60)
        
        for hidden_sizes, name in architectures:
            print(f"\n📐 Testing {name}: {hidden_sizes}")
            
            # Create dense network
            dense_network = self.create_dense_network(784, hidden_sizes, 10, activation='tanh')
            
            # Count parameters
            total_params = sum(p.numel() for p in dense_network.parameters())
            trainable_params = sum(p.numel() for p in dense_network.parameters() if p.requires_grad)
            
            print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            # Train and evaluate
            results = self.train_and_evaluate(
                dense_network, train_loader, test_loader, 
                epochs=50, lr=0.002, name=name
            )
            
            self.results[f'dense_{name.lower().replace(" ", "_")}'] = {
                'architecture': hidden_sizes,
                'total_params': total_params,
                'trainable_params': trainable_params,
                **results
            }
        
        # Test sparse network for comparison
        print("\n" + "=" * 60)
        print("🕸️  TESTING SPARSE ARCHITECTURE")
        print("=" * 60)
        
        print(f"\n📐 Testing Sparse Network: [512, 256, 128] with 0.01% connectivity")
        
        # Create sparse network (our current architecture)
        sparse_network = create_multi_scale_network(
            784, [512, 256, 128], 10,
            sparsity=0.0001,  # 0.01% initial connectivity
            device=self.device,
            snapshot_dir=os.path.join(self.save_dir, "sparse_snapshots")
        )
        
        # Get connectivity stats
        connectivity_stats = sparse_network.network.get_connectivity_stats()
        print(f"   Initial connections: {connectivity_stats['total_active_connections']}")
        print(f"   Initial sparsity: {connectivity_stats['sparsity']:.6f}")
        print(f"   Connectivity ratio: {connectivity_stats['connectivity_ratio']:.6f}")
        
        # Train sparse network using its own training method
        print(f"🚀 Training Sparse Network...")
        
        optimizer = optim.Adam(sparse_network.parameters(), lr=0.002)
        criterion = nn.CrossEntropyLoss()
        
        sparse_results = {
            'train_losses': [],
            'train_accs': [],
            'test_losses': [],
            'test_accs': [],
            'connections': [],
            'growth_events': []
        }
        
        best_test_acc = 0
        start_time = time.time()
        
        for epoch in range(50):
            # Training epoch using sparse network's method
            epoch_stats = sparse_network.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Evaluation using sparse network's method
            eval_stats = sparse_network.evaluate(test_loader, criterion)
            
            sparse_results['train_losses'].append(epoch_stats['loss'])
            sparse_results['train_accs'].append(epoch_stats['performance'])
            sparse_results['test_losses'].append(eval_stats['loss'])
            sparse_results['test_accs'].append(eval_stats['performance'])
            sparse_results['connections'].append(epoch_stats['total_connections'])
            
            if epoch_stats.get('growth_events', 0) > 0:
                sparse_results['growth_events'].append({
                    'epoch': epoch,
                    'connections_added': epoch_stats.get('connections_added', 0),
                    'total_connections': epoch_stats['total_connections']
                })
            
            if eval_stats['performance'] > best_test_acc:
                best_test_acc = eval_stats['performance']
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == 49:
                print(f"   Epoch {epoch:3d}/50: "
                      f"Train Loss={epoch_stats['loss']:.4f}, Train Acc={epoch_stats['performance']:.3f}, "
                      f"Test Loss={eval_stats['loss']:.4f}, Test Acc={eval_stats['performance']:.3f}, "
                      f"Connections={epoch_stats['total_connections']}")
        
        total_time = time.time() - start_time
        final_connectivity = sparse_network.network.get_connectivity_stats()
        
        print(f"✅ Sparse Network training completed!")
        print(f"   Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.1f}%)")
        print(f"   Final test accuracy: {sparse_results['test_accs'][-1]:.4f}")
        print(f"   Final connections: {final_connectivity['total_active_connections']}")
        print(f"   Connection growth: {final_connectivity['total_active_connections'] - connectivity_stats['total_active_connections']}")
        print(f"   Growth events: {len(sparse_results['growth_events'])}")
        print(f"   Training time: {total_time:.2f} seconds")
        
        self.results['sparse_network'] = {
            'initial_connectivity': connectivity_stats,
            'final_connectivity': final_connectivity,
            'best_test_acc': best_test_acc,
            'final_test_acc': sparse_results['test_accs'][-1],
            'training_time': total_time,
            'growth_events': len(sparse_results['growth_events']),
            'connection_growth': final_connectivity['total_active_connections'] - connectivity_stats['total_active_connections'],
            **sparse_results
        }
        
        # Final analysis
        self.analyze_results()
        self.save_results()
        self.create_visualizations()
    
    def analyze_results(self):
        """Analyze and compare all results."""
        print("\n" + "=" * 60)
        print("📊 VALIDATION ANALYSIS")
        print("=" * 60)
        
        random_acc = self.results['random_baseline']['random_accuracy']
        majority_acc = self.results['random_baseline']['majority_class_accuracy']
        
        print(f"🎲 Baselines:")
        print(f"   Random: {random_acc:.4f} ({random_acc*100:.1f}%)")
        print(f"   Majority class: {majority_acc:.4f} ({majority_acc*100:.1f}%)")
        
        print(f"\n🧪 Dense Network Results:")
        dense_results = []
        for key, result in self.results.items():
            if key.startswith('dense_'):
                name = key.replace('dense_', '').replace('_', ' ').title()
                best_acc = result['best_test_acc']
                params = result['total_params']
                
                print(f"   {name}: {best_acc:.4f} ({best_acc*100:.1f}%) - {params:,} params")
                dense_results.append((name, best_acc, params))
                
                # Check if significantly better than random
                improvement = (best_acc - random_acc) / random_acc * 100
                if best_acc > random_acc * 1.5:  # 50% better than random
                    print(f"      ✅ {improvement:.1f}% improvement over random - LEARNING!")
                else:
                    print(f"      ⚠️  Only {improvement:.1f}% improvement over random - limited learning")
        
        print(f"\n🕸️  Sparse Network Results:")
        if 'sparse_network' in self.results:
            sparse_result = self.results['sparse_network']
            sparse_acc = sparse_result['best_test_acc']
            initial_conn = sparse_result['initial_connectivity']['total_active_connections']
            final_conn = sparse_result['final_connectivity']['total_active_connections']
            growth_events = sparse_result['growth_events']
            
            print(f"   Best accuracy: {sparse_acc:.4f} ({sparse_acc*100:.1f}%)")
            print(f"   Connections: {initial_conn} → {final_conn} (+{final_conn-initial_conn})")
            print(f"   Growth events: {growth_events}")
            
            improvement = (sparse_acc - random_acc) / random_acc * 100
            if sparse_acc > random_acc * 1.5:
                print(f"      ✅ {improvement:.1f}% improvement over random - LEARNING!")
            else:
                print(f"      ⚠️  Only {improvement:.1f}% improvement over random - limited learning")
            
            # Compare to best dense network
            best_dense_acc = max(result['best_test_acc'] for key, result in self.results.items() if key.startswith('dense_'))
            dense_vs_sparse = (sparse_acc / best_dense_acc) * 100
            print(f"   Sparse vs best dense: {dense_vs_sparse:.1f}% of dense performance")
        
        print(f"\n🔍 Key Findings:")
        
        # Check if any network learned significantly
        any_learning = False
        for key, result in self.results.items():
            if key.startswith('dense_') or key == 'sparse_network':
                best_acc = result['best_test_acc']
                if best_acc > random_acc * 2:  # 100% better than random
                    any_learning = True
                    break
        
        if any_learning:
            print("   ✅ Architecture CAN learn - networks significantly outperform random baseline")
        else:
            print("   ❌ Architecture struggles to learn - performance close to random baseline")
            print("   🔧 Consider: different activation functions, learning rates, or architectures")
        
        # Check gradient flow issues
        gradient_issues = []
        for key, result in self.results.items():
            if key.startswith('dense_') and 'gradient_stats' in result:
                for param_name, stats in result['gradient_stats'].items():
                    if stats['avg_norm'] < 1e-6:
                        gradient_issues.append(f"Vanishing gradients in {param_name}")
                    elif stats['avg_norm'] > 10:
                        gradient_issues.append(f"Exploding gradients in {param_name}")
        
        if gradient_issues:
            print("   ⚠️  Gradient flow issues detected:")
            for issue in gradient_issues[:3]:  # Show first 3 issues
                print(f"      - {issue}")
        else:
            print("   ✅ Gradient flow appears healthy across all layers")
        
        # Architecture recommendations
        print(f"\n💡 Recommendations:")
        if any_learning:
            print("   ✅ Architecture is fundamentally sound - proceed with sparsity experiments")
            print("   📊 Focus on optimizing sparsity levels and growth parameters")
        else:
            print("   🔧 Fix fundamental learning issues before exploring sparsity:")
            print("   - Try different activation functions (ReLU, LeakyReLU)")
            print("   - Adjust learning rates (try 0.001, 0.01)")
            print("   - Consider batch normalization or layer normalization")
            print("   - Verify dataset complexity is appropriate")
    
    def save_results(self):
        """Save all validation results."""
        print(f"\n💾 Saving validation results...")
        
        # Save detailed results
        with open(os.path.join(self.save_dir, "validation_results.json"), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'random_baseline': self.results['random_baseline']['random_accuracy'],
            'majority_baseline': self.results['random_baseline']['majority_class_accuracy'],
            'dense_networks': {},
            'sparse_network': {}
        }
        
        for key, result in self.results.items():
            if key.startswith('dense_'):
                name = key.replace('dense_', '')
                summary['dense_networks'][name] = {
                    'best_accuracy': result['best_test_acc'],
                    'parameters': result['total_params'],
                    'improvement_over_random': (result['best_test_acc'] - summary['random_baseline']) / summary['random_baseline'] * 100
                }
        
        if 'sparse_network' in self.results:
            sparse = self.results['sparse_network']
            summary['sparse_network'] = {
                'best_accuracy': sparse['best_test_acc'],
                'initial_connections': sparse['initial_connectivity']['total_active_connections'],
                'final_connections': sparse['final_connectivity']['total_active_connections'],
                'growth_events': sparse['growth_events'],
                'improvement_over_random': (sparse['best_test_acc'] - summary['random_baseline']) / summary['random_baseline'] * 100
            }
        
        with open(os.path.join(self.save_dir, "validation_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved to {self.save_dir}/")
    
    def create_visualizations(self):
        """Create validation visualizations."""
        print("📊 Creating validation visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Architecture Validation Results', fontsize=16)
        
        # 1. Accuracy comparison
        networks = []
        accuracies = []
        param_counts = []
        
        # Add baselines
        random_acc = self.results['random_baseline']['random_accuracy']
        majority_acc = self.results['random_baseline']['majority_class_accuracy']
        
        # Add dense networks
        for key, result in self.results.items():
            if key.startswith('dense_'):
                name = key.replace('dense_', '').replace('_', ' ').title()
                networks.append(name)
                accuracies.append(result['best_test_acc'])
                param_counts.append(result['total_params'])
        
        # Add sparse network
        if 'sparse_network' in self.results:
            networks.append('Sparse Network')
            accuracies.append(self.results['sparse_network']['best_test_acc'])
            param_counts.append(self.results['sparse_network']['final_connectivity']['total_active_connections'])
        
        bars = axes[0, 0].bar(networks, accuracies, alpha=0.7)
        axes[0, 0].axhline(y=random_acc, color='red', linestyle='--', label=f'Random ({random_acc:.3f})')
        axes[0, 0].axhline(y=majority_acc, color='orange', linestyle='--', label=f'Majority ({majority_acc:.3f})')
        axes[0, 0].set_title('Best Test Accuracy by Architecture')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Parameter efficiency (accuracy per parameter)
        if param_counts:
            efficiency = [acc / (params / 1000) for acc, params in zip(accuracies, param_counts)]  # per 1K params
            axes[0, 1].bar(networks, efficiency, alpha=0.7, color='green')
            axes[0, 1].set_title('Parameter Efficiency (Accuracy per 1K params)')
            axes[0, 1].set_ylabel('Efficiency')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training curves for dense networks
        colors = ['blue', 'green', 'red', 'purple']
        for i, (key, result) in enumerate(self.results.items()):
            if key.startswith('dense_') and i < len(colors):
                name = key.replace('dense_', '').replace('_', ' ').title()
                epochs = range(len(result['test_accs']))
                axes[0, 2].plot(epochs, result['test_accs'], color=colors[i], label=name, alpha=0.8)
        
        if 'sparse_network' in self.results:
            sparse_result = self.results['sparse_network']
            epochs = range(len(sparse_result['test_accs']))
            axes[0, 2].plot(epochs, sparse_result['test_accs'], color='black', label='Sparse Network', alpha=0.8, linestyle='--')
        
        axes[0, 2].axhline(y=random_acc, color='red', linestyle=':', alpha=0.5, label='Random')
        axes[0, 2].set_title('Test Accuracy Over Training')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Test Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Loss curves
        for i, (key, result) in enumerate(self.results.items()):
            if key.startswith('dense_') and i < len(colors):
                name = key.replace('dense_', '').replace('_', ' ').title()
                epochs = range(len(result['test_losses']))
                axes[1, 0].plot(epochs, result['test_losses'], color=colors[i], label=name, alpha=0.8)
        
        if 'sparse_network' in self.results:
            sparse_result = self.results['sparse_network']
            epochs = range(len(sparse_result['test_losses']))
            axes[1, 0].plot(epochs, sparse_result['test_losses'], color='black', label='Sparse Network', alpha=0.8, linestyle='--')
        
        axes[1, 0].set_title('Test Loss Over Training')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Test Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Gradient norms (if available)
        gradient_data = []
        gradient_names = []
        for key, result in self.results.items():
            if key.startswith('dense_') and 'gradient_stats' in result:
                name = key.replace('dense_', '').replace('_', ' ').title()
                avg_norms = [stats['avg_norm'] for stats in result['gradient_stats'].values()]
                if avg_norms:
                    gradient_data.append(np.mean(avg_norms))
                    gradient_names.append(name)
        
        if gradient_data:
            axes[1, 1].bar(gradient_names, gradient_data, alpha=0.7, color='orange')
            axes[1, 1].set_title('Average Gradient Norms')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')  # Log scale for gradient norms
        
        # 6. Connection growth for sparse network
        if 'sparse_network' in self.results:
            sparse_result = self.results['sparse_network']
            epochs = range(len(sparse_result['connections']))
            axes[1, 2].plot(epochs, sparse_result['connections'], color='green', linewidth=2)
            axes[1, 2].set_title('Sparse Network Connection Growth')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Active Connections')
            axes[1, 2].grid(True, alpha=0.3)
            
            # Mark growth events
            for event in sparse_result['growth_events']:
                axes[1, 2].axvline(x=event['epoch'], color='red', linestyle='--', alpha=0.7)
                axes[1, 2].annotate(f"+{event['connections_added']}", 
                                  xy=(event['epoch'], event['total_connections']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Sparse Network Data', ha='center', va='center', 
                           transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].set_title('Sparse Network Connection Growth')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'validation_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Validation visualizations saved to {self.save_dir}/validation_results.png")


def main():
    """Main function to run architecture validation."""
    print("🔬 ARCHITECTURE VALIDATION SCRIPT")
    print("=" * 60)
    print("This script validates the fundamental capabilities before sparsity experiments:")
    print("1. Can this architecture learn anything when dense?")
    print("2. What's the random baseline for the task?")
    print("3. Are gradients flowing through all layers?")
    print("=" * 60)
    
    validator = ArchitectureValidator()
    validator.run_validation()
    
    print("\n🎉 Architecture validation completed!")
    print("📁 Check validation_results/ for detailed analysis and visualizations")


if __name__ == "__main__":
    main()
