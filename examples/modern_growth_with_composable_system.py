#!/usr/bin/env python3
"""
Modern Indefinite Growth with Composable System

This example recreates the modern indefinite growth experiment using the new
composable evolution architecture. It demonstrates how to achieve the same
sophisticated growth behavior with the new modular system.

Key features recreated:
- Sophisticated extrema detection
- Data-driven connection placement
- Embedded patches (no separate modules)
- Growth decision logic
- Performance tracking
- Multi-GPU support
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import multiprocessing as mp

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the new composable system
from structure_net.evolution.components import (
    ComposableEvolutionSystem,
    NetworkContext,
    StandardExtremaAnalyzer,
    NetworkStatsAnalyzer,
    SimpleInformationFlowAnalyzer,
    ExtremaGrowthStrategy,
    InformationFlowGrowthStrategy,
    ResidualBlockGrowthStrategy,
    HybridGrowthStrategy,
    StandardNetworkTrainer,
    create_standard_evolution_system,
    create_extrema_focused_system,
    create_hybrid_system
)

# Import existing infrastructure
from structure_net.core.network_factory import create_standard_network
from structure_net.core.network_analysis import get_network_stats
from structure_net.core.io_operations import save_model_seed, load_model_seed


class ModernComposableGrowth:
    """
    Modern indefinite growth using the new composable evolution system.
    
    This recreates the sophisticated behavior of the original ModernIndefiniteGrowth
    but uses the new composable architecture for better modularity and extensibility.
    """
    
    def __init__(self, seed_architecture, scaffold_sparsity=0.02, device='cuda', 
                 allow_patches=True, growth_strategy='standard'):
        self.device = device
        self.scaffold_sparsity = scaffold_sparsity
        self.seed_architecture = seed_architecture
        self.allow_patches = allow_patches
        self.growth_strategy = growth_strategy
        
        # Create initial network
        self.network = create_standard_network(
            architecture=seed_architecture,
            sparsity=scaffold_sparsity,
            device=device
        )
        
        # Create composable evolution system based on strategy
        self.evolution_system = self._create_evolution_system()
        
        # Growth tracking
        self.growth_history = []
        self.current_accuracy = 0.0
        self.iteration = 0
        
        # Growth thresholds (configurable)
        self.layer_addition_threshold = 0.6  # 60% extrema ratio triggers new layer
        self.patch_threshold = 5             # 5+ extrema triggers patches
        
        print(f"🚀 Modern Composable Growth initialized")
        print(f"   Seed architecture: {seed_architecture}")
        print(f"   Scaffold sparsity: {scaffold_sparsity:.1%}")
        print(f"   Device: {device}")
        print(f"   Growth strategy: {growth_strategy}")
        print(f"   Allow patches: {allow_patches}")
    
    def _create_evolution_system(self):
        """Create the appropriate evolution system based on strategy."""
        if self.growth_strategy == 'standard':
            return create_standard_evolution_system()
        elif self.growth_strategy == 'extrema_focused':
            return create_extrema_focused_system()
        elif self.growth_strategy == 'hybrid':
            return create_hybrid_system()
        elif self.growth_strategy == 'custom':
            return self._create_custom_system()
        else:
            # Default to standard
            return create_standard_evolution_system()
    
    def _create_custom_system(self):
        """Create a custom evolution system that mimics the original behavior."""
        system = ComposableEvolutionSystem()
        
        # Add sophisticated extrema analyzer (like the original)
        extrema_analyzer = StandardExtremaAnalyzer()
        extrema_analyzer.configure({
            'dead_threshold': 0.01,        # Match original dead_threshold
            'saturated_multiplier': 2.5,   # Match original sophisticated thresholds
            'max_batches': 2               # Match original optimization (reduced from 5 to 2)
        })
        system.add_component(extrema_analyzer)
        
        # Add network stats analyzer
        system.add_component(NetworkStatsAnalyzer())
        
        # Add information flow analyzer for bottleneck detection
        info_analyzer = SimpleInformationFlowAnalyzer()
        info_analyzer.configure({
            'min_bottleneck_severity': 0.05
        })
        system.add_component(info_analyzer)
        
        # Add extrema growth strategy (matches original layer addition logic)
        extrema_strategy = ExtremaGrowthStrategy()
        extrema_strategy.configure({
            'extrema_threshold': 0.6,      # Match layer_addition_threshold
            'dead_neuron_threshold': 5,    # Match patch_threshold
            'saturated_neuron_threshold': 10,  # Match original saturation logic
            'patch_size': 5                # Match original patch size
        })
        system.add_component(extrema_strategy)
        
        # Add information flow strategy for bottleneck handling
        info_strategy = InformationFlowGrowthStrategy()
        info_strategy.configure({
            'bottleneck_threshold': 0.1,
            'efficiency_threshold': 0.7
        })
        system.add_component(info_strategy)
        
        # Add patches only if enabled
        if self.allow_patches:
            # The extrema strategy already handles patches, but we can add more sophisticated ones
            pass
        
        # Add trainer with dual learning rate concept
        trainer = StandardNetworkTrainer()
        trainer.configure({
            'learning_rate': 0.001,
            'scaffold_lr': 0.0001,  # Slower for stability (like original dual rates)
            'patch_lr': 0.001       # Faster for new learning
        })
        system.add_component(trainer)
        
        return system
    
    @property
    def current_architecture(self):
        """Get current architecture from network stats."""
        stats = get_network_stats(self.network)
        return stats['architecture']
    
    def evaluate_network(self, test_loader):
        """Evaluate network performance."""
        self.network.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)
                output = self.network(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        self.current_accuracy = accuracy
        print(f"📊 Current accuracy: {accuracy:.2%}")
        return accuracy
    
    def growth_step(self, train_loader, test_loader):
        """Perform one growth step using the composable system."""
        self.iteration += 1
        
        print(f"\n🌱 GROWTH ITERATION {self.iteration}")
        print("=" * 50)
        
        # Create network context for the composable system
        context = NetworkContext(
            network=self.network,
            data_loader=train_loader,
            device=self.device,
            metadata={
                'val_loader': test_loader,
                'iteration': self.iteration,
                'allow_patches': self.allow_patches
            }
        )
        
        # Run evolution using the composable system
        print("🧬 Running composable evolution...")
        evolved_context = self.evolution_system.evolve_network(context, num_iterations=1)
        
        # Update our network with the evolved one
        self.network = evolved_context.network
        
        # Evaluate performance
        accuracy = self.evaluate_network(test_loader)
        
        # Check if growth occurred by comparing architectures or performance
        growth_occurred = len(evolved_context.performance_history) > 1
        if not growth_occurred:
            # Check if network architecture changed
            new_stats = get_network_stats(self.network)
            if self.iteration == 1:
                growth_occurred = True  # First iteration always counts as growth
            else:
                # Compare with previous iteration
                prev_record = self.growth_history[-1] if self.growth_history else None
                if prev_record:
                    growth_occurred = (new_stats['total_connections'] != prev_record['total_connections'] or
                                     new_stats['architecture'] != prev_record['architecture'])
        
        # Get evolution summary for detailed metrics
        summary = self.evolution_system.get_evolution_summary()
        
        # Record growth event
        stats = get_network_stats(self.network)
        self.growth_history.append({
            'iteration': self.iteration,
            'architecture': stats['architecture'],
            'accuracy': accuracy,
            'total_connections': stats['total_connections'],
            'sparsity': stats['overall_sparsity'],
            'growth_occurred': growth_occurred,
            'evolution_metrics': summary['metrics'],
            'components_used': summary['components']
        })
        
        print(f"📊 Iteration {self.iteration}: Acc {accuracy:.2%}, Growth: {'Yes' if growth_occurred else 'No'}")
        print(f"📐 Architecture: {stats['architecture']}")
        print(f"🔗 Connections: {stats['total_connections']:,}")
        print(f"🕳️  Sparsity: {stats['overall_sparsity']:.1%}")
        
        return accuracy, growth_occurred
    
    def grow_until_target_accuracy(self, target_accuracy, train_loader, test_loader, max_iterations=10):
        """Main growth loop - grow until target accuracy."""
        print(f"🎯 MODERN COMPOSABLE GROWTH EXPERIMENT")
        print("=" * 60)
        print(f"🎯 Target accuracy: {target_accuracy:.1%}")
        print(f"🌱 Starting architecture: {self.current_architecture}")
        print(f"🧬 Evolution strategy: {self.growth_strategy}")
        print(f"🔧 Components: {len(self.evolution_system.get_components())}")
        
        while self.current_accuracy < target_accuracy and self.iteration < max_iterations:
            accuracy, growth_occurred = self.growth_step(train_loader, test_loader)
            
            # Save checkpoint on significant improvements
            if accuracy > 0.7 and (not hasattr(self, '_last_saved_acc') or accuracy > self._last_saved_acc + 0.05):
                try:
                    current_stats = get_network_stats(self.network)
                    save_model_seed(
                        model=self.network,
                        architecture=current_stats['architecture'],
                        seed=42,
                        metrics={
                            'accuracy': accuracy,
                            'iteration': self.iteration,
                            'composable_growth': True,
                            'strategy': self.growth_strategy
                        },
                        filepath=f"data/composable_growth_iter{self.iteration}_acc{accuracy:.2f}.pt"
                    )
                    self._last_saved_acc = accuracy
                    print(f"   💾 Saved checkpoint (significant improvement)")
                except Exception as e:
                    print(f"   ⚠️  Failed to save checkpoint: {e}")
            
            if accuracy >= target_accuracy:
                print(f"\n🎉 TARGET ACCURACY {target_accuracy:.1%} ACHIEVED!")
                break
            
            if not growth_occurred:
                print(f"\n⚠️  No growth occurred - network may have converged")
        
        print(f"\n🏁 GROWTH COMPLETED")
        print(f"📊 Final accuracy: {self.current_accuracy:.2%}")
        print(f"🌱 Growth iterations: {self.iteration}")
        print(f"📐 Final architecture: {self.current_architecture}")
        
        # Print component metrics
        final_metrics = self.evolution_system.get_metrics()
        print(f"\n📈 Component Performance:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value}")
        
        return self.current_accuracy, self.growth_history
    
    def save_growth_summary(self, filepath="data/composable_growth_results.json"):
        """Save comprehensive growth summary."""
        stats = get_network_stats(self.network)
        
        summary = {
            'experiment_type': 'composable_growth',
            'timestamp': datetime.now().isoformat(),
            'final_accuracy': self.current_accuracy,
            'growth_iterations': self.iteration,
            'final_architecture': stats['architecture'],
            'final_connections': stats['total_connections'],
            'final_sparsity': stats['overall_sparsity'],
            'seed_architecture': self.seed_architecture,
            'scaffold_sparsity': self.scaffold_sparsity,
            'growth_strategy': self.growth_strategy,
            'allow_patches': self.allow_patches,
            'growth_history': self.growth_history,
            'evolution_system': {
                'components': self.evolution_system.get_configuration(),
                'final_metrics': self.evolution_system.get_metrics(),
                'summary': self.evolution_system.get_evolution_summary()
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Growth summary saved to {filepath}")
        return summary


def run_composable_experiment(device, growth_strategy, seed_arch, target_accuracy, 
                            dataset, output_filename, allow_patches=True):
    """Worker function to run a composable growth experiment."""
    print(f"🚀 Starting Composable Growth experiment on {device}")
    print(f"   Strategy: {growth_strategy}")
    print(f"   Patches: {'Enabled' if allow_patches else 'Disabled'}")
    
    # Load data
    if dataset == 'mnist':
        train_loader, test_loader = load_mnist_data()
    else:
        train_loader, test_loader = load_cifar10_data()
    
    # Create and run the growth engine
    engine = ModernComposableGrowth(
        seed_architecture=seed_arch,
        scaffold_sparsity=0.02,
        device=device,
        allow_patches=allow_patches,
        growth_strategy=growth_strategy
    )
    
    engine.grow_until_target_accuracy(
        target_accuracy=target_accuracy,
        train_loader=train_loader,
        test_loader=test_loader
    )
    engine.save_growth_summary(output_filename)
    print(f"✅ Composable Growth experiment on {device} finished. Results saved to {output_filename}")


def run_strategy_comparison(device, seed_arch, target_accuracy, dataset, output_prefix):
    """Compare different evolution strategies."""
    print(f"🚀 Starting Strategy Comparison on {device}")
    
    strategies = ['standard', 'extrema_focused', 'hybrid', 'custom']
    results = {}
    
    # Load data once
    if dataset == 'mnist':
        train_loader, test_loader = load_mnist_data()
    else:
        train_loader, test_loader = load_cifar10_data()
    
    for strategy in strategies:
        print(f"\n🔬 Testing strategy: {strategy}")
        
        engine = ModernComposableGrowth(
            seed_architecture=seed_arch,
            scaffold_sparsity=0.02,
            device=device,
            allow_patches=True,
            growth_strategy=strategy
        )
        
        final_acc, history = engine.grow_until_target_accuracy(
            target_accuracy=target_accuracy,
            train_loader=train_loader,
            test_loader=test_loader,
            max_iterations=3  # Shorter for comparison
        )
        
        results[strategy] = {
            'final_accuracy': final_acc,
            'iterations': len(history),
            'final_architecture': engine.current_architecture,
            'growth_history': history
        }
        
        # Save individual results
        engine.save_growth_summary(f"{output_prefix}_{strategy}.json")
    
    # Save comparison summary
    comparison_summary = {
        'experiment_type': 'strategy_comparison',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'seed_architecture': seed_arch,
        'target_accuracy': target_accuracy,
        'dataset': dataset,
        'results': results
    }
    
    with open(f"{output_prefix}_comparison.json", 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    print(f"\n📊 Strategy Comparison Results:")
    for strategy, result in results.items():
        print(f"   {strategy}: {result['final_accuracy']:.2%} in {result['iterations']} iterations")


def load_cifar10_data(batch_size=64):
    """Load CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_mnist_data(batch_size=64):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def main():
    """Main function for modern composable growth experiment."""
    import argparse
    parser = argparse.ArgumentParser(description='Modern Composable Growth Experiment')
    parser.add_argument('--type', type=str, choices=['mnist', 'cifar'], default='mnist', 
                       help='Dataset to use')
    parser.add_argument('--target-accuracy', type=float, default=0.95, 
                       help='Target accuracy to grow towards')
    parser.add_argument('--seed-arch', type=str, default='784,128,10', 
                       help='Seed architecture (comma-separated)')
    parser.add_argument('--strategy', type=str, 
                       choices=['standard', 'extrema_focused', 'hybrid', 'custom'], 
                       default='custom', help='Evolution strategy to use')
    parser.add_argument('--no-patches', action='store_true', 
                       help='Disable patch addition')
    parser.add_argument('-c', '--compare', action='store_true', 
                       help='Compare all evolution strategies')
    parser.add_argument('-m', '--multi-gpu', action='store_true', 
                       help='Run multi-GPU comparison')
    args = parser.parse_args()

    # Determine architecture based on dataset type
    if args.type == 'mnist':
        seed_arch = [int(x) for x in args.seed_arch.split(',')] if args.seed_arch else [784, 128, 10]
    else:  # cifar
        seed_arch = [3072, 128, 10]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.compare:
        # Compare all strategies
        run_strategy_comparison(
            device=device,
            seed_arch=seed_arch,
            target_accuracy=args.target_accuracy,
            dataset=args.type,
            output_prefix='data/strategy_comparison'
        )
    elif args.multi_gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print("⚠️  Multi-GPU mode requires at least 2 available GPUs. Exiting.")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("🚀 MULTI-GPU COMPOSABLE GROWTH COMPARISON")
        print("="*60)

        procs = []
        strategies = ['standard', 'extrema_focused', 'hybrid', 'custom']
        
        for i, strategy in enumerate(strategies):
            gpu_id = f'cuda:{i % torch.cuda.device_count()}'
            output_file = f'data/composable_{strategy}_growth.json'
            
            p = mp.Process(target=run_composable_experiment, args=(
                gpu_id, strategy, seed_arch, args.target_accuracy, args.type, 
                output_file, not args.no_patches
            ))
            procs.append(p)
            p.start()

        for p in procs:
            p.join()

        print("\n✅ Multi-GPU comparison complete.")
    else:
        # Single experiment
        run_composable_experiment(
            device=device,
            growth_strategy=args.strategy,
            seed_arch=seed_arch,
            target_accuracy=args.target_accuracy,
            dataset=args.type,
            output_filename=f'data/composable_{args.strategy}_results.json',
            allow_patches=not args.no_patches
        )


if __name__ == "__main__":
    # Set start method for multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
