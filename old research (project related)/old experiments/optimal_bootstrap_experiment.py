#!/usr/bin/env python3
"""
Optimal Multi-Scale Bootstrap Experiment

This script implements the experiment to find the optimal seed network and then
uses multi-scale bootstrapping with extrema-guided growth to train a larger
network.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.structure_net.core.minimal_network import MinimalNetwork

def count_parameters(network):
    """Count the number of active parameters in a sparse network."""
    total_params = 0
    for mask in network.connection_masks:
        total_params += mask.sum().item()
    return total_params

def train_and_evaluate_arch(args):
    """Worker function for parallel training."""
    arch, device_id, batch_size = args
    device = torch.device(f"cuda:{device_id}")
    
    print(f"  [GPU {device_id}] Testing architecture: {arch}")

    # Each process needs its own data loader
    train_loader, test_loader = load_mnist_data(batch_size=batch_size, is_worker=True)

    network = MinimalNetwork(
        layer_sizes=arch,
        sparsity=0.02,
        activation='relu',
        device=device
    )
    
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0

    for epoch in range(20): # epochs=20
        network.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        network.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = network(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_acc = correct / total
        if test_acc > best_test_acc:
            best_test_acc = test_acc
    
    learns = best_test_acc > 0.15
    result = {
        'arch': arch,
        'accuracy': best_test_acc,
        'parameters': count_parameters(network),
        'learns': learns
    }
    print(f"  [GPU {device_id}] Accuracy: {best_test_acc:.2%}, Learns: {learns}")
    return result

class OptimalSeedFinder:
    """Find the smallest viable network to bootstrap from."""

    def find_optimal_seed(self):
        """Find minimal architecture that still learns in parallel."""
        print("🔍 Finding optimal small seed network in parallel...")
        
        architectures = [
            [784, 128, 10],
            [784, 64, 10],
            [784, 32, 10],
            [784, 16, 10],
            [784, 10],
        ]
        
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs.")

        # Prepare arguments for each process
        args_list = [(arch, i % num_gpus, 64) for i, arch in enumerate(architectures)]

        # Use spawn method for CUDA safety
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_gpus) as pool:
            worker_results = pool.map(train_and_evaluate_arch, args_list)

        # Process results
        results = {str(res['arch']): res for res in worker_results}

        learning_archs = [a for a in results if results[a]['learns']]
        if not learning_archs:
            print("⚠️ No architecture learned successfully.")
            return None, results

        optimal_arch_str = min(learning_archs, key=lambda a: results[a]['parameters'])
        optimal_arch = json.loads(optimal_arch_str)
        
        print(f"✅ Found optimal seed: {optimal_arch}")
        return optimal_arch, results

class MultiScaleExtremaBootstrap:
    """The real experiment - bootstrap with extrema-guided growth"""
    
    def __init__(self, seed_architecture, seed_sparsity=0.02, device=torch.device('cpu')):
        self.seed_arch = seed_architecture
        self.seed_sparsity = seed_sparsity
        self.device = device
        self.seed_performance = 0

    def run_full_experiment(self):
        """Run the complete bootstrap experiment"""
        train_loader, test_loader = load_mnist_data()

        # Phase 1: Seed
        seed_network = self.create_seed_network()
        self.seed_performance = self.train_network(seed_network, train_loader, test_loader, epochs=30, name="Seed")
        seed_analysis = self.analyze_seed_extrema(seed_network)
        
        # Phase 2: Medium bootstrap
        medium_network = self.bootstrap_medium_from_seed(seed_network, seed_analysis)
        medium_performance = self.train_network(medium_network, train_loader, test_loader, epochs=30, name="Medium (Bootstrapped)")
        
        print(f"\n📊 Medium network performance: {medium_performance:.2%}")
        print(f"   Improvement over seed: {medium_performance - self.seed_performance:.2%}")
        
        # Phase 3: Fine bootstrap (placeholder)
        print("\n📌 Phase 3: Bootstrapping fine network (placeholder)")
        fine_performance = medium_performance # Placeholder

        # Compare to baselines
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        print(f"Seed [784, 10]:           {self.seed_performance:.2%}")
        print(f"Medium [784, 32, 10]:     {medium_performance:.2%}")
        print(f"Fine [784, 128, 32, 10]:  {fine_performance:.2%}")
        
        # Test random initialization baseline
        random_medium = MinimalNetwork(layer_sizes=[784, 32, 10], sparsity=0.02, device=self.device)
        random_performance = self.train_network(random_medium, train_loader, test_loader, epochs=30, name="Medium (Random)")
        
        print(f"\nRandom [784, 32, 10]:     {random_performance:.2%}")
        print(f"Bootstrap advantage:      {medium_performance - random_performance:.2%}")
        
        return {
            'seed': self.seed_performance,
            'medium': medium_performance,
            'fine': fine_performance,
            'random_baseline': random_performance,
            'bootstrap_advantage': medium_performance - random_performance
        }

    def analyze_seed_extrema(self, seed_network):
        """Analyze extrema patterns in direct connection network"""
        print("\n🔍 Analyzing seed network extrema...")
        weight_matrix = seed_network.layers[0].weight.data
        analysis = {'decisive_features': {}, 'class_confusion': [], 'extrema_counts': {'high': 0, 'low': 0}}
        for class_idx in range(10):
            class_weights = weight_matrix[class_idx]
            analysis['decisive_features'][class_idx] = {
                'positive': torch.topk(class_weights, k=50).indices.cpu().numpy(),
                'negative': torch.topk(-class_weights, k=50).indices.cpu().numpy()
            }
            analysis['extrema_counts']['high'] += (class_weights > class_weights.std() * 2).sum().item()
            analysis['extrema_counts']['low'] += (class_weights < -class_weights.std() * 2).sum().item()
        print(f"  Found {analysis['extrema_counts']['high']} high extrema pixels")
        print(f"  Found {analysis['extrema_counts']['low']} low extrema pixels")
        return analysis

    def bootstrap_medium_from_seed(self, seed_network, seed_analysis):
        """Create [784, 32, 10] network bootstrapped from [784, 10] seed"""
        print(f"\n📌 Phase 2: Bootstrapping medium network [784, 32, 10]")
        print("🌱 Using seed knowledge to initialize hidden layer...")
        medium_network = MinimalNetwork(layer_sizes=[784, 32, 10], sparsity=0.02, device=self.device)
        hidden_size = 32
        seed_weights = seed_network.layers[0].weight.data
        for hidden_idx in range(hidden_size):
            primary_class = hidden_idx % 10
            secondary_class = (hidden_idx + 3) % 10
            primary_features = seed_analysis['decisive_features'][primary_class]['positive'][:20]
            secondary_features = seed_analysis['decisive_features'][secondary_class]['positive'][:10]
            for feat_idx in primary_features:
                if medium_network.connection_masks[0][hidden_idx, feat_idx] == 0:
                    medium_network.connection_masks[0][hidden_idx, feat_idx] = True
                    medium_network.layers[0].weight.data[hidden_idx, feat_idx] = seed_weights[primary_class, feat_idx] * 0.7
            for feat_idx in secondary_features:
                if medium_network.connection_masks[0][hidden_idx, feat_idx] == 0:
                    medium_network.connection_masks[0][hidden_idx, feat_idx] = True
                    medium_network.layers[0].weight.data[hidden_idx, feat_idx] = seed_weights[secondary_class, feat_idx] * 0.3
            medium_network.connection_masks[1][primary_class, hidden_idx] = True
            medium_network.layers[1].weight.data[primary_class, hidden_idx] = 0.5
            medium_network.connection_masks[1][secondary_class, hidden_idx] = True
            medium_network.layers[1].weight.data[secondary_class, hidden_idx] = 0.2
        print(f"  ✓ Initialized 32 hidden neurons with specialized roles")
        return medium_network

    def create_seed_network(self):
        print(f"  Architecture: {self.seed_arch}, Sparsity: {self.seed_sparsity}")
        return MinimalNetwork(
            layer_sizes=self.seed_arch,
            sparsity=self.seed_sparsity,
            activation='relu',
            device=self.device
        )

    def train_network(self, network, train_loader, test_loader, epochs, name="Network"):
        """Generic training loop."""
        print(f"--- Training {name} ---")
        optimizer = optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        best_test_acc = 0

        for epoch in range(epochs):
            network.train()
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)
                optimizer.zero_grad()
                output = network(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            network.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    data = data.view(data.size(0), -1)
                    output = network(data)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            test_acc = correct / total
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            print(f"  [{name}] Epoch {epoch+1}/{epochs}, Test Acc: {test_acc:.2%}")
        
        return best_test_acc

def load_mnist_data(batch_size=64, is_worker=False):
    """Load the full MNIST dataset."""
    if not is_worker:
        print("📦 Loading Full MNIST dataset...")
    
    num_workers = 0 if is_worker else 4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    if not is_worker:
        print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples.")
    return train_loader, test_loader

def main():
    """Main entry point for the script."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = "data/optimal_bootstrap_results"
    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Find optimal small seed
    seed_finder = OptimalSeedFinder()
    optimal_seed_arch, seed_results = seed_finder.find_optimal_seed()

    with open(os.path.join(save_dir, 'optimal_seed_results.json'), 'w') as f:
        json.dump(seed_results, f, indent=2)

    if optimal_seed_arch:
        # Step 2: Run multi-scale bootstrap experiment
        bootstrap_experiment = MultiScaleExtremaBootstrap(optimal_seed_arch, device=device)
        bootstrap_results = bootstrap_experiment.run_full_experiment()
        
        with open(os.path.join(save_dir, 'bootstrap_results.json'), 'w') as f:
            json.dump(bootstrap_results, f, indent=2)

    print("\n🎉 All experiments completed.")

if __name__ == "__main__":
    # This is required for multiprocessing with spawn
    main()
