#!/usr/bin/env python3
"""
Full MNIST Dataset Experiments

This script conducts a battery of tests on the full MNIST dataset to validate
the findings from smaller-scale experiments.

Tests:
1. Goldilocks Zone Test: Validates optimal sparsity levels at scale.
2. Growth Mechanism Test: Tests the cliff rescue mechanism on the full dataset.
3. Efficiency Frontier Mapping: Finds the limits of parameter reduction.
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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.structure_net.core.minimal_network import MinimalNetwork
from src.structure_net.models.multi_scale_network import create_multi_scale_network
from src.structure_net.core.growth_scheduler import GrowthScheduler, StructuralLimits

class FullMNISTExperiment:
    """Conducts experiments on the full MNIST dataset."""

    def __init__(self, save_dir="full_mnist_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Use RTX 2060s (GPU 0 or 2) instead of Blackwell GPU
        if torch.cuda.is_available():
            # Check available GPUs and prefer RTX 2060s
            gpu_count = torch.cuda.device_count()
            selected_gpu = 0  # Default to GPU 0 (RTX 2060)
            
            # Print available GPUs for reference
            print("🖥️  Available GPUs:")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
            
            # Use RTX 2060 (avoid GPU 1 which is the Blackwell)
            if gpu_count > 2:
                selected_gpu = 2  # Use RTX 2060 Super on GPU 2
            else:
                selected_gpu = 0  # Use RTX 2060 on GPU 0
                
            self.device = torch.device(f'cuda:{selected_gpu}')
            print(f"🎯 Selected device: {self.device} ({torch.cuda.get_device_name(selected_gpu)})")
        else:
            self.device = torch.device('cpu')
            print(f"🖥️  Using device: {self.device}")

    def load_mnist_data(self, batch_size=64):
        """Load the full MNIST dataset."""
        print("📦 Loading Full MNIST dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples.")
        return train_loader, test_loader

    def train_network(self, network, train_loader, test_loader, epochs, lr=0.001):
        """Generic training loop for a network."""
        network = network.to(self.device)
        optimizer = optim.Adam(network.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = {'epoch': [], 'train_acc': [], 'test_acc': [], 'loss': []}
        best_test_acc = 0

        for epoch in range(epochs):
            # Training phase
            network.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)  # Flatten images
                
                optimizer.zero_grad()
                output = network(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Track training statistics
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()

            # Calculate training accuracy
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # Test evaluation
            network.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    data = data.view(data.size(0), -1)
                    output = network(data)
                    loss = criterion(output, target)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()
            
            test_acc = 100 * test_correct / test_total
            avg_test_loss = test_loss / len(test_loader)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc

            # Update history
            history['epoch'].append(epoch)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['loss'].append(avg_train_loss)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        return {'best_accuracy': best_test_acc, 'history': history}

    def run_goldilocks_zone_test(self):
        """Test 1: Validate Goldilocks Zone Scales."""
        print("\n" + "="*60)
        print("🔬 Test 1: Goldilocks Zone Test")
        print("="*60)
        
        # Test parameters
        sparsity_levels = [0.02, 0.01, 0.005, 0.002, 0.001]
        architectures = [[256], [256, 128], [512, 256, 128]]
        input_size = 784
        output_size = 10
        epochs = 10 # A shorter run for this broad test

        train_loader, test_loader = self.load_mnist_data()
        results = {}

        for arch in architectures:
            arch_key = str(arch)
            results[arch_key] = []
            print(f"\nARCHITECTURE: {arch}")
            for sparsity in sparsity_levels:
                print(f"  Sparsity: {sparsity}")
                network = MinimalNetwork(
                    layer_sizes=[input_size] + arch + [output_size],
                    sparsity=sparsity,
                    activation='relu', # ReLU is common for MNIST
                    device=self.device
                )
                
                stats = self.train_network(network, train_loader, test_loader, epochs)
                stats['sparsity'] = sparsity
                stats['architecture'] = arch
                results[arch_key].append(stats)

        with open(os.path.join(self.save_dir, 'goldilocks_zone_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ Goldilocks Zone Test completed.")
        return results

    def run_growth_rescue_test(self):
        """Test 2: Growth Mechanism at Scale."""
        print("\n" + "="*60)
        print("🔬 Test 2: Growth Mechanism Test at Scale")
        print("="*60)
        
        # Test parameters
        cliff_sparsity = 0.002  # Very sparse network that should struggle
        target_performance = 85.0  # 85% accuracy target for MNIST
        epochs = 20
        
        train_loader, test_loader = self.load_mnist_data()
        results = {}
        
        print(f"🎯 Goal: Rescue network with {cliff_sparsity} sparsity")
        print(f"📈 Target: {target_performance:.1f}% performance on MNIST")
        
        # Test 1: Static baseline (for comparison)
        print(f"\n📊 BASELINE: Static Sparse Network")
        print("-" * 40)
        
        static_network = MinimalNetwork(
            layer_sizes=[784, 256, 128, 10],
            sparsity=cliff_sparsity,
            activation='relu',
            device=self.device
        )
        
        static_results = self.train_network(static_network, train_loader, test_loader, epochs)
        results['static'] = static_results
        
        # Test 2: Growth-enabled network using MultiScaleNetwork
        print(f"\n🌱 GROWTH: Multi-Scale Growth Network")
        print("-" * 40)
        
        try:
            growth_network = create_multi_scale_network(
                input_size=784,
                hidden_sizes=[256, 128],
                output_size=10,
                sparsity=cliff_sparsity,
                activation='relu',
                device=self.device,
                snapshot_dir=os.path.join(self.save_dir, "growth_snapshots")
            )
            
            growth_results = self.train_growth_network(growth_network, train_loader, test_loader, epochs)
            results['growth'] = growth_results
            
        except Exception as e:
            print(f"❌ Growth network failed: {e}")
            results['growth'] = {'error': str(e), 'best_accuracy': 0}
        
        # Analysis
        self.analyze_growth_rescue_results(results, target_performance)
        
        # Save results
        with open(os.path.join(self.save_dir, 'growth_rescue_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n✅ Growth Rescue Test completed.")
        return results

    def train_growth_network(self, network, train_loader, test_loader, epochs, lr=0.001):
        """Train growth-enabled network with multi-scale capabilities."""
        optimizer = optim.Adam(network.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = {'epoch': [], 'train_acc': [], 'test_acc': [], 'loss': [], 'growth_events': []}
        best_test_acc = 0

        for epoch in range(epochs):
            # Use the network's built-in training method if available
            if hasattr(network, 'train_epoch'):
                epoch_stats = network.train_epoch(train_loader, optimizer, criterion, epoch)
                eval_stats = network.evaluate(test_loader, criterion)
                
                train_acc = epoch_stats['performance'] * 100
                test_acc = eval_stats['performance'] * 100
                avg_train_loss = epoch_stats['loss']
                
                # Track growth events
                if epoch_stats.get('growth_events', 0) > 0:
                    history['growth_events'].append({
                        'epoch': epoch,
                        'connections_added': epoch_stats.get('connections_added', 0),
                        'total_connections': epoch_stats['total_connections']
                    })
                    print(f"   🌱 GROWTH EVENT! Added {epoch_stats.get('connections_added', 0)} connections")
                
            else:
                # Fallback to standard training
                network.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    data = data.view(data.size(0), -1)
                    
                    optimizer.zero_grad()
                    output = network(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    train_total += target.size(0)
                    train_correct += (predicted == target).sum().item()

                train_acc = 100 * train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)

                # Test evaluation
                network.eval()
                test_correct = 0
                test_total = 0
                
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        data = data.view(data.size(0), -1)
                        output = network(data)
                        _, predicted = torch.max(output.data, 1)
                        test_total += target.size(0)
                        test_correct += (predicted == target).sum().item()
                
                test_acc = 100 * test_correct / test_total

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            # Update history
            history['epoch'].append(epoch)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['loss'].append(avg_train_loss)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        return {'best_accuracy': best_test_acc, 'history': history}

    def analyze_growth_rescue_results(self, results, target_performance):
        """Analyze the results of the growth rescue test."""
        print(f"\n📊 GROWTH RESCUE ANALYSIS")
        print("=" * 60)
        
        static_perf = results['static']['best_accuracy']
        growth_perf = results['growth']['best_accuracy'] if 'error' not in results['growth'] else 0
        
        print(f"🎯 Target Performance: {target_performance:.1f}%")
        print(f"📉 Static Performance: {static_perf:.1f}%")
        print(f"🌱 Growth Performance: {growth_perf:.1f}%")
        
        if growth_perf > 0:
            improvement = growth_perf - static_perf
            improvement_pct = (improvement / static_perf) * 100 if static_perf > 0 else 0
            
            print(f"\n🔍 RESCUE ANALYSIS:")
            print(f"   Improvement: {improvement:.1f}% ({improvement_pct:+.1f}%)")
            
            # Count growth events
            growth_events = len(results['growth']['history'].get('growth_events', []))
            print(f"   Growth events: {growth_events}")
            
            if growth_perf >= target_performance:
                print(f"\n🎉 CLIFF RESCUE SUCCESS!")
                print(f"   ✅ Growth mechanism rescued the sparse network!")
            elif improvement > 5.0:  # 5% improvement
                print(f"\n📈 PARTIAL CLIFF RESCUE")
                print(f"   📊 Significant improvement achieved")
            else:
                print(f"\n❌ CLIFF RESCUE FAILED")
                print(f"   🔧 Growth mechanism needs improvement")
        else:
            print(f"\n❌ GROWTH NETWORK ERROR")
            print(f"   Error: {results['growth'].get('error', 'Unknown error')}")

    def run_efficiency_frontier_test(self):
        """Test 3: Efficiency Frontier Mapping."""
        print("\n" + "="*60)
        print("🔬 Test 3: Efficiency Frontier Mapping")
        print("="*60)
        # This will require an iterative search.
        # For now, this is a placeholder.
        print("⚠️  Efficiency frontier test not yet implemented.")
        return {}


def main():
    """Main entry point for the script."""
    experiment = FullMNISTExperiment()
    
    # Run all tests
    goldilocks_results = experiment.run_goldilocks_zone_test()
    # growth_results = experiment.run_growth_rescue_test()
    # efficiency_results = experiment.run_efficiency_frontier_test()

    print("\n� All experiments completed.")

if __name__ == "__main__":
    main()
