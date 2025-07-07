#!/usr/bin/env python3
"""
Multi-Layer Growth Comparison Experiment

This experiment compares the effectiveness of adding 1, 2, or 3 sparse
layers at a time during network growth. It collects detailed metrics
and generates comparison charts to visualize the results.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.structure_net import (
    create_standard_network,
    get_network_stats
)

sparsity = 0.02  # Standard sparsity for all networks

# ============================================================================
# Data Collection and Experiment Runner
# ============================================================================

class MultiLayerGrowthRunner:
    """
    Manages a single experimental run for a specific growth strategy.
    """
    def __init__(self, num_layers_to_add, seed_arch, device):
        self.num_layers_to_add = num_layers_to_add
        self.device = device
        self.network = create_standard_network(seed_arch, sparsity, device=device)
        self.history = []
        self.epoch = 0

    def train_epoch(self, train_loader, test_loader):
        """Train for one epoch, collecting metrics twice."""
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        self.network.train()

        # --- Mid-epoch evaluation ---
        self.evaluate(test_loader, "mid_epoch")

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = self.network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # --- End-of-epoch evaluation ---
        self.evaluate(test_loader, "end_of_epoch")
        self.epoch += 1

    def evaluate(self, test_loader, timing):
        """Evaluate the network and record metrics."""
        self.network.eval()
        correct = 0
        total = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)
                output = self.network(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        stats = get_network_stats(self.network)

        self.history.append({
            'epoch': self.epoch,
            'timing': timing,
            'step': self.epoch + (0.5 if timing == 'mid_epoch' else 1.0),
            'accuracy': accuracy,
            'loss': avg_loss,
            'num_layers': len(stats['architecture']) - 1,
            'total_params': stats['total_parameters'],
            'strategy': f'add_{self.num_layers_to_add}_layers'
        })
        print(f"  [Strategy: +{self.num_layers_to_add}] Step {self.history[-1]['step']:.1f}: Acc {accuracy:.2%}, Loss {avg_loss:.4f}")

    def grow_network(self):
        """Applies the growth strategy."""
        print(f"  [Strategy: +{self.num_layers_to_add}] Growing network...")
        current_arch = get_network_stats(self.network)['architecture']
        
        # Add new layers in the middle
        insert_pos = len(current_arch) // 2
        new_layers = [128] * self.num_layers_to_add
        new_arch = current_arch[:insert_pos] + new_layers + current_arch[insert_pos:]

        self.network = create_standard_network(new_arch, sparsity, device=self.device)
        print(f"    New architecture: {new_arch}")

# ============================================================================
# Main Experiment Orchestration
# ============================================================================

def run_comparison_experiment(seed_arch, train_loader, test_loader, device, growth_iterations=3, epochs_per_iteration=5):
    """
    Runs a comparison of adding 1, 2, or 3 layers at each growth step.
    """
    all_results = []
    base_network = create_standard_network(seed_arch, sparsity, device=device)
    
    for i in range(growth_iterations):
        print(f"\n" + "="*60)
        print(f"🌱 GROWTH ITERATION {i+1}/{growth_iterations}")
        print(f"   Base Architecture: {get_network_stats(base_network)['architecture']}")
        print("="*60)

        # At each iteration, test all three growth strategies from the same base
        for num_layers_to_add in [1, 2, 3]:
            print(f"\n  🔬 Testing Strategy: Add {num_layers_to_add} Layer(s)")
            
            # Create a fresh runner with a copy of the base network
            runner = MultiLayerGrowthRunner(num_layers_to_add, get_network_stats(base_network)['architecture'], device)
            runner.network = copy.deepcopy(base_network) # Use a copy for the test
            
            # Apply the growth strategy to the copy
            runner.grow_network()
            
            # Train the candidate and collect metrics
            for epoch in range(epochs_per_iteration):
                runner.train_epoch(train_loader, test_loader)
            
            # Store the results from this candidate
            all_results.extend(runner.history)

        # After testing all strategies, we would normally select a winner and
        # update the base_network. For this comparison, we will just grow
        # the base network with the simplest strategy (add 1 layer) to
        # create the starting point for the next iteration's comparison.
        print("\n  📈 Updating base network for next iteration (using +1 layer strategy)...")
        base_arch = get_network_stats(base_network)['architecture']
        insert_pos = len(base_arch) // 2
        new_arch = base_arch[:insert_pos] + [128] + base_arch[insert_pos:]
        base_network = create_standard_network(new_arch, sparsity, device=device)

    return pd.DataFrame(all_results)

def plot_results(df, output_dir="data/comparison_charts"):
    """Generates and saves comparison charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot Accuracy vs. Step
    plt.figure(figsize=(12, 8))
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        plt.plot(strategy_df['step'], strategy_df['accuracy'], marker='o', linestyle='-', label=strategy)
    plt.title('Accuracy vs. Training Step')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()

    # Plot Loss vs. Step
    plt.figure(figsize=(12, 8))
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        plt.plot(strategy_df['step'], strategy_df['loss'], marker='o', linestyle='-', label=strategy)
    plt.title('Loss vs. Training Step')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'))
    plt.close()

    print(f"📊 Charts saved to {output_dir}")

def main():
    """Main function to run the experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_arch = [784, 128, 10] # MNIST
    
    # Load MNIST Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Run the experiment
    results_df = run_comparison_experiment(seed_arch, train_loader, test_loader, device)
    
    # Plot the results
    plot_results(results_df)

if __name__ == "__main__":
    main()
