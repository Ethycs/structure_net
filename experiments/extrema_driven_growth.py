#!/usr/bin/env python3
"""
Extrema-Driven Growth Experiment

This experiment bypasses MI analysis and connects extrema detection directly
to growth decisions. Implements neck block creation when large dead zones are detected.

Key Innovation: Direct extrema → growth pipeline for dramatic performance improvement.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.structure_net import (
    create_standard_network,
    save_model_seed,
    load_model_seed,
    get_network_stats,
    sort_all_network_layers
)
from src.structure_net.evolution.extrema_analyzer import detect_network_extrema

class ExtremaGrowthEngine:
    """
    Direct extrema-to-growth engine bypassing MI analysis.
    
    Implements immediate growth decisions based on extrema patterns:
    - Large dead zones → Neck block creation
    - Saturated clusters → Relief connections
    - Isolated extrema → Targeted patches
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.network = None
        self.growth_history = []
        self.current_accuracy = 0.0
        
        # Growth thresholds
        self.dead_zone_threshold = 50  # Neurons to trigger neck block
        self.saturation_threshold = 10  # Neurons to trigger relief
        self.isolation_threshold = 5   # Isolated extrema for patches
        
    def load_network(self, checkpoint_path):
        """Load pretrained network."""
        print(f"🔬 Loading network: {checkpoint_path}")
        self.network, metadata = load_model_seed(checkpoint_path, device=self.device)
        self.current_accuracy = metadata.get('accuracy', 0.0)
        print(f"   ✅ Loaded: {metadata['architecture']}, Acc: {self.current_accuracy:.2%}")
        return self.network
    
    def analyze_extrema_patterns(self, train_loader):
        """Analyze extrema patterns for direct growth decisions."""
        print("\n🔍 EXTREMA PATTERN ANALYSIS")
        print("=" * 40)
        
        # Get extrema patterns
        extrema_patterns = detect_network_extrema(
            self.network, 
            train_loader, 
            str(self.device),
            max_batches=5
        )
        
        # Analyze patterns for growth opportunities
        growth_decisions = []
        
        # Handle extrema_patterns as either dict or list
        if isinstance(extrema_patterns, dict):
            pattern_items = extrema_patterns.items()
        else:
            # Convert list to dict with indices
            pattern_items = enumerate(extrema_patterns)
        
        for layer_idx, pattern in pattern_items:
            dead_count = len(pattern.get('low', []))
            saturated_count = len(pattern.get('high', []))
            
            print(f"📊 Layer {layer_idx}: {dead_count} dead, {saturated_count} saturated")
            
            # Decision 1: Large dead zone → Neck block
            if dead_count >= self.dead_zone_threshold:
                growth_decisions.append({
                    'type': 'neck_block',
                    'layer': layer_idx,
                    'dead_neurons': pattern['low'],
                    'severity': dead_count / (dead_count + saturated_count + 1),
                    'priority': 'HIGH'
                })
                print(f"   🚨 HIGH PRIORITY: Neck block needed ({dead_count} dead neurons)")
            
            # Decision 2: Saturated cluster → Relief connections
            elif saturated_count >= self.saturation_threshold:
                growth_decisions.append({
                    'type': 'relief_connections',
                    'layer': layer_idx,
                    'saturated_neurons': pattern['high'],
                    'severity': saturated_count / (dead_count + saturated_count + 1),
                    'priority': 'MEDIUM'
                })
                print(f"   ⚡ MEDIUM PRIORITY: Relief connections ({saturated_count} saturated)")
            
            # Decision 3: Isolated extrema → Targeted patches
            elif dead_count >= self.isolation_threshold or saturated_count >= self.isolation_threshold:
                growth_decisions.append({
                    'type': 'targeted_patches',
                    'layer': layer_idx,
                    'dead_neurons': pattern['low'],
                    'saturated_neurons': pattern['high'],
                    'severity': (dead_count + saturated_count) / 100,
                    'priority': 'LOW'
                })
                print(f"   🎯 LOW PRIORITY: Targeted patches ({dead_count + saturated_count} extrema)")
        
        # Sort by priority and severity
        priority_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        growth_decisions.sort(key=lambda x: (priority_order[x['priority']], x['severity']), reverse=True)
        
        return growth_decisions
    
    def create_neck_block(self, layer_idx, dead_neurons, neck_size=64):
        """
        Create a neck block to revive large dead zones.
        
        A neck block is a dense bypass that routes around dead neurons
        and provides alternative pathways for information flow.
        """
        print(f"\n🏗️  CREATING NECK BLOCK")
        print(f"   Target layer: {layer_idx}")
        print(f"   Dead neurons: {len(dead_neurons)}")
        print(f"   Neck size: {neck_size}")
        
        # Get current architecture
        stats = get_network_stats(self.network)
        current_arch = stats['architecture']
        
        # Insert neck block layer
        new_arch = current_arch.copy()
        new_arch.insert(layer_idx + 1, neck_size)
        
        print(f"   Old architecture: {current_arch}")
        print(f"   New architecture: {new_arch}")
        
        # Create new network with neck block
        new_network = create_standard_network(
            architecture=new_arch,
            sparsity=0.05,  # Dense neck block
            device=self.device
        )
        
        # Copy weights from old network, preserving learned features
        self._transfer_weights_with_neck_block(self.network, new_network, layer_idx, neck_size)
        
        self.network = new_network
        print(f"   ✅ Neck block created and weights transferred")
        
        return True
    
    def create_relief_connections(self, layer_idx, saturated_neurons):
        """
        Create relief connections to distribute saturated neuron load.
        
        Adds sparse connections from saturated neurons to underutilized areas.
        """
        print(f"\n⚡ CREATING RELIEF CONNECTIONS")
        print(f"   Target layer: {layer_idx}")
        print(f"   Saturated neurons: {len(saturated_neurons)}")
        
        # Get sparse layers
        sparse_layers = [layer for layer in self.network if hasattr(layer, 'mask')]
        
        if layer_idx < len(sparse_layers):
            target_layer = sparse_layers[layer_idx]
            
            # Add relief connections from saturated neurons
            with torch.no_grad():
                for neuron_idx in saturated_neurons[:10]:  # Limit to top 10
                    if neuron_idx < target_layer.mask.shape[1]:
                        # Add sparse connections to random outputs
                        relief_mask = torch.rand(target_layer.mask.shape[0]) < 0.1
                        target_layer.mask[:, neuron_idx] = torch.maximum(
                            target_layer.mask[:, neuron_idx],
                            relief_mask.float()
                        )
                        
                        # Initialize new connections with small weights
                        new_connections = relief_mask & (target_layer.mask[:, neuron_idx] > 0)
                        target_layer.linear.weight.data[new_connections, neuron_idx] *= 0.1
            
            print(f"   ✅ Relief connections added for {len(saturated_neurons)} neurons")
            return True
        
        return False
    
    def create_targeted_patches(self, layer_idx, dead_neurons, saturated_neurons):
        """
        Create targeted patches for isolated extrema.
        
        Adds small dense patches to handle specific extrema patterns.
        """
        print(f"\n🎯 CREATING TARGETED PATCHES")
        print(f"   Target layer: {layer_idx}")
        print(f"   Dead neurons: {len(dead_neurons)}")
        print(f"   Saturated neurons: {len(saturated_neurons)}")
        
        # For now, implement as enhanced connectivity
        sparse_layers = [layer for layer in self.network if hasattr(layer, 'mask')]
        
        if layer_idx < len(sparse_layers):
            target_layer = sparse_layers[layer_idx]
            
            with torch.no_grad():
                # Enhance connectivity for dead neurons (inputs)
                for neuron_idx in dead_neurons[:5]:
                    if neuron_idx < target_layer.mask.shape[0]:
                        enhancement = torch.rand(target_layer.mask.shape[1]) < 0.15
                        target_layer.mask[neuron_idx, :] = torch.maximum(
                            target_layer.mask[neuron_idx, :],
                            enhancement.float()
                        )
                
                # Enhance connectivity for saturated neurons (outputs)
                for neuron_idx in saturated_neurons[:5]:
                    if neuron_idx < target_layer.mask.shape[1]:
                        enhancement = torch.rand(target_layer.mask.shape[0]) < 0.15
                        target_layer.mask[:, neuron_idx] = torch.maximum(
                            target_layer.mask[:, neuron_idx],
                            enhancement.float()
                        )
            
            print(f"   ✅ Targeted patches created")
            return True
        
        return False
    
    def _transfer_weights_with_neck_block(self, old_network, new_network, neck_position, neck_size):
        """Transfer weights from old network to new network with neck block insertion."""
        old_sparse = [layer for layer in old_network if hasattr(layer, 'mask')]
        new_sparse = [layer for layer in new_network if hasattr(layer, 'mask')]
        
        print(f"   🔄 Transferring weights: {len(old_sparse)} → {len(new_sparse)} layers")
        
        with torch.no_grad():
            new_idx = 0
            for old_idx, old_layer in enumerate(old_sparse):
                if old_idx == neck_position:
                    # Skip the neck block layer (keep random initialization)
                    print(f"      Skipping neck block at position {new_idx}")
                    new_idx += 1
                
                if new_idx < len(new_sparse):
                    new_layer = new_sparse[new_idx]
                    
                    # Copy compatible weights
                    min_out = min(old_layer.linear.weight.shape[0], new_layer.linear.weight.shape[0])
                    min_in = min(old_layer.linear.weight.shape[1], new_layer.linear.weight.shape[1])
                    
                    new_layer.linear.weight.data[:min_out, :min_in] = old_layer.linear.weight.data[:min_out, :min_in]
                    new_layer.linear.bias.data[:min_out] = old_layer.linear.bias.data[:min_out]
                    new_layer.mask[:min_out, :min_in] = old_layer.mask[:min_out, :min_in]
                    
                    print(f"      Copied layer {old_idx} → {new_idx}: {min_out}x{min_in}")
                    new_idx += 1
    
    def apply_growth_decision(self, decision):
        """Apply a single growth decision."""
        if decision['type'] == 'neck_block':
            return self.create_neck_block(
                decision['layer'], 
                decision['dead_neurons']
            )
        elif decision['type'] == 'relief_connections':
            return self.create_relief_connections(
                decision['layer'],
                decision['saturated_neurons']
            )
        elif decision['type'] == 'targeted_patches':
            return self.create_targeted_patches(
                decision['layer'],
                decision['dead_neurons'],
                decision['saturated_neurons']
            )
        return False
    
    def extrema_driven_growth_step(self, train_loader, test_loader):
        """Perform one extrema-driven growth step."""
        print("\n🧬 EXTREMA-DRIVEN GROWTH STEP")
        print("=" * 50)
        
        # Analyze extrema patterns
        growth_decisions = self.analyze_extrema_patterns(train_loader)
        
        if not growth_decisions:
            print("✅ No growth needed - network is well-balanced")
            return False
        
        # Apply the highest priority decision
        top_decision = growth_decisions[0]
        print(f"\n🎯 Applying {top_decision['priority']} priority action: {top_decision['type']}")
        
        growth_occurred = self.apply_growth_decision(top_decision)
        
        if growth_occurred:
            # Apply neuron sorting after growth
            sort_all_network_layers(self.network)
            print("   🔄 Applied neuron sorting")
        
        return growth_occurred
    
    def evaluate_network(self, test_loader):
        """Evaluate network performance."""
        self.network.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)  # Flatten for sparse networks
                output = self.network(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        self.current_accuracy = accuracy
        print(f"📊 Current accuracy: {accuracy:.2%}")
        return accuracy
    
    def train_network(self, train_loader, test_loader, epochs=5):
        """Train network."""
        print(f"🚀 Training network for {epochs} epochs")
        
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.network.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)  # Flatten for sparse networks
                
                optimizer.zero_grad()
                output = self.network(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                accuracy = self.evaluate_network(test_loader)
                print(f"   Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}, Acc {accuracy:.2%}")
        
        return self.evaluate_network(test_loader)
    
    def run_extrema_driven_experiment(self, train_loader, test_loader, target_accuracy=0.80, max_iterations=5):
        """Run the extrema-driven growth experiment."""
        print("🔬 EXTREMA-DRIVEN GROWTH EXPERIMENT")
        print("=" * 60)
        print("🎯 Innovation: Direct extrema → growth pipeline")
        print("🏗️  Key feature: Neck block creation for dead zones")
        
        iteration = 0
        while self.current_accuracy < target_accuracy and iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            
            # Train current network
            accuracy = self.train_network(train_loader, test_loader, epochs=5)
            
            # Apply extrema-driven growth
            growth_occurred = self.extrema_driven_growth_step(train_loader, test_loader)
            
            print(f"📊 Iteration {iteration} complete: Acc {accuracy:.2%}, Growth: {'Yes' if growth_occurred else 'No'}")
            
            # Save checkpoint
            if accuracy > 0.3:
                checkpoint_path = f"data/extrema_growth_iter{iteration}_acc{accuracy:.2f}.pt"
                stats = get_network_stats(self.network)
                
                save_model_seed(
                    model=self.network,
                    architecture=stats['architecture'],
                    seed=42,
                    metrics={'accuracy': accuracy, 'iteration': iteration, 'extrema_driven': True},
                    filepath=checkpoint_path
                )
                print(f"   💾 Saved checkpoint: {checkpoint_path}")
        
        print(f"\n✅ Experiment complete! Final accuracy: {self.current_accuracy:.2%}")
        return self.growth_history

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

def main():
    """Main function for extrema-driven growth experiment."""
    import argparse
    parser = argparse.ArgumentParser(description='Extrema-Driven Growth Experiment')
    parser.add_argument('--load-model', type=str, required=True, help='Path to pretrained model checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_cifar10_data()
    
    # Create extrema-driven growth engine
    growth_engine = ExtremaGrowthEngine(device=device)
    
    # Load pretrained network
    growth_engine.load_network(args.load_model)
    
    # Run extrema-driven growth experiment
    history = growth_engine.run_extrema_driven_experiment(
        train_loader=train_loader,
        test_loader=test_loader,
        target_accuracy=0.80,
        max_iterations=5
    )
    
    print("\n🎯 EXTREMA-DRIVEN GROWTH COMPLETE!")
    print("Key innovation: Direct extrema analysis → immediate growth decisions")
    print("Expected result: Dramatic performance improvement via neck blocks")

if __name__ == "__main__":
    main()
