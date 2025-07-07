#!/usr/bin/env python3
"""
Direct Extrema Growth - No MI Analysis

Bypasses all MI analysis and connects extrema detection directly to growth decisions.
Uses core structure_net with simplified extrema-driven growth.
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

class DirectExtremaGrowth:
    """
    Direct extrema-to-growth engine with NO MI analysis.
    
    Simple approach:
    1. Detect dead/saturated neurons directly
    2. Apply immediate growth decisions
    3. No complex analysis - just direct action
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.network = None
        self.current_accuracy = 0.0
        
    def load_network(self, checkpoint_path):
        """Load pretrained network."""
        print(f"🔬 Loading network: {checkpoint_path}")
        self.network, metadata = load_model_seed(checkpoint_path, device=self.device)
        self.current_accuracy = metadata.get('accuracy', 0.0)
        print(f"   ✅ Loaded: {metadata['architecture']}, Acc: {self.current_accuracy:.2%}")
        return self.network
    
    def detect_simple_extrema(self, train_loader):
        """Simple extrema detection without MI analysis."""
        print("\n🔍 SIMPLE EXTREMA DETECTION")
        print("=" * 30)
        
        self.network.eval()
        activations = []
        
        # Collect activations from a few batches
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 3:  # Just 3 batches for speed
                    break
                    
                data = data.to(self.device).view(data.size(0), -1)
                
                # Forward pass and collect activations
                x = data
                batch_activations = []
                
                for layer in self.network:
                    if hasattr(layer, 'mask'):  # Sparse layer
                        x = layer(x)
                        batch_activations.append(x.detach())
                    elif isinstance(layer, nn.ReLU):
                        x = layer(x)
                        # Update the last activation with ReLU applied
                        if batch_activations:
                            batch_activations[-1] = x.detach()
                
                activations.append(batch_activations)
        
        # Analyze extrema patterns
        extrema_decisions = []
        
        for layer_idx in range(len(activations[0])):
            # Combine activations across batches
            layer_acts = torch.cat([batch[layer_idx] for batch in activations], dim=0)
            mean_acts = layer_acts.mean(dim=0)
            
            # Simple thresholds
            dead_threshold = 0.01
            saturated_threshold = mean_acts.mean() + 2 * mean_acts.std()
            
            dead_neurons = torch.where(mean_acts < dead_threshold)[0].cpu().numpy().tolist()
            saturated_neurons = torch.where(mean_acts > saturated_threshold)[0].cpu().numpy().tolist()
            
            print(f"📊 Layer {layer_idx}: {len(dead_neurons)} dead, {len(saturated_neurons)} saturated")
            
            # Simple decision rules
            if len(dead_neurons) > 50:
                extrema_decisions.append({
                    'type': 'add_layer',
                    'layer': layer_idx,
                    'reason': f'Large dead zone ({len(dead_neurons)} neurons)',
                    'priority': 'HIGH'
                })
                print(f"   🚨 HIGH: Add layer after {layer_idx} (large dead zone)")
            
            elif len(dead_neurons) > 10 or len(saturated_neurons) > 5:
                extrema_decisions.append({
                    'type': 'enhance_connectivity',
                    'layer': layer_idx,
                    'dead_neurons': dead_neurons,
                    'saturated_neurons': saturated_neurons,
                    'reason': f'Moderate extrema ({len(dead_neurons)} dead, {len(saturated_neurons)} saturated)',
                    'priority': 'MEDIUM'
                })
                print(f"   ⚡ MEDIUM: Enhance connectivity in layer {layer_idx}")
        
        return extrema_decisions
    
    def add_layer_after(self, layer_idx, new_size=128):
        """Add a new layer after the specified position."""
        print(f"\n🏗️  ADDING LAYER")
        print(f"   Position: After layer {layer_idx}")
        print(f"   Size: {new_size}")
        
        # Get current architecture
        stats = get_network_stats(self.network)
        current_arch = stats['architecture']
        
        # Insert new layer
        new_arch = current_arch.copy()
        new_arch.insert(layer_idx + 1, new_size)
        
        print(f"   Old: {current_arch}")
        print(f"   New: {new_arch}")
        
        # Create new network
        new_network = create_standard_network(
            architecture=new_arch,
            sparsity=0.02,
            device=self.device
        )
        
        # Transfer weights
        self._transfer_weights_with_insertion(self.network, new_network, layer_idx)
        
        self.network = new_network
        print(f"   ✅ Layer added successfully")
        return True
    
    def enhance_connectivity(self, layer_idx, dead_neurons, saturated_neurons):
        """Enhance connectivity for extrema neurons."""
        print(f"\n⚡ ENHANCING CONNECTIVITY")
        print(f"   Layer: {layer_idx}")
        print(f"   Dead: {len(dead_neurons)}, Saturated: {len(saturated_neurons)}")
        
        sparse_layers = [layer for layer in self.network if hasattr(layer, 'mask')]
        
        if layer_idx < len(sparse_layers):
            target_layer = sparse_layers[layer_idx]
            
            with torch.no_grad():
                # Add connections for dead neurons
                for neuron_idx in dead_neurons[:10]:
                    if neuron_idx < target_layer.mask.shape[0]:
                        # Add random input connections
                        new_connections = torch.rand(target_layer.mask.shape[1]) < 0.1
                        target_layer.mask[neuron_idx, :] = torch.maximum(
                            target_layer.mask[neuron_idx, :],
                            new_connections.float()
                        )
                
                # Add connections for saturated neurons
                for neuron_idx in saturated_neurons[:5]:
                    if neuron_idx < target_layer.mask.shape[1]:
                        # Add random output connections
                        new_connections = torch.rand(target_layer.mask.shape[0]) < 0.1
                        target_layer.mask[:, neuron_idx] = torch.maximum(
                            target_layer.mask[:, neuron_idx],
                            new_connections.float()
                        )
            
            print(f"   ✅ Connectivity enhanced")
            return True
        
        return False
    
    def _transfer_weights_with_insertion(self, old_network, new_network, insert_position):
        """Transfer weights with layer insertion."""
        old_sparse = [layer for layer in old_network if hasattr(layer, 'mask')]
        new_sparse = [layer for layer in new_network if hasattr(layer, 'mask')]
        
        print(f"   🔄 Transferring: {len(old_sparse)} → {len(new_sparse)} layers")
        
        with torch.no_grad():
            new_idx = 0
            for old_idx, old_layer in enumerate(old_sparse):
                if old_idx == insert_position:
                    # Skip the inserted layer
                    new_idx += 1
                
                if new_idx < len(new_sparse):
                    new_layer = new_sparse[new_idx]
                    
                    # Copy compatible dimensions
                    min_out = min(old_layer.linear.weight.shape[0], new_layer.linear.weight.shape[0])
                    min_in = min(old_layer.linear.weight.shape[1], new_layer.linear.weight.shape[1])
                    
                    new_layer.linear.weight.data[:min_out, :min_in] = old_layer.linear.weight.data[:min_out, :min_in]
                    new_layer.linear.bias.data[:min_out] = old_layer.linear.bias.data[:min_out]
                    new_layer.mask[:min_out, :min_in] = old_layer.mask[:min_out, :min_in]
                    
                    print(f"      {old_idx} → {new_idx}: {min_out}x{min_in}")
                    new_idx += 1
    
    def apply_growth_decision(self, decision):
        """Apply a growth decision."""
        if decision['type'] == 'add_layer':
            return self.add_layer_after(decision['layer'])
        elif decision['type'] == 'enhance_connectivity':
            return self.enhance_connectivity(
                decision['layer'],
                decision.get('dead_neurons', []),
                decision.get('saturated_neurons', [])
            )
        return False
    
    def direct_growth_step(self, train_loader):
        """Perform direct extrema-driven growth."""
        print("\n🧬 DIRECT GROWTH STEP")
        print("=" * 40)
        
        # Detect extrema
        decisions = self.detect_simple_extrema(train_loader)
        
        if not decisions:
            print("✅ No growth needed")
            return False
        
        # Apply highest priority decision
        top_decision = decisions[0]
        print(f"\n🎯 Applying: {top_decision['type']}")
        print(f"   Reason: {top_decision['reason']}")
        
        growth_occurred = self.apply_growth_decision(top_decision)
        
        if growth_occurred:
            sort_all_network_layers(self.network)
            print("   🔄 Applied neuron sorting")
        
        return growth_occurred
    
    def evaluate_network(self, test_loader):
        """Evaluate network."""
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
        print(f"📊 Accuracy: {accuracy:.2%}")
        return accuracy
    
    def train_network(self, train_loader, test_loader, epochs=5):
        """Train network."""
        print(f"🚀 Training for {epochs} epochs")
        
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.network.train()
            total_loss = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)
                
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
    
    def run_experiment(self, train_loader, test_loader, max_iterations=3):
        """Run direct extrema growth experiment."""
        print("🔬 DIRECT EXTREMA GROWTH EXPERIMENT")
        print("=" * 50)
        print("🎯 NO MI ANALYSIS - Direct extrema → growth")
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n🔄 Iteration {iteration}/{max_iterations}")
            
            # Train
            accuracy = self.train_network(train_loader, test_loader)
            
            # Grow
            growth_occurred = self.direct_growth_step(train_loader)
            
            print(f"📊 Iteration {iteration}: Acc {accuracy:.2%}, Growth: {'Yes' if growth_occurred else 'No'}")
            
            # Save
            if accuracy > 0.35:
                # Manually construct the correct architecture
                sparse_layers = [layer for layer in self.network if hasattr(layer, 'mask')]
                print(f"   🔧 Debug: Found {len(sparse_layers)} sparse layers")
                
                # Build architecture from actual layer dimensions
                architecture = []
                for i, layer in enumerate(sparse_layers):
                    if i == 0:
                        architecture.append(layer.linear.in_features)
                    architecture.append(layer.linear.out_features)
                
                print(f"   🔧 Debug: Constructed architecture {architecture}")
                
                save_model_seed(
                    model=self.network,
                    architecture=architecture,
                    seed=42,
                    metrics={'accuracy': accuracy, 'iteration': iteration},
                    filepath=f"data/direct_extrema_iter{iteration}_acc{accuracy:.2f}.pt"
                )
                print(f"   💾 Saved checkpoint")
        
        print(f"\n✅ Experiment complete! Final: {self.current_accuracy:.2%}")

def load_cifar10_data(batch_size=64):
    """Load CIFAR-10."""
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
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description='Direct Extrema Growth')
    parser.add_argument('--load-model', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    
    train_loader, test_loader = load_cifar10_data()
    
    growth_engine = DirectExtremaGrowth(device=device)
    growth_engine.load_network(args.load_model)
    growth_engine.run_experiment(train_loader, test_loader)

if __name__ == "__main__":
    main()
