#!/usr/bin/env python3
"""
Indefinite Growth Experiment

Implements the indefinite growth hypothesis: keep adding sparse→dense layers 
until we hit target accuracy by following extrema signals.
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
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class IterativeGrowthNetwork(nn.Module):
    """Network that grows itself to target accuracy using extrema guidance"""
    
    def __init__(self, seed_architecture=[784, 128, 10], scaffold_sparsity=0.02, device='cuda'):
        super().__init__()
        self.device = device
        self.scaffold_sparsity = scaffold_sparsity
        self.growth_history = []
        
        # Start with sparse seed
        self.architecture = seed_architecture
        self.scaffold = self._create_sparse_scaffold(seed_architecture, scaffold_sparsity)
        
        # Dense patches
        self.patches = nn.ModuleDict()
        self.patch_connections = {}
        
        # Growth tracking
        self.activations = []
        self.current_accuracy = 0.0
        
    def _create_sparse_scaffold(self, architecture, sparsity):
        """Create sparse scaffold layers"""
        layers = nn.ModuleList()
        
        for i in range(len(architecture) - 1):
            in_features = architecture[i]
            out_features = architecture[i + 1]
            
            layer = nn.Linear(in_features, out_features)
            
            # Apply sparsity mask
            num_connections = int(sparsity * in_features * out_features)
            mask = torch.zeros(out_features, in_features)
            indices = torch.randperm(in_features * out_features)[:num_connections]
            mask.view(-1)[indices] = 1.0
            
            layer.register_buffer('mask', mask)
            with torch.no_grad():
                layer.weight.mul_(mask)
            
            layers.append(layer.to(self.device))
            
        return layers
    
    def forward(self, x):
        """Forward pass through growing network"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        self.activations = []
        h = x
        
        for i, layer in enumerate(self.scaffold):
            # Sparse computation
            sparse_out = F.linear(h, layer.weight * layer.mask, layer.bias)
            
            # Add patch contributions
            patch_out = self._compute_patch_contributions(h, i, sparse_out)
            h = sparse_out + patch_out
            
            # Store for extrema detection
            self.activations.append(h.detach())
            
            # Apply activation (except last layer)
            if i < len(self.scaffold) - 1:
                h = F.relu(h)
        
        return h
    
    def _compute_patch_contributions(self, input_h, layer_idx, sparse_out):
        """Compute dense patch contributions"""
        batch_size = input_h.size(0)
        output_size = sparse_out.size(1)
        
        patch_contribution = torch.zeros(batch_size, output_size, device=self.device)
        
        for patch_name, patch_info in self.patch_connections.items():
            if patch_info['target_layer'] == layer_idx:
                patch = self.patches[patch_name]
                
                if patch_info['type'] == 'high_extrema':
                    source_neuron = patch_info['source_neuron']
                    if (layer_idx > 0 and 
                        len(self.activations) > layer_idx - 1 and 
                        source_neuron < self.activations[layer_idx-1].size(1)):
                        
                        neuron_act = self.activations[layer_idx-1][:, source_neuron:source_neuron+1]
                        patch_out = patch(neuron_act)
                        
                        target_neurons = patch_info['target_neurons']
                        for j, target in enumerate(target_neurons):
                            if j < patch_out.size(1) and target < output_size:
                                patch_contribution[:, target] += patch_out[:, j] * 0.3
                
                elif patch_info['type'] == 'low_extrema':
                    target_neuron = patch_info['target_neuron']
                    if target_neuron < output_size and layer_idx > 0:
                        prev_size = input_h.size(1)
                        sample_size = min(20, prev_size)
                        sample_indices = torch.randperm(prev_size, device=self.device)[:sample_size]
                        sampled_input = input_h[:, sample_indices]
                        
                        patch_out = patch(sampled_input)
                        patch_contribution[:, target_neuron] += patch_out.squeeze(-1) * 0.5
        
        return patch_contribution
    
    def detect_extrema(self):
        """Detect extrema for growth guidance"""
        extrema = {'high': {}, 'low': {}}
        
        if not self.activations:
            return extrema
        
        for i, activations in enumerate(self.activations[:-1]):  # Skip output layer
            mean_acts = activations.mean(dim=0)
            
            # High extrema (saturated neurons)
            high_threshold = mean_acts.mean() + 2 * mean_acts.std()
            high_indices = torch.where(mean_acts > high_threshold)[0]
            if len(high_indices) > 0:
                extrema['high'][i] = high_indices.cpu().numpy().tolist()
            
            # Low extrema (dead neurons)
            low_threshold = 0.1
            low_indices = torch.where(mean_acts < low_threshold)[0]
            if len(low_indices) > 0:
                extrema['low'][i] = low_indices.cpu().numpy().tolist()
        
        return extrema
    
    def should_add_layer(self, extrema):
        """Decide if we need a new sparse layer"""
        total_extrema = sum(len(indices) for indices in extrema['high'].values()) + \
                       sum(len(indices) for indices in extrema['low'].values())
        
        # If >50% of neurons are extrema, we need more capacity
        total_neurons = sum(self.architecture[1:-1])  # Hidden layers only
        extrema_ratio = total_extrema / max(total_neurons, 1)
        
        return extrema_ratio > 0.5
    
    def add_sparse_layer_at_bottleneck(self, extrema):
        """Add new sparse layer where bottleneck is worst"""
        # Find layer with most extrema
        extrema_counts = {}
        for layer_idx, indices in extrema['high'].items():
            extrema_counts[layer_idx] = extrema_counts.get(layer_idx, 0) + len(indices)
        for layer_idx, indices in extrema['low'].items():
            extrema_counts[layer_idx] = extrema_counts.get(layer_idx, 0) + len(indices)
        
        if not extrema_counts:
            return
        
        worst_layer = max(extrema_counts.keys(), key=lambda k: extrema_counts[k])
        
        # Calculate new layer size (4x expansion of extrema)
        new_layer_size = min(extrema_counts[worst_layer] * 4, 256)
        
        # Insert new layer after worst bottleneck
        insert_pos = worst_layer + 1
        
        # Update architecture
        new_arch = self.architecture[:insert_pos+1] + [new_layer_size] + self.architecture[insert_pos+1:]
        
        # Create new scaffold
        new_scaffold = self._create_sparse_scaffold(new_arch, self.scaffold_sparsity)
        
        # Copy weights from old scaffold with proper dimension handling
        for i, old_layer in enumerate(self.scaffold):
            if i <= worst_layer:
                # Copy directly - dimensions should match
                if (new_scaffold[i].weight.shape == old_layer.weight.shape and
                    new_scaffold[i].bias.shape == old_layer.bias.shape):
                    new_scaffold[i].weight.data.copy_(old_layer.weight.data)
                    new_scaffold[i].bias.data.copy_(old_layer.bias.data)
                    new_scaffold[i].mask.copy_(old_layer.mask)
                else:
                    print(f"    ⚠️  Dimension mismatch at layer {i}, reinitializing")
            elif i > worst_layer:
                # Shift by one position - need to handle dimension changes
                new_idx = i + 1
                if new_idx < len(new_scaffold):
                    # Check if we need to handle the connection to the new layer
                    if i == worst_layer + 1:
                        # This layer now connects FROM the new layer, so input size changed
                        # We need to reinitialize this layer since input dimensions changed
                        print(f"    🔄 Reinitializing layer {new_idx} due to new input dimensions")
                    else:
                        # Normal copy if dimensions match
                        if (new_scaffold[new_idx].weight.shape == old_layer.weight.shape and
                            new_scaffold[new_idx].bias.shape == old_layer.bias.shape):
                            new_scaffold[new_idx].weight.data.copy_(old_layer.weight.data)
                            new_scaffold[new_idx].bias.data.copy_(old_layer.bias.data)
                            new_scaffold[new_idx].mask.copy_(old_layer.mask)
                        else:
                            print(f"    ⚠️  Dimension mismatch at layer {new_idx}, reinitializing")
        
        # Update network
        self.architecture = new_arch
        self.scaffold = new_scaffold
        
        print(f"  🌱 Added sparse layer with {new_layer_size} neurons after layer {worst_layer}")
        print(f"  📐 New architecture: {self.architecture}")
    
    def add_targeted_patches(self, extrema):
        """Add dense patches around extrema"""
        patches_created = 0
        
        # High extrema patches
        for layer_idx, neuron_indices in extrema.get('high', {}).items():
            if layer_idx + 1 < len(self.scaffold):
                next_layer_size = self.scaffold[layer_idx + 1].out_features
            else:
                next_layer_size = self.architecture[-1]
            
            for j, neuron_idx in enumerate(neuron_indices[:5]):  # Limit patches
                patch_name = f"high_{layer_idx}_{neuron_idx}_{len(self.patches)}"
                
                patch = nn.Sequential(
                    nn.Linear(1, 8),
                    nn.ReLU(),
                    nn.Linear(8, min(4, next_layer_size))
                ).to(self.device)
                
                self.patches[patch_name] = patch
                
                target_layer_idx = min(layer_idx + 1, len(self.scaffold) - 1)
                target_neurons = list(range(min(4, next_layer_size)))
                self.patch_connections[patch_name] = {
                    'type': 'high_extrema',
                    'source_layer': layer_idx,
                    'source_neuron': int(neuron_idx),
                    'target_layer': target_layer_idx,
                    'target_neurons': target_neurons
                }
                
                patches_created += 1
        
        # Low extrema patches
        for layer_idx, neuron_indices in extrema.get('low', {}).items():
            if layer_idx > 0:
                for j, neuron_idx in enumerate(neuron_indices[:5]):
                    patch_name = f"low_{layer_idx}_{neuron_idx}_{len(self.patches)}"
                    
                    patch = nn.Sequential(
                        nn.Linear(20, 8),
                        nn.ReLU(),
                        nn.Linear(8, 1)
                    ).to(self.device)
                    
                    self.patches[patch_name] = patch
                    
                    self.patch_connections[patch_name] = {
                        'type': 'low_extrema',
                        'source_layer': layer_idx - 1,
                        'target_layer': layer_idx,
                        'target_neuron': int(neuron_idx)
                    }
                    
                    patches_created += 1
        
        print(f"  🔧 Added {patches_created} dense patches")
        return patches_created
    
    def train_to_convergence(self, train_loader, test_loader, max_epochs=20):
        """Train current network to convergence"""
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        patience = 5
        no_improve = 0
        
        for epoch in range(max_epochs):
            # Training
            self.train()
            total_loss = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Evaluation
            self.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            accuracy = correct / len(test_loader.dataset)
            
            if accuracy > best_acc:
                best_acc = accuracy
                no_improve = 0
            else:
                no_improve += 1
            
            if epoch % 5 == 0:
                print(f"    Epoch {epoch}: {accuracy:.2%} (best: {best_acc:.2%})")
            
            # Early stopping
            if no_improve >= patience:
                break
        
        self.current_accuracy = best_acc
        return best_acc
    
    def grow_until_target_accuracy(self, target_acc=0.95, train_loader=None, test_loader=None):
        """Main growth loop - keep growing until target accuracy"""
        
        print(f"🎯 Growing network until {target_acc:.1%} accuracy")
        print(f"🌱 Starting with: {self.architecture}")
        
        iteration = 0
        
        while self.current_accuracy < target_acc and iteration < 10:  # Safety limit
            iteration += 1
            print(f"\n🌱 Growth Iteration {iteration}")
            
            # Step 1: Train current network to convergence
            print("  📚 Training to convergence...")
            current_acc = self.train_to_convergence(train_loader, test_loader)
            print(f"  📊 Current accuracy: {current_acc:.2%}")
            
            if current_acc >= target_acc:
                print(f"🎉 Target accuracy {target_acc:.1%} achieved!")
                break
            
            # Step 2: Analyze network with test data
            print("  🔍 Analyzing extrema...")
            self.eval()
            with torch.no_grad():
                for i, (data, _) in enumerate(test_loader):
                    if i >= 3:  # Use first 3 batches
                        break
                    _ = self(data.to(self.device))
            
            extrema = self.detect_extrema()
            total_high = sum(len(indices) for indices in extrema['high'].values())
            total_low = sum(len(indices) for indices in extrema['low'].values())
            print(f"  🔍 Found {total_high} high, {total_low} low extrema")
            
            # Step 3: Decide growth strategy
            if self.should_add_layer(extrema):
                print("  🏗️  Adding new sparse layer...")
                self.add_sparse_layer_at_bottleneck(extrema)
            
            # Step 4: Always add patches
            print("  🔧 Adding dense patches...")
            patches_added = self.add_targeted_patches(extrema)
            
            # Record growth event
            self.growth_history.append({
                'iteration': iteration,
                'architecture': self.architecture.copy(),
                'accuracy': current_acc,
                'layers': len(self.scaffold),
                'patches': len(self.patches),
                'extrema_high': total_high,
                'extrema_low': total_low,
                'total_params': sum(p.numel() for p in self.parameters())
            })
            
            print(f"  📈 Network stats: {len(self.scaffold)} layers, {len(self.patches)} patches")
        
        print(f"\n🏁 Growth completed after {iteration} iterations")
        print(f"📊 Final accuracy: {self.current_accuracy:.2%}")
        print(f"📐 Final architecture: {self.architecture}")
        print(f"🔧 Total patches: {len(self.patches)}")
        
        return self.current_accuracy, self.growth_history

def load_mnist_data(batch_size=64):
    """Load MNIST dataset"""
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
    """Run indefinite growth experiment"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_mnist_data()
    
    # Create growing network
    network = IterativeGrowthNetwork(
        seed_architecture=[784, 128, 10],
        scaffold_sparsity=0.02,
        device=device
    )
    
    # Grow until target accuracy
    final_acc, history = network.grow_until_target_accuracy(
        target_acc=0.95,
        train_loader=train_loader,
        test_loader=test_loader
    )
    
    # Print growth summary
    print("\n" + "="*60)
    print("📈 GROWTH SUMMARY")
    print("="*60)
    
    for event in history:
        print(f"Iteration {event['iteration']}: "
              f"{event['architecture']} → {event['accuracy']:.2%} "
              f"({event['patches']} patches)")
    
    print(f"\n🎯 Target: 95.0%")
    print(f"🏆 Achieved: {final_acc:.2%}")
    print(f"🌱 Growth iterations: {len(history)}")
    
    # Calculate efficiency
    final_params = history[-1]['total_params'] if history else sum(p.numel() for p in network.parameters())
    dense_params = 784 * 128 + 128 * 10  # Equivalent dense network
    sparsity = 1 - (final_params / dense_params)
    
    print(f"📊 Final parameters: {final_params:,}")
    print(f"💾 Equivalent dense: {dense_params:,}")
    print(f"⚡ Effective sparsity: {sparsity:.1%}")
    
    # Save results
    results = {
        'final_accuracy': final_acc,
        'growth_history': history,
        'final_architecture': network.architecture,
        'total_parameters': final_params,
        'effective_sparsity': sparsity
    }
    
    os.makedirs('data/growth_results', exist_ok=True)
    with open('data/growth_results/indefinite_growth_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to data/growth_results/indefinite_growth_results.json")

if __name__ == "__main__":
    main()
