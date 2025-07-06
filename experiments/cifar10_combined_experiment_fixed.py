#!/usr/bin/env python3
"""
CIFAR-10 Combined Experiment - FIXED VERSION

This script combines the Optimal Seed Finder and Patched Density experiments,
and adapts them for the CIFAR-10 dataset.

BUGS FIXED:
1. Activation storage timing bug - activations stored before ReLU activation
2. Index mismatch in patch contribution computation
3. Missing activation clearing between forward passes
4. Incorrect layer indexing in extrema detection
5. Statistical threshold calculation issues
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
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.structure_net.core.minimal_network import MinimalNetwork

def count_parameters(network):
    """Count the number of active parameters in a sparse network."""
    total_params = 0
    if hasattr(network, 'connection_masks'):
        for mask in network.connection_masks:
            total_params += mask.sum().item()
    else:
        for param in network.parameters():
            total_params += param.numel()
    return total_params

def load_cifar10_data(batch_size=64, is_worker=False):
    """Load the CIFAR-10 dataset."""
    if not is_worker:
        print("📦 Loading CIFAR-10 dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    num_workers = 0 if is_worker else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    if not is_worker:
        print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples.")
    return train_loader, test_loader

def train_and_evaluate_arch_cifar(args):
    """Worker function for parallel training on CIFAR-10."""
    arch, device_id, batch_size = args
    device = torch.device(f"cuda:{device_id}")
    
    print(f"  [GPU {device_id}] Testing architecture: {arch}")

    train_loader, test_loader = load_cifar10_data(batch_size=batch_size, is_worker=True)

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
    
    learns = best_test_acc > 0.15 # CIFAR-10 has 10 classes, so >10% is learning
    result = {
        'arch': arch,
        'accuracy': best_test_acc,
        'parameters': count_parameters(network),
        'learns': learns
    }
    print(f"  [GPU {device_id}] Accuracy: {best_test_acc:.2%}, Learns: {learns}")
    return result

class OptimalSeedFinder:
    """Find the smallest viable network to bootstrap from for CIFAR-10."""

    def find_optimal_seed(self):
        """Find minimal architecture that still learns in parallel."""
        print("🔍 Finding optimal small seed network for CIFAR-10 in parallel...")
        
        input_size = 32 * 32 * 3
        architectures = [
            [input_size, 256, 10],
            [input_size, 128, 10],
            [input_size, 64, 10],
            [input_size, 32, 10],
            [input_size, 10],
        ]
        
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs.")

        args_list = [(arch, i % num_gpus, 64) for i, arch in enumerate(architectures)]

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_gpus) as pool:
            worker_results = pool.map(train_and_evaluate_arch_cifar, args_list)

        results = {str(res['arch']): res for res in worker_results}

        learning_archs = [a for a in results if results[a]['learns']]
        if not learning_archs:
            print("⚠️ No architecture learned successfully.")
            return None, results

        optimal_arch_str = min(learning_archs, key=lambda a: results[a]['parameters'])
        optimal_arch = json.loads(optimal_arch_str)
        
        print(f"✅ Found optimal seed for CIFAR-10: {optimal_arch}")
        return optimal_arch, results

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

class PatchedDensityNetwork(nn.Module):
    """Patched density network for CIFAR-10."""
    
    def __init__(self, architecture=[3072, 256, 128, 10], scaffold_sparsity=0.02, device='cuda'):
        super().__init__()
        self.device = device
        self.architecture = architecture
        
        # Create sparse scaffold
        self.scaffold = self._create_sparse_scaffold(architecture, scaffold_sparsity)
        
        # These will be our dense patches
        self.patches = nn.ModuleDict()
        self.patch_connections = {}  # Track where patches connect
        
        # FIXED: Initialize activations storage
        self.activations = []
        
    def _create_sparse_scaffold(self, architecture, sparsity):
        """Create the initial sparse network"""
        layers = nn.ModuleList()
        
        for i in range(len(architecture) - 1):
            in_features = architecture[i]
            out_features = architecture[i + 1]
            
            # Create sparse layer
            layer = nn.Linear(in_features, out_features)
            
            # Apply sparsity mask
            num_connections = int(sparsity * in_features * out_features)
            mask = torch.zeros(out_features, in_features)
            
            # Random sparse connections
            indices = torch.randperm(in_features * out_features)[:num_connections]
            mask.view(-1)[indices] = 1.0
            
            # Register mask as buffer
            layer.register_buffer('mask', mask)
            
            # Apply mask to weights
            with torch.no_grad():
                layer.weight.mul_(mask)
            
            layers.append(layer.to(self.device))
            
        return layers
    
    def forward(self, x):
        """Forward pass through scaffold + patches"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # FIXED: Clear previous activations at start of forward pass
        self.activations = []
        h = x
        
        for i, layer in enumerate(self.scaffold):
            # Apply sparse weights
            sparse_out = F.linear(h, layer.weight * layer.mask, layer.bias)
            
            # Add patch contributions for this layer
            patch_out = self._compute_patch_contributions(h, i, sparse_out)
            
            # Combine sparse and patch outputs
            h = sparse_out + patch_out
            
            # CRITICAL FIX: Store activation BEFORE applying ReLU for extrema detection
            # This matches the original working version
            self.activations.append(h.detach())
            
            # Apply activation (except last layer)
            if i < len(self.scaffold) - 1:
                h = F.relu(h)
        
        return h

    def _compute_patch_contributions(self, input_h, layer_idx, sparse_out):
        """Compute contributions from all patches at this layer"""
        batch_size = input_h.size(0)
        output_size = sparse_out.size(1)
        
        patch_contribution = torch.zeros(batch_size, output_size, device=self.device)
        
        # Process patches that affect this layer
        for patch_name, patch_info in self.patch_connections.items():
            if patch_info['target_layer'] == layer_idx:
                patch = self.patches[patch_name]
                
                if patch_info['type'] == 'high_extrema':
                    # High extrema patch: takes saturated neuron, creates alternative paths
                    source_neuron = patch_info['source_neuron']
                    # FIXED: Check bounds and ensure we have previous activations
                    if (layer_idx > 0 and 
                        len(self.activations) > layer_idx - 1 and 
                        source_neuron < self.activations[layer_idx-1].size(1)):
                        
                        # Get the saturated neuron's activation from previous layer
                        neuron_act = self.activations[layer_idx-1][:, source_neuron:source_neuron+1]
                        
                        # Process through patch
                        patch_out = patch(neuron_act)
                        
                        # Add to specific output neurons
                        target_neurons = patch_info['target_neurons']
                        for j, target in enumerate(target_neurons):
                            if j < patch_out.size(1) and target < output_size:
                                patch_contribution[:, target] += patch_out[:, j] * 0.3
                
                elif patch_info['type'] == 'low_extrema':
                    # Low extrema patch: provides additional input to dead neurons
                    target_neuron = patch_info['target_neuron']
                    if target_neuron < output_size:
                        # Take broader input from previous layer
                        if layer_idx > 0:
                            prev_size = input_h.size(1)
                            sample_size = min(20, prev_size)
                            sample_indices = torch.randperm(prev_size, device=self.device)[:sample_size]
                            sampled_input = input_h[:, sample_indices]
                            
                            # Process through patch
                            patch_out = patch(sampled_input)
                            
                            # Add to dead neuron
                            patch_contribution[:, target_neuron] += patch_out.squeeze(-1) * 0.5
        
        return patch_contribution
    
    def detect_extrema(self):
        """Detect extrema in the network"""
        extrema = {'high': {}, 'low': {}}
        
        if not hasattr(self, 'activations') or not self.activations:
            print("⚠️  Activations not found. Run a forward pass before detecting extrema.")
            return extrema

        for i, activations in enumerate(self.activations[:-1]):  # Skip output layer
            # Average across batch
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
        
        # Add debug information
        total_high = sum(len(indices) for indices in extrema['high'].values())
        total_low = sum(len(indices) for indices in extrema['low'].values())
        print(f"🔍 Detected extrema: {total_high} high, {total_low} low across {len(self.activations)} layers")
        
        return extrema
    
    def create_dense_patches(self, extrema, patch_density=0.5):
        """Create dense patches around extrema - FIXED VERSION"""
        patches_created = 0
        
        # Create patches for high extrema
        for layer_idx, neuron_indices in extrema.get('high', {}).items():
            # FIXED: Proper bounds checking for next layer
            if layer_idx + 1 < len(self.scaffold):
                next_layer_size = self.scaffold[layer_idx + 1].out_features
            else:
                next_layer_size = self.architecture[-1]  # Output layer size
            
            for j, neuron_idx in enumerate(neuron_indices[:5]):  # Limit to 5 patches per layer
                patch_name = f"high_{layer_idx}_{neuron_idx}"
                
                # Create a small dense network patch
                patch = nn.Sequential(
                    nn.Linear(1, 8),
                    nn.ReLU(),
                    nn.Linear(8, min(4, next_layer_size))
                ).to(self.device)
                
                self.patches[patch_name] = patch
                
                # Track connections - FIXED: Ensure target layer index is valid
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
        
        # Create patches for low extrema
        for layer_idx, neuron_indices in extrema.get('low', {}).items():
            # FIXED: Only create patches for layers that aren't the first layer
            if layer_idx > 0:
                for j, neuron_idx in enumerate(neuron_indices[:5]):
                    patch_name = f"low_{layer_idx}_{neuron_idx}"
                    
                    # Create input patch for dead neuron
                    patch = nn.Sequential(
                        nn.Linear(20, 8),  # Sample 20 inputs
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
        
        return patches_created
    
    def analyze_patch_effectiveness(self):
        """Analyze how effective patches are"""
        stats = {
            'active_count': 0,
            'avg_contribution': 0.0,
            'weight_norms': {}
        }
        
        for name, patch in self.patches.items():
            weight_norm = sum(p.norm().item() for p in patch.parameters())
            stats['weight_norms'][name] = weight_norm
            
            if weight_norm > 0.1:  # Active threshold
                stats['active_count'] += 1
                stats['avg_contribution'] += weight_norm
        
        if stats['active_count'] > 0:
            stats['avg_contribution'] /= stats['active_count']
            
        return stats

def main():
    """Main entry point for the script."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Running on CPU will be very slow.")

    save_dir = "data/cifar10_combined_results"
    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Use a multi-layer architecture that can have extrema
    # The original [3072, 10] has no hidden layers, so no extrema can be detected
    optimal_seed_arch = [3072, 64, 10]  # Add a hidden layer for extrema detection
    print(f"✅ Using multi-layer seed for CIFAR-10: {optimal_seed_arch}")

    if optimal_seed_arch:
        # Step 2: Run the patched density experiment
        print("\n" + "="*60)
        print("🔬 Running Patched Density Experiment for CIFAR-10")
        print("="*60)

        patched_network = PatchedDensityNetwork(architecture=optimal_seed_arch, device=device)
        
        # Phase 1: Train sparse scaffold
        train_loader, test_loader = load_cifar10_data()
        print("\n🏗️  Phase 1: Training sparse scaffold")
        optimizer = optim.Adam(patched_network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(20):
            train_epoch(patched_network, train_loader, optimizer, criterion, device)
            test_acc = evaluate(patched_network, test_loader, device)
            print(f"  Epoch {epoch+1}/20, Scaffold Test Acc: {test_acc:.2f}%")

        # Detect extrema - FIXED: Use multiple batches for better statistics
        print("\nDetecting extrema on test set...")
        print(f"Network architecture: {optimal_seed_arch}")
        print(f"Number of layers in scaffold: {len(patched_network.scaffold)}")
        
        patched_network.eval()
        all_activations = []
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                if i >= 5:  # Use first 5 batches for extrema detection
                    break
                _ = patched_network(data.to(device))
                print(f"Batch {i}: Got {len(patched_network.activations)} activations")
                for j, act in enumerate(patched_network.activations):
                    print(f"  Layer {j}: shape {act.shape}, mean {act.mean().item():.4f}, std {act.std().item():.4f}")
                
                # Accumulate activations across batches
                if i == 0:
                    all_activations = [act.clone() for act in patched_network.activations]
                else:
                    for j, act in enumerate(patched_network.activations):
                        all_activations[j] = torch.cat([all_activations[j], act], dim=0)
        
        # Set accumulated activations for extrema detection
        patched_network.activations = all_activations
        print(f"\nFinal accumulated activations: {len(all_activations)} layers")
        for j, act in enumerate(all_activations):
            print(f"  Layer {j}: shape {act.shape}, mean {act.mean().item():.4f}, std {act.std().item():.4f}")
        
        extrema = patched_network.detect_extrema()
        
        # Phase 2: Create dense patches
        print(f"\n🔧 Phase 2: Creating dense patches")
        patches_created = patched_network.create_dense_patches(extrema)
        print(f"✅ Created {patches_created} dense patches")

        # Phase 3: Train with dual learning rates
        print(f"\n🎯 Phase 3: Training with dual learning rates")
        param_groups = [
            {'params': patched_network.scaffold.parameters(), 'lr': 0.0001},
            {'params': patched_network.patches.parameters(), 'lr': 0.0005}
        ]
        optimizer = optim.Adam(param_groups)
        
        for epoch in range(30):
            train_epoch(patched_network, train_loader, optimizer, criterion, device)
            test_acc = evaluate(patched_network, test_loader, device)
            print(f"  Epoch {epoch+1}/30, Patched Test Acc: {test_acc:.2f}%")

    print("\n🎉 All CIFAR-10 experiments completed.")

if __name__ == "__main__":
    main()
