#!/usr/bin/env python3
"""
Hybrid Growth Experiment

This experiment implements a network that combines a sparse scaffold with 
multi-scale dense patches, and grows based on extrema patterns.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Advanced Architecture Components ---

class ExtremaAwareSparseLayer(nn.Module):
    """A sparse linear layer where connectivity density is guided by extrema.
    
    This is the core component for "embedding" patches. It has a base low
    density everywhere, but much higher density in regions that correspond
    to reviving dead neurons or providing relief for saturated ones.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 base_sparsity: float = 0.02, 
                 extrema_to_embed: Dict[str, List[int]] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_sparsity = base_sparsity

        # Create the variable-density mask
        mask = self._create_extrema_aware_mask(extrema_to_embed)
        self.register_buffer('mask', mask)

        # Create the single weight and bias matrix
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def _create_extrema_aware_mask(self, extrema_to_embed: Dict[str, List[int]]) -> torch.Tensor:
        """Creates a mask with high density in specified regions."""
        # Start with a base uniform sparse mask
        base_mask = (torch.rand(self.out_features, self.in_features) < self.base_sparsity).float()
        
        if not extrema_to_embed:
            return base_mask

        # Create denser regions for embedding
        patch_mask = torch.zeros_like(base_mask)

        # 1. High-density connections TO revive dead neurons (targets)
        # These neurons need more diverse inputs to become active.
        dead_neurons = extrema_to_embed.get('low', [])
        if dead_neurons:
            # For each dead neuron, give it a 20% dense input connection field
            revival_density = 0.20
            for neuron_idx in dead_neurons:
                if neuron_idx < self.out_features:
                    patch_mask[neuron_idx, :] = (torch.rand(self.in_features) < revival_density).float()

        # 2. High-density connections FROM saturated neurons (sources)
        # These neurons need more outlets to relieve information pressure.
        saturated_neurons = extrema_to_embed.get('high', [])
        if saturated_neurons:
            # For each saturated neuron, give it a 15% dense output connection field
            relief_density = 0.15
            for neuron_idx in saturated_neurons:
                if neuron_idx < self.in_features:
                    patch_mask[:, neuron_idx] = (torch.rand(self.out_features) < relief_density).float()

        # Combine the masks: take the union of all connections
        return torch.maximum(base_mask, patch_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A single, efficient sparse matrix multiplication
        return F.linear(x, self.weight * self.mask, self.bias)

class TemporaryPatch(nn.Module):
    """A small, dense module to fix a single, severe extremum.
    
    Architecture: 1 -> 8 -> 4. It takes input from one source neuron
    and produces a 4-dimensional correction vector.
    """
    def __init__(self, source_neuron_idx: int):
        super().__init__()
        self.source_neuron_idx = source_neuron_idx
        self.layer1 = nn.Linear(1, 8)
        self.layer2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract the single neuron's activation for the whole batch
        patch_input = x[:, self.source_neuron_idx].unsqueeze(-1)
        h = F.relu(self.layer1(patch_input))
        return self.layer2(h)

# --- Analysis Functions ---

@torch.no_grad()
def analyze_layer_extrema(
    activations: torch.Tensor, 
    dead_threshold: float = 0.01, 
    saturated_threshold: float = 0.99
) -> Dict[str, List[int]]:
    """Identifies dead and saturated neurons from their activations."""
    mean_activations = activations.mean(dim=0)
    
    dead_neurons = torch.where(mean_activations < dead_threshold)[0].tolist()
    saturated_neurons = torch.where(mean_activations > saturated_threshold)[0].tolist()
    
    return {'low': dead_neurons, 'high': saturated_neurons}

def estimate_mi_sparse(x: torch.Tensor, y: torch.Tensor) -> float:
    """Placeholder for a mutual information estimator."""
    # A real implementation would use a k-NN estimator like Kraskov (KSG)
    # or a binning-based method, but that's complex. We'll simulate it.
    # The MI should be related to the log of the number of active dimensions.
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    active_dims_x = x.shape[1]
    active_dims_y = y.shape[1]
    # Simple approximation: MI is limited by the bottleneck dimension.
    return np.log2(min(active_dims_x, active_dims_y) + 1)

def _create_sparse_mask(shape, sparsity):
    """Creates a sparse mask with a given shape and sparsity."""
    rows, cols = shape
    num_connections = int(sparsity * rows * cols)
    mask = torch.zeros(rows, cols)
    indices = torch.randperm(rows * cols)[:num_connections]
    mask.view(-1)[indices] = 1.0
    return mask

def _lsuv_init_layer(layer, sample_input):
    """LSUV for a single layer."""
    with torch.no_grad():
        for _ in range(10):
            out = F.linear(sample_input, layer.weight * layer.mask, layer.bias)
            var = out.var()
            if var > 0:
                layer.weight.data /= torch.sqrt(var)
            if abs(out.var() - 1.0) < 0.01:
                break
        return F.relu(out)

def create_lsuv_sparse_scaffold(architecture, sparsity, device, sample_batch, skip_lsuv=False):
    """Creates a sparse scaffold with optional LSUV initialization."""
    layers = nn.ModuleList()
    h = sample_batch
    for i in range(len(architecture) - 1):
        in_features, out_features = architecture[i], architecture[i+1]
        layer = nn.Linear(in_features, out_features).to(device)
        
        mask = _create_sparse_mask((out_features, in_features), sparsity).to(device)
        layer.register_buffer('mask', mask)
        layer.weight.data.mul_(mask)
        
        if not skip_lsuv:
            h = _lsuv_init_layer(layer, h)
        else:
            # For pretrained networks, just pass through without LSUV
            h = F.relu(F.linear(h, layer.weight * layer.mask, layer.bias))
        layers.append(layer)
    return layers

def create_sparse_scaffold_from_pretrained(architecture, sparsity, device, pretrained_state_dict):
    """Creates a sparse scaffold from pretrained weights WITHOUT LSUV."""
    layers = nn.ModuleList()
    
    # Debug: Print all available keys in the pretrained state dict
    print(f"   🔍 Available keys in pretrained checkpoint:")
    for key in sorted(pretrained_state_dict.keys()):
        print(f"      {key}: {pretrained_state_dict[key].shape if hasattr(pretrained_state_dict[key], 'shape') else type(pretrained_state_dict[key])}")
    
    for i in range(len(architecture) - 1):
        in_features, out_features = architecture[i], architecture[i+1]
        layer = nn.Linear(in_features, out_features).to(device)
        
        # Try different possible key patterns for pretrained weights
        # The checkpoint uses nn.Sequential indexing: 0, 2, 4 (with ReLU in between)
        sequential_layer_idx = i * 2  # Map our layer index to sequential index
        possible_keys = [
            f'{sequential_layer_idx}.weight',  # CRITICAL: Sequential pattern (0, 2, 4)
            f'{i}.weight',           # Simple numeric pattern
            f'scaffold.{i}.weight',  # Original attempt
            f'layers.{i}.weight',    # Alternative pattern
            f'sparse_layers.{i}.weight',  # Another pattern
            f'network.{i}.weight',   # Yet another pattern
        ]
        
        loaded = False
        for layer_key in possible_keys:
            if layer_key in pretrained_state_dict:
                layer.weight.data = pretrained_state_dict[layer_key]
                bias_key = layer_key.replace('.weight', '.bias')
                if bias_key in pretrained_state_dict:
                    layer.bias.data = pretrained_state_dict[bias_key]
                
                # CRITICAL FIX: Load the original mask from checkpoint
                # The pretrained model stores the actual mask used during training
                mask_key = layer_key.replace('.weight', '.mask')
                if mask_key in pretrained_state_dict:
                    original_mask = pretrained_state_dict[mask_key]
                    layer.register_buffer('mask', original_mask)
                    print(f"   ✅ Loaded pretrained weights & mask for layer {i} using key '{layer_key}' (preserved {original_mask.sum().item()}/{original_mask.numel()} connections)")
                else:
                    # Fallback: extract mask from non-zero weights
                    original_mask = (layer.weight.data != 0).float()
                    layer.register_buffer('mask', original_mask)
                    print(f"   ✅ Loaded pretrained weights for layer {i} using key '{layer_key}' (extracted mask: {original_mask.sum().item()}/{original_mask.numel()} connections)")
                
                loaded = True
                break
        
        if not loaded:
            # Only create new mask for layers without pretrained weights
            mask = _create_sparse_mask((out_features, in_features), sparsity).to(device)
            layer.register_buffer('mask', mask)
            layer.weight.data.mul_(mask)
            print(f"   🌱 Created new sparse layer {i} with {mask.sum().item()}/{mask.numel()} connections")
        
        layers.append(layer)
    return layers

def lsuv_init_new_layer(new_layer, prev_layer_output, target_variance=1.0):
    """LSUV initialization for a single new layer based on previous layer output."""
    with torch.no_grad():
        # Get statistics from previous layer
        prev_mean = prev_layer_output.mean()
        prev_std = prev_layer_output.std()
        
        # Initialize new layer to handle previous layer statistics
        for _ in range(10):
            out = F.linear(prev_layer_output, new_layer.weight, new_layer.bias)
            var = out.var()
            if var > 0:
                new_layer.weight.data /= torch.sqrt(var / target_variance)
            if abs(out.var() - target_variance) < 0.01:
                break
        return F.relu(out)

# --- Delta-Guided Architecture Evolver ---

class PatchedSparseNetwork(nn.Module):
    """Manages sparse layers and any active temporary patches."""
    def __init__(self, architecture: List[int], base_sparsity: float = 0.02):
        super().__init__()
        self.architecture = architecture
        self.base_sparsity = base_sparsity
        
        # Core sparse layers
        self.sparse_layers = nn.ModuleList()
        for i in range(len(architecture) - 1):
            self.sparse_layers.append(
                ExtremaAwareSparseLayer(architecture[i], architecture[i+1], base_sparsity)
            )
            
        # Temporary patches, organized by the layer they are attached to
        self.patches = nn.ModuleDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.view(x.size(0), -1)
        
        # Store patch corrections for the next layer
        patch_corrections = {}

        for i, layer in enumerate(self.sparse_layers):
            # 1. Apply corrections from patches on the PREVIOUS layer
            if i in patch_corrections:
                h = h + patch_corrections[i]
            
            # 2. Main sparse layer forward pass
            pre_activation = layer(h)
            
            # 3. Apply activation function (except for last layer)
            if i < len(self.sparse_layers) - 1:
                h = F.relu(pre_activation)
            else:
                h = pre_activation  # Last layer outputs logits
            
            # 4. Compute outputs of patches attached to THIS layer's output
            layer_key = str(i)
            if layer_key in self.patches:
                total_patch_output = 0
                for patch in self.patches[layer_key]:
                    total_patch_output += patch(h)
                
                # Project patch output to match next layer's input dimension
                next_layer_idx = i + 1
                if next_layer_idx < len(self.sparse_layers):
                    next_layer_input_dim = self.architecture[next_layer_idx]
                    
                    if total_patch_output.shape[1] != next_layer_input_dim:
                        # Simple projection: tile and slice
                        correction_size = total_patch_output.shape[1]
                        repeats = (next_layer_input_dim + correction_size - 1) // correction_size
                        projected_output = total_patch_output.repeat(1, repeats)[:, :next_layer_input_dim]
                    else:
                        projected_output = total_patch_output
                    
                    # Store for next layer
                    patch_corrections[next_layer_idx] = projected_output
        
        return h

# =================================================================
# ADVANCED EVOLVER (V2.0) - Replaces DeltaGuidedEvolver
# =================================================================

class OptimalGrowthEvolver:
    """
    An advanced evolver that uses information theory to precisely identify
    bottlenecks and calculate the minimal, optimal intervention needed.
    """
    def __init__(self, 
                 seed_arch: List[int],
                 seed_sparsity: float,
                 data_loader: DataLoader, 
                 device: torch.device):
        
        self.network = PatchedSparseNetwork(seed_arch, seed_sparsity).to(device)
        self.data_loader = data_loader
        self.device = device
        self.history = []
        print(f"🚀 Initialized OptimalGrowthEvolver (V2.0) with seed: {seed_arch}, sparsity: {seed_sparsity}")

    def load_pretrained_scaffold(self, state_dict: Dict[str, Any]):
        """Correctly loads weights from a saved nn.Sequential model."""
        print("   ✅ Attempting to load pretrained scaffold weights...")
        for i, layer in enumerate(self.network.sparse_layers):
            layer_key = str(i * 2) 
            weight_key, bias_key = f'{layer_key}.weight', f'{layer_key}.bias'
            if weight_key in state_dict and bias_key in state_dict:
                layer.weight.data = state_dict[weight_key]
                layer.bias.data = state_dict[bias_key]
                mask = (layer.weight.data != 0).float()
                layer.register_buffer('mask', mask)
                print(f"      Layer {i}: Loaded weights and preserved {mask.sum().item():.0f} connections.")
            else:
                print(f"      Layer {i}: Not found in checkpoint. Using random initialization.")

    def _estimate_mi_proxy(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """A fast proxy for Mutual Information based on correlation."""
        if x.numel() == 0 or y.numel() == 0: 
            return 0.0
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        min_dim = min(x_norm.shape[1], y_norm.shape[1])
        correlation = (x_norm[:, :min_dim] * y_norm[:, :min_dim]).sum(dim=1).mean()
        mi_approx = -0.5 * torch.log(1 - correlation**2 + 1e-8)
        return mi_approx.item()

    def analyze_information_flow(self) -> List[Dict[str, Any]]:
        """Finds information bottlenecks by measuring MI loss between layers."""
        print("\n--- Analyzing Information Flow (MI) ---")
        bottlenecks = []
        activations = self.get_layer_activations()
        
        # Calculate MI between each layer transition
        mi_flow = [self._estimate_mi_proxy(activations[i], activations[i+1]) for i in range(len(activations)-1)]
        
        # Calculate information loss at each step
        for i in range(len(mi_flow) - 1):
            info_loss = mi_flow[i] - mi_flow[i+1]
            if info_loss > 0.05: # Only consider non-trivial loss
                bottlenecks.append({
                    'position': i + 1, # Bottleneck is AT layer i+1
                    'info_loss': info_loss,
                    'severity': info_loss / (mi_flow[0] + 1e-6) # Loss relative to input info
                })
        
        print(f"   MI Flow Detected: {[f'{m:.2f}' for m in mi_flow]}")
        if bottlenecks:
            worst = max(bottlenecks, key=lambda x: x['severity'])
            print(f"   🔥 Worst Bottleneck found at layer {worst['position']} with {worst['info_loss']:.2f} bits lost.")
        
        return sorted(bottlenecks, key=lambda x: x['severity'], reverse=True)

    def calculate_optimal_intervention(self, bottleneck: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses information theory to calculate the minimal intervention needed.
        """
        info_loss = bottleneck['info_loss']
        s = self.network.base_sparsity
        
        # Capacity formula: I_max = -s * log(s) * width
        capacity_per_neuron = -s * np.log(s) if 0 < s < 1 else 0
        
        if capacity_per_neuron > 0:
            neurons_needed = int(np.ceil(info_loss / capacity_per_neuron))
        else:
            neurons_needed = 64 # Fallback if sparsity is 0 or 1
        
        # Differentiated strategy based on severity
        if bottleneck['severity'] > 0.5: # Severe loss
            return {'type': 'insert_layer', 'width': neurons_needed, 'position': bottleneck['position']}
        elif bottleneck['severity'] > 0.2: # Moderate loss
            # Future implementation: add skip connection
            return {'type': 'add_skip_connection', 'position': bottleneck['position'], 'info': 'Moderate severity, skip connection suggested.'}
        else: # Mild loss
            # Future implementation: increase local density
            return {'type': 'increase_density', 'position': bottleneck['position'], 'info': 'Mild severity, local density increase suggested.'}

    def apply_growth_action(self, action: Dict[str, Any]):
        """Applies the calculated optimal growth action."""
        print(f"\n--- Applying Optimal Action: {action['type']} at position {action['position']} ---")
        
        if action['type'] == 'insert_layer':
            self._insert_layer(action['position'], action['width'])
        elif action['type'] == 'add_skip_connection':
            print(f"   (SKIPPED) Action '{action['type']}' is not yet implemented.")
            # self._add_skip_connection(action['position'])
        elif action['type'] == 'increase_density':
            print(f"   (SKIPPED) Action '{action['type']}' is not yet implemented.")
            # self._increase_local_density(action['position'])

    def _insert_layer(self, position: int, new_width: int):
        """Inserts a new layer with the calculated optimal width."""
        new_width = min(max(new_width, 16), 1024) # Clamp width for stability
        
        old_arch = self.network.architecture
        new_arch = old_arch[:position] + [new_width] + old_arch[position:]
        
        print(f"      Growing architecture from {old_arch} to {new_arch}")
        
        new_network = PatchedSparseNetwork(new_arch, self.network.base_sparsity).to(self.device)
        
        # Smartly copy weights to preserve learning
        with torch.no_grad():
            old_layers = self.network.sparse_layers
            new_layers = new_network.sparse_layers
            for i in range(len(new_layers)):
                if i < position:
                    new_layers[i].load_state_dict(old_layers[i].state_dict())
                elif i > position:
                    new_layers[i].load_state_dict(old_layers[i-1].state_dict())
            # The new layer at 'position' is randomly initialized.

        self.network = new_network
        self.history.append(f"Inserted layer of width {new_width} at position {position} to fix info loss.")

    def evolve_step(self):
        """Performs one full cycle of analysis and optimal growth."""
        # 1. Use MI to find the worst information bottleneck
        bottlenecks = self.analyze_information_flow()
        
        if not bottlenecks:
            print("No significant bottlenecks found. Evolution paused.")
            return

        worst_bottleneck = bottlenecks[0]
        
        # 2. Calculate the optimal intervention for that bottleneck
        optimal_action = self.calculate_optimal_intervention(worst_bottleneck)
        
        # 3. Apply the action
        self.apply_growth_action(optimal_action)
        
        print(f"\nCurrent Architecture: {self.network.architecture}")

    def get_layer_activations(self) -> List[torch.Tensor]:
        """Utility to get activations from each layer for analysis."""
        activations = []
        with torch.no_grad():
            h = next(iter(self.data_loader))[0].view(-1, self.network.architecture[0]).to(self.device)
            activations.append(h) # Input layer
            for layer in self.network.sparse_layers:
                h = F.relu(layer(h))
                activations.append(h)
        return activations

# HybridGrowthNetwork removed - replaced by superior DeltaGuidedEvolver system

def load_cifar10_data(batch_size=64):
    """Loads the CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_pretrained_into_patched_network(checkpoint_path: str, device: torch.device) -> PatchedSparseNetwork:
    """Load a pretrained model from GPU seed hunter into our PatchedSparseNetwork structure."""
    print(f"🔬 Loading pretrained model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    architecture = checkpoint['architecture']
    state_dict = checkpoint['model_state_dict']
    
    # Extract sparsity from checkpoint or filename
    base_sparsity = checkpoint.get('sparsity', 0.02)
    if 'patch' in checkpoint_path:
        import re
        sparsity_match = re.search(r'patch([\d.]+)', checkpoint_path)
        if sparsity_match:
            base_sparsity = float(sparsity_match.group(1))
            print(f"   🎯 Extracted sparsity from filename: {base_sparsity}")
    
    print(f"   🏗️  Architecture: {architecture}")
    print(f"   📊 Sparsity: {base_sparsity}")
    
    # Create the PatchedSparseNetwork
    network = PatchedSparseNetwork(architecture, base_sparsity).to(device)
    
    # Load pretrained weights from GPU seed hunter format
    for i, layer in enumerate(network.sparse_layers):
        layer_key = str(i * 2)  # nn.Sequential keys: 0, 2, 4...
        
        if f'{layer_key}.weight' in state_dict:
            print(f"   ✅ Loading pretrained weights for layer {i} (key '{layer_key}.weight')")
            
            # Load weights and bias
            layer.weight.data = state_dict[f'{layer_key}.weight']
            layer.bias.data = state_dict[f'{layer_key}.bias']
            
            # Extract mask from loaded weights to preserve learned sparsity
            mask = (layer.weight.data != 0).float()
            layer.register_buffer('mask', mask)  # Overwrite random mask
            
            print(f"      Preserved {mask.sum().item()}/{mask.numel()} connections")
        else:
            print(f"   ⚠️  No pretrained weights found for layer {i}")
    
    return network

def test_network_accuracy(network: PatchedSparseNetwork, test_loader: DataLoader, device: torch.device) -> float:
    """Test the accuracy of a network."""
    network.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten
            
            # Debug first batch
            if i == 0:
                print(f"   🔍 Debug: Input shape: {data.shape}")
                print(f"   🔍 Debug: Expected input dim: {network.architecture[0]}")
                print(f"   🔍 Debug: Target shape: {target.shape}")
                
                # Test forward pass step by step
                h = data
                for j, layer in enumerate(network.sparse_layers):
                    print(f"   🔍 Debug: Layer {j} input shape: {h.shape}")
                    print(f"   🔍 Debug: Layer {j} weight shape: {layer.weight.shape}")
                    print(f"   🔍 Debug: Layer {j} mask sum: {layer.mask.sum().item()}")
                    h_out = layer(h)
                    print(f"   🔍 Debug: Layer {j} output shape: {h_out.shape}")
                    print(f"   🔍 Debug: Layer {j} output stats: mean={h_out.mean().item():.4f}, std={h_out.std().item():.4f}")
                    if j < len(network.sparse_layers) - 1:
                        h = F.relu(h_out)
                    else:
                        h = h_out
                print(f"   🔍 Debug: Final output shape: {h.shape}")
                print(f"   🔍 Debug: Final output sample: {h[0][:5]}")
            
            output = network(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            # Only debug first batch
            if i == 0:
                break
    
    accuracy = correct / total
    return accuracy

def main():
    """Main function using the DeltaGuidedEvolver system."""
    import argparse
    parser = argparse.ArgumentParser(description='Delta-Guided Architecture Evolution')
    parser.add_argument('--load-model', type=str, help='Path to a saved model checkpoint to start from.')
    parser.add_argument('--evolution-steps', type=int, default=5, help='Number of evolution steps to run.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_cifar10_data()
    
    if args.load_model:
        # Load pretrained model into PatchedSparseNetwork
        network = load_pretrained_into_patched_network(args.load_model, device)
        
        # Test initial accuracy
        print("🧪 Testing initial accuracy...")
        initial_accuracy = test_network_accuracy(network, test_loader, device)
        print(f"   📊 Initial accuracy: {initial_accuracy:.2%}")
        
        # Create OptimalGrowthEvolver with the loaded network
        evolver = OptimalGrowthEvolver(
            seed_arch=network.architecture,
            seed_sparsity=network.base_sparsity,
            data_loader=train_loader,
            device=device
        )
        
        # Replace the evolver's network with our loaded one
        evolver.network = network
        
    else:
        # Create new network from scratch
        initial_arch = [3072, 128, 32, 10]
        base_sparsity = 0.05
        
        evolver = OptimalGrowthEvolver(
            seed_arch=initial_arch,
            seed_sparsity=base_sparsity,
            data_loader=train_loader,
            device=device
        )
        
        # Test initial accuracy
        print("🧪 Testing initial accuracy...")
        initial_accuracy = test_network_accuracy(evolver.network, test_loader, device)
        print(f"   📊 Initial accuracy: {initial_accuracy:.2%}")
    
    print(f"\n🚀 Starting Delta-Guided Evolution for {args.evolution_steps} steps...")
    
    # Run evolution steps
    for step in range(args.evolution_steps):
        print(f"\n🧬 Evolution Step {step + 1}/{args.evolution_steps}")
        
        # Perform one evolution step
        evolver.evolve_step()
        
        # Test accuracy after evolution
        accuracy = test_network_accuracy(evolver.network, test_loader, device)
        print(f"📊 Accuracy after step {step + 1}: {accuracy:.2%}")
        
        # Optional: Early stopping if accuracy is high enough
        if accuracy > 0.80:
            print(f"🎯 Target accuracy reached! Stopping evolution.")
            break
    
    print(f"\n✅ Evolution complete!")
    print(f"📈 Evolution history:")
    for i, event in enumerate(evolver.history):
        print(f"   {i+1}. {event}")

if __name__ == "__main__":
    main()
