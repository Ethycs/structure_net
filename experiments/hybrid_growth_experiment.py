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
        # Store patch outputs to be added to the *next* layer's pre-activations
        patch_outputs_for_next_layer = {}

        h = x
        for i, layer in enumerate(self.sparse_layers):
            # 1. Calculate pre-activation from the main sparse layer
            pre_activation = layer(h)

            # 2. Add corrections from patches on the PREVIOUS layer
            if i in patch_outputs_for_next_layer:
                pre_activation += patch_outputs_for_next_layer[i]

            # 3. Apply activation function
            h = F.relu(pre_activation)

            # 4. If this layer has patches, compute their outputs for the NEXT layer
            layer_key = str(i)
            if layer_key in self.patches:
                # Sum the outputs of all patches attached to this layer
                total_patch_correction = 0
                for patch in self.patches[layer_key]:
                    total_patch_correction += patch(h)
                
                # The output dimension of each patch is 4. We need to project it
                # to the next layer's size. For simplicity, we tile and slice.
                # A more advanced version would use a learned projection.
                next_layer_dim = self.architecture[i+2] if i + 2 < len(self.architecture) else 0
                if next_layer_dim > 0:
                    correction_size = total_patch_correction.shape[1]
                    repeats = (next_layer_dim + correction_size - 1) // correction_size
                    full_correction = total_patch_correction.repeat(1, repeats)[:, :next_layer_dim]
                    patch_outputs_for_next_layer[i + 1] = full_correction
        
        return h

class DeltaGuidedEvolver:
    """
    Starts with a seed architecture and continuously finds the most efficient 
    place to add sparse capacity (layers) or targeted fixes (patches).
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
        print(f"Initialized DeltaGuidedEvolver with seed: {seed_arch}, sparsity: {seed_sparsity}")

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

    def _calculate_params(self, arch: List[int], sparsity: float) -> int:
        """Calculates the number of parameters in a sparse architecture."""
        params = 0
        for i in range(len(arch) - 1):
            params += arch[i] * arch[i+1] * sparsity
        return int(params)

    def analyze_growth_options(self) -> List[Dict[str, Any]]:
        """
        Calculates the "efficiency" (delta) of all possible growth actions.
        This is the core of the decision-making process.
        """
        print("\n--- Analyzing All Possible Growth Options ---")
        options = []
        activations = self.get_layer_activations()
        current_arch = self.network.architecture
        current_sparsity = self.network.base_sparsity
        current_params = self._calculate_params(current_arch, current_sparsity)

        # --- Option Type 1: Insert a New Sparse Layer ---
        for i in range(1, len(current_arch) - 1): # Can insert between any two existing layers
            # Heuristic for new layer width
            new_width = int((current_arch[i-1] + current_arch[i]) / 4) # Be conservative
            new_width = min(max(new_width, 16), 512) # Clamp width
            
            # Estimate information gain
            # Gain comes from relieving the bottleneck at layer i-1 -> i
            mi_before = estimate_mi_sparse(activations[i-1], activations[i])
            # The new layer will ideally pass more info
            mi_after_est = np.log2(new_width + 1)
            info_gain = max(0, mi_after_est - mi_before)
            
            # Calculate parameter cost
            new_arch = current_arch[:i] + [new_width] + current_arch[i:]
            params_added = self._calculate_params(new_arch, current_sparsity) - current_params
            
            efficiency = info_gain / (params_added + 1e-6)

            options.append({
                'type': 'insert_layer',
                'position': i,
                'new_width': new_width,
                'efficiency': efficiency,
                'info': f"Relieves MI bottleneck ({mi_before:.2f} bits)"
            })

        # --- Option Type 2: Add Patches for Severe Extrema ---
        for i in range(len(self.network.sparse_layers)):
            extrema = analyze_layer_extrema(activations[i+1])
            num_extrema = len(extrema['low']) + len(extrema['high'])
            
            if num_extrema > 0:
                # Info gain from patching is about fixing broken neurons
                info_gain = num_extrema * 0.1 # Heuristic: each patch provides a small, fixed gain
                
                # Parameter cost of a patch is constant (1->8->4 = 40 params)
                params_added = num_extrema * 40
                
                efficiency = info_gain / (params_added + 1e-6)

                options.append({
                    'type': 'add_patches',
                    'position': i,
                    'num_extrema': num_extrema,
                    'efficiency': efficiency,
                    'info': f"Fixing {num_extrema} dead/saturated neurons"
                })

        return sorted(options, key=lambda x: x['efficiency'], reverse=True)

    def apply_growth_action(self, action: Dict[str, Any]):
        """Applies the chosen growth action to the network."""
        print(f"\n--- Applying Best Action: {action['type']} at position {action['position']} (Efficiency: {action['efficiency']:.6f}) ---")
        print(f"      Reason: {action['info']}")
        
        if action['type'] == 'insert_layer':
            self._insert_layer(action['position'], action['new_width'])
        elif action['type'] == 'add_patches':
            self._add_patches(action['position'])

    def _insert_layer(self, position: int, new_width: int):
        """Inserts a new ExtremaAwareSparseLayer and rebuilds the network."""
        old_arch = self.network.architecture
        new_arch = old_arch[:position] + [new_width] + old_arch[position:]
        
        print(f"Growing architecture from {old_arch} to {new_arch}")
        
        new_network = PatchedSparseNetwork(new_arch, self.network.base_sparsity).to(self.device)
        
        # Smartly copy weights to preserve learning
        with torch.no_grad():
            for i in range(len(new_network.sparse_layers)):
                if i < position: # Copy layers before the insertion point
                    new_network.sparse_layers[i].load_state_dict(self.network.sparse_layers[i].state_dict())
                elif i > position: # Copy layers after the insertion point
                    new_network.sparse_layers[i].load_state_dict(self.network.sparse_layers[i-1].state_dict())
            # The new layer at 'position' is randomly initialized.

        self.network = new_network
        self.history.append(f"Inserted layer of width {new_width} at position {position}. New arch: {new_arch}")

    def _add_patches(self, layer_idx: int):
        """Adds temporary patches to fix extrema at a given layer."""
        activations = self.get_layer_activations()
        extrema = analyze_layer_extrema(activations[layer_idx+1])
        
        layer_key = str(layer_idx)
        if layer_key not in self.network.patches:
            self.network.patches[layer_key] = nn.ModuleList()

        # Patch the most severe extrema
        sources_to_patch = extrema['high'][:3] + extrema['low'][:3]
        for source_idx in sources_to_patch:
            self.network.patches[layer_key].append(TemporaryPatch(source_idx).to(self.device))
        
        self.history.append(f"Added {len(sources_to_patch)} patches to layer {layer_idx}.")

    def evolve_step(self):
        """Performs one full cycle of analysis and growth."""
        # 1. Analyze all possible growth moves and their efficiency
        growth_options = self.analyze_growth_options()
        
        if not growth_options:
            print("No growth options found. Evolution complete.")
            return

        # 2. Select the single best action
        best_action = growth_options[0]
        
        # 3. Apply the action
        self.apply_growth_action(best_action)
        
        print(f"\nCurrent Architecture: {self.network.architecture}")
        print(f"Active Patches: { {k: len(v) for k,v in self.network.patches.items()} }")

class HybridGrowthNetwork(nn.Module):
    """
    A network that combines a sparse scaffold with multi-scale dense patches,
    and grows based on extrema patterns.
    """
    
    def __init__(self, initial_arch, base_sparsity, device='cuda', sample_batch=None, is_pretrained=False, pretrained_state_dict=None):
        super().__init__()
        self.device = device
        self.architecture = initial_arch
        self.is_pretrained = is_pretrained
        
        if is_pretrained and pretrained_state_dict is not None:
            # DON'T apply LSUV to pretrained scaffold - it would destroy learned features
            print("   🔒 Creating scaffold from pretrained weights (NO LSUV)")
            self.scaffold = create_sparse_scaffold_from_pretrained(initial_arch, base_sparsity, device, pretrained_state_dict)
        else:
            # Apply LSUV only for new networks
            print("   🌱 Creating new scaffold with LSUV initialization")
            self.scaffold = create_lsuv_sparse_scaffold(initial_arch, base_sparsity, device, sample_batch)
            
        self.scale_patches = nn.ModuleDict({
            'coarse': nn.ModuleList(),
            'medium': nn.ModuleList(),
            'fine': nn.ModuleList()
        })
        self.patch_connections = {}
        self.growth_history = []
        self.current_accuracy = 0.0
        self.activations = []

    def forward(self, x):
        x = x.view(x.size(0), -1)
        self.activations = []
        self.layer_inputs = []  # Store inputs to each layer for patch initialization
        h = x

        for i, layer in enumerate(self.scaffold):
            # Store the input to this layer (for patch creation)
            self.layer_inputs.append(h.detach())
            
            # CRITICAL FIX: Apply mask during forward pass like the original network
            # The original network applies mask during forward, not just during initialization
            if hasattr(layer, 'mask'):
                # Apply the mask to weights during forward pass
                masked_weight = layer.weight * layer.mask
                sparse_out = F.linear(h, masked_weight, layer.bias)
            else:
                # Fallback for layers without masks
                sparse_out = layer(h)
            
            patch_out = self._compute_patch_contributions(h, i, sparse_out)
            h = sparse_out + patch_out
            
            self.activations.append(h.detach())
            if i < len(self.scaffold) - 1:
                h = F.relu(h)
        
        return h

    def _compute_patch_contributions(self, input_h, layer_idx, sparse_out):
        return torch.zeros_like(sparse_out)

    def analyze_multiscale_extrema(self):
        scale_extrema = {
            'coarse': {'high': [], 'low': []},
            'medium': {'high': [], 'low': []},
            'fine': {'high': [], 'low': []}
        }
        
        print(f"🔍 Analyzing extrema across {len(self.activations[:-1])} layers...")
        
        for i, activation in enumerate(self.activations[:-1]):
            mean_acts = activation.mean(dim=0)
            high_threshold = mean_acts.mean() + 2 * mean_acts.std()
            low_threshold = 0.1
            
            high_indices = torch.where(mean_acts > high_threshold)[0].cpu().numpy().tolist()
            low_indices = torch.where(mean_acts < low_threshold)[0].cpu().numpy().tolist()

            if i < len(self.scaffold) / 3:
                scale = 'coarse'
            elif i < 2 * len(self.scaffold) / 3:
                scale = 'medium'
            else:
                scale = 'fine'
                
            scale_extrema[scale]['high'].extend([(i, idx) for idx in high_indices])
            scale_extrema[scale]['low'].extend([(i, idx) for idx in low_indices])
            
            print(f"   Layer {i} ({scale}): {len(high_indices)} high, {len(low_indices)} low extrema")
            print(f"      Activation stats: mean={mean_acts.mean():.3f}, std={mean_acts.std():.3f}")
            print(f"      Thresholds: high>{high_threshold:.3f}, low<{low_threshold:.3f}")
            
        # Summary
        print(f"📊 Extrema Summary:")
        for scale, extrema in scale_extrema.items():
            print(f"   {scale.capitalize()}: {len(extrema['high'])} high, {len(extrema['low'])} low")
            
        return scale_extrema

    def determine_growth_scale(self, scale_extrema):
        coarse_pressure = len(scale_extrema['coarse']['high'])
        medium_pressure = len(scale_extrema['medium']['high'])
        fine_pressure = len(scale_extrema['fine']['high'])
        
        print(f"🎯 Growth Pressure Analysis:")
        print(f"   Coarse: {coarse_pressure}, Medium: {medium_pressure}, Fine: {fine_pressure}")
        
        primary_scale = 'coarse'
        if medium_pressure > coarse_pressure and medium_pressure > fine_pressure:
            primary_scale = 'medium'
        elif fine_pressure > coarse_pressure and fine_pressure > medium_pressure:
            primary_scale = 'fine'

        needs_new_layer = coarse_pressure > 10
        print(f"   Primary scale: {primary_scale}")
        print(f"   Needs new layer: {needs_new_layer} (coarse pressure: {coarse_pressure})")

        growth_decision = {
            'primary_scale': primary_scale,
            'needs_new_layer': needs_new_layer,
            'patch_density': {
                'coarse': min(0.5, coarse_pressure / 20),
                'medium': min(0.3, medium_pressure / 50),
                'fine': min(0.2, fine_pressure / 100)
            }
        }
        return growth_decision

    def add_sparse_layer_at_scale(self, growth_scale, sample_batch):
        print("   🌱 Adding a new sparse layer...")
        new_layer_size = 128
        old_architecture = self.architecture.copy()
        self.architecture.insert(-1, new_layer_size)
        
        if self.is_pretrained:
            # For pretrained networks, only add the new layer with LSUV
            # Keep existing pretrained layers intact
            print("   🔒 Preserving pretrained layers, only initializing new layer")
            
            # Get the output from the previous layer to initialize the new layer
            with torch.no_grad():
                sample_batch_flat = sample_batch.view(sample_batch.size(0), -1)
                h = sample_batch_flat
                for layer in self.scaffold[:-1]:  # All but the last layer
                    h = F.relu(F.linear(h, layer.weight * layer.mask, layer.bias))
                
                # Create new layer and insert it before the output layer
                new_layer = nn.Linear(old_architecture[-2], new_layer_size).to(self.device)
                mask = _create_sparse_mask((new_layer_size, old_architecture[-2]), 0.05).to(self.device)
                new_layer.register_buffer('mask', mask)
                new_layer.weight.data.mul_(mask)
                
                # Apply LSUV only to the new layer
                lsuv_init_new_layer(new_layer, h)
                
                # Insert the new layer before the output layer
                self.scaffold.insert(-1, new_layer)
        else:
            # For new networks, recreate the entire scaffold with LSUV
            self.scaffold = create_lsuv_sparse_scaffold(self.architecture, 0.05, self.device, sample_batch)
            
        print(f"   🌱 New architecture: {self.architecture}")

    def add_multiscale_patches(self, scale_extrema, growth_scale):
        patches_added = {'coarse': 0, 'medium': 0, 'fine': 0}
        print(f"🔧 Creating patches for extrema...")
        
        for scale, extrema in scale_extrema.items():
            for layer_idx, neuron_idx in extrema['high'][:5]:
                if layer_idx < len(self.architecture) - 1:
                    # Debug patch dimensions
                    in_features = self.architecture[layer_idx]
                    out_features = self.architecture[layer_idx+1]
                    print(f"   Creating {scale} patch for layer {layer_idx}: {in_features} -> {out_features}")
                    
                    patch = nn.Linear(in_features, out_features).to(self.device)
                    
                    # Apply LSUV to new patches using stored layer inputs
                    if hasattr(self, 'layer_inputs') and len(self.layer_inputs) > layer_idx:
                        with torch.no_grad():
                            # Use the stored input to this layer
                            sample_input = self.layer_inputs[layer_idx][:32]  # First 32 samples
                            
                            print(f"      Input shape: {sample_input.shape}, Patch expects: {in_features}")
                            
                            # Validate dimensions before LSUV
                            if sample_input.shape[1] == in_features:
                                lsuv_init_new_layer(patch, sample_input)
                                print(f"   ✅ LSUV initialized {scale} patch for layer {layer_idx}")
                            else:
                                print(f"   ⚠️  Dimension mismatch for layer {layer_idx}: input {sample_input.shape[1]} != expected {in_features}")
                                print(f"      Skipping LSUV for this patch")
                    else:
                        print(f"   ⚠️  No layer inputs available for layer {layer_idx}, skipping LSUV")
                    
                    self.scale_patches[scale].append(patch)
                    patches_added[scale] += 1
                    
        print(f"   🔧 Added patches - Coarse: {patches_added['coarse']}, Medium: {patches_added['medium']}, Fine: {patches_added['fine']}")

    def train_to_convergence(self, train_loader, test_loader, max_epochs=5):
        optimizer_groups = [
            {'params': self.scaffold.parameters(), 'lr': 0.0001},
            {'params': self.scale_patches['coarse'].parameters(), 'lr': 0.001},
            {'params': self.scale_patches['medium'].parameters(), 'lr': 0.005},
            {'params': self.scale_patches['fine'].parameters(), 'lr': 0.01}
        ]
        optimizer = optim.Adam(optimizer_groups)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        for epoch in range(max_epochs):
            self.train()
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

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
            print(f"    Epoch {epoch}: Accuracy {accuracy:.2%}")

        self.current_accuracy = best_acc
        return best_acc

    def test_initial_accuracy(self, test_loader):
        """Test the initial accuracy of the loaded model."""
        print("🧪 Testing initial accuracy...")
        self.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        initial_accuracy = correct / len(test_loader.dataset)
        print(f"   📊 Initial accuracy: {initial_accuracy:.2%}")
        self.current_accuracy = initial_accuracy
        return initial_accuracy

    def grow_with_scale_aware_patches(self, train_loader, test_loader, target_accuracy=0.95):
        iteration = 0
        sample_batch, _ = next(iter(train_loader))
        sample_batch = sample_batch.to(self.device)

        # Test initial accuracy first
        if self.is_pretrained:
            self.test_initial_accuracy(test_loader)

        while self.current_accuracy < target_accuracy and iteration < 5:
            iteration += 1
            print(f"\n🌱 Multi-Scale Growth Iteration {iteration}")
            
            self.train_to_convergence(train_loader, test_loader)
            
            scale_extrema = self.analyze_multiscale_extrema()
            growth_scale = self.determine_growth_scale(scale_extrema)
            
            if growth_scale['needs_new_layer']:
                self.add_sparse_layer_at_scale(growth_scale, sample_batch)
            
            self.add_multiscale_patches(scale_extrema, growth_scale)
            
            print(f"📊 Current accuracy: {self.current_accuracy:.2%}")

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

def main():
    """Main function to run the experiment."""
    import argparse
    parser = argparse.ArgumentParser(description='Hybrid Growth Experiment')
    parser.add_argument('--load-model', type=str, help='Path to a saved model checkpoint to start from.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_cifar10_data()
    
    sample_batch, _ = next(iter(train_loader))
    sample_batch = sample_batch.to(device)

    if args.load_model:
        print(f"🔬 Loading model from checkpoint: {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        initial_arch = checkpoint['architecture']
        
        # Debug: Print the loaded architecture
        print(f"   🏗️  Loaded architecture: {initial_arch}")
        
        # CRITICAL FIX: Extract sparsity from filename if not in checkpoint
        base_sparsity = checkpoint.get('sparsity', 0.02)
        if 'patch' in args.load_model:
            # Extract sparsity from filename like "patch0.065"
            import re
            sparsity_match = re.search(r'patch([\d.]+)', args.load_model)
            if sparsity_match:
                base_sparsity = float(sparsity_match.group(1))
                print(f"   🎯 Extracted sparsity from filename: {base_sparsity}")
        
        print(f"   📊 Using sparsity: {base_sparsity} (from {'checkpoint' if 'sparsity' in checkpoint else 'filename'})")
        
        # Create network with pretrained flag and state dict
        network = HybridGrowthNetwork(
            initial_arch=initial_arch,
            base_sparsity=base_sparsity,
            device=device,
            sample_batch=sample_batch.view(sample_batch.size(0), -1),
            is_pretrained=True,
            pretrained_state_dict=checkpoint['model_state_dict']
        )
        
        # Load remaining state (patches, etc.) with strict=False to allow missing keys
        network.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("   ✅ Pretrained model loaded successfully (LSUV skipped for scaffold)")
    else:
        network = HybridGrowthNetwork(
            initial_arch=[3072, 128, 32, 10],
            base_sparsity=0.05,
            device=device,
            sample_batch=sample_batch.view(sample_batch.size(0), -1)
        )

    network.grow_with_scale_aware_patches(train_loader, test_loader, target_accuracy=0.80)

if __name__ == "__main__":
    main()
