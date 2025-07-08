#!/usr/bin/env python3
"""
Advanced Layer Components for Integrated Growth System

This module provides the sophisticated layer types needed for advanced growth strategies,
extracted from the hybrid_growth_experiment.py for integration into the tournament system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional

from ..core.layers import StandardSparseLayer


class ThresholdConfig:
    """Configuration for thresholds used in growth and analysis."""
    def __init__(self):
        # Type tag for experiment tracking
        self.type = "config"
        
        self.activation_threshold = 0.01
        self.weight_threshold = 0.01
        self.gradient_threshold = 1e-4
        self.persistence_ratio = 0.8
        self.adaptive = True
        self.min_active_ratio = 0.05
        self.max_active_ratio = 0.5


class MetricsConfig:
    """Configuration for which metrics to compute."""
    def __init__(self):
        # Type tag for experiment tracking
        self.type = "config"
        
        self.compute_mi = True
        self.compute_activity = True
        self.compute_sensli = True
        self.compute_graph = True
        self.compute_betweenness = False
        self.compute_spectral = False


class ExtremaAwareSparseLayer(StandardSparseLayer):
    """
    Advanced sparse layer with extrema-guided connectivity.
    
    Extends the canonical StandardSparseLayer with extrema-aware mask creation
    while maintaining full compatibility with the canonical standard.
    """
    
    def __init__(self, in_features: int, out_features: int, base_sparsity: float, 
                 extrema_to_embed: Optional[Dict[str, List[int]]] = None):
        # Initialize as standard sparse layer first
        super().__init__(in_features, out_features, base_sparsity)
        
        # Override type tag for experiment tracking
        self.type = "layer"
        
        # Store extrema information for potential mask updates
        self.base_sparsity = base_sparsity
        self.extrema_to_embed = extrema_to_embed or {}
        
        # Update mask based on extrema if provided
        if extrema_to_embed:
            self._update_mask_for_extrema()
    
    def _update_mask_for_extrema(self):
        """Update mask to have higher density around extrema regions."""
        # Start with current mask
        enhanced_mask = self.mask.clone()
        
        # High-density connections TO revive dead neurons (targets)
        dead_neurons = self.extrema_to_embed.get('low', [])
        if dead_neurons:
            revival_density = 0.20  # 20% dense input connections for dead neurons
            for neuron_idx in dead_neurons:
                if neuron_idx < self.linear.out_features:
                    # Give dead neuron more diverse inputs
                    revival_connections = (torch.rand(self.linear.in_features) < revival_density).float()
                    enhanced_mask[neuron_idx, :] = torch.maximum(
                        enhanced_mask[neuron_idx, :], 
                        revival_connections
                    )
        
        # High-density connections FROM saturated neurons (sources)
        saturated_neurons = self.extrema_to_embed.get('high', [])
        if saturated_neurons:
            relief_density = 0.15  # 15% dense output connections for saturated neurons
            for neuron_idx in saturated_neurons:
                if neuron_idx < self.linear.in_features:
                    # Give saturated neuron more output channels
                    relief_connections = (torch.rand(self.linear.out_features) < relief_density).float()
                    enhanced_mask[:, neuron_idx] = torch.maximum(
                        enhanced_mask[:, neuron_idx], 
                        relief_connections
                    )
        
        # Update the registered buffer
        self.mask.data = enhanced_mask
        
        # Re-apply mask to weights
        with torch.no_grad():
            self.linear.weight.data *= self.mask


class TemporaryPatchModule(nn.Module):
    """
    Dense patch module for fixing specific extrema.
    
    Compatible with canonical standard - can be saved/loaded with standard functions.
    """
    
    def __init__(self, source_neuron_idx: int, patch_size: int = 8, output_size: int = 4):
        super().__init__()
        
        # Type tag for experiment tracking
        self.type = "layer"
        
        self.source_neuron_idx = source_neuron_idx
        self.patch_size = patch_size
        self.output_size = output_size
        
        # Dense layers for the patch
        self.layer1 = nn.Linear(1, patch_size)
        self.layer2 = nn.Linear(patch_size, output_size)
        
        # Initialize with small weights to avoid disrupting main network
        with torch.no_grad():
            self.layer1.weight.data *= 0.1
            self.layer2.weight.data *= 0.1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract single neuron activation and process through dense patch."""
        # Extract the source neuron's activation
        patch_input = x[:, self.source_neuron_idx].unsqueeze(-1)
        
        # Process through dense layers
        h = F.relu(self.layer1(patch_input))
        return self.layer2(h)


def lsuv_init_layer(layer: nn.Module, sample_input: torch.Tensor, target_variance: float = 1.0):
    """
    LSUV initialization for a single layer based on sample input.
    
    Args:
        layer: The layer to initialize
        sample_input: Sample input to the layer
        target_variance: Target output variance
    """
    with torch.no_grad():
        for _ in range(10):  # Max 10 iterations
            if hasattr(layer, 'mask'):
                # For sparse layers
                output = F.linear(sample_input, layer.linear.weight * layer.mask, layer.linear.bias)
            else:
                # For dense layers
                output = layer(sample_input)
            
            var = output.var()
            if var > 0:
                if hasattr(layer, 'linear'):
                    layer.linear.weight.data /= torch.sqrt(var / target_variance)
                else:
                    layer.weight.data /= torch.sqrt(var / target_variance)
            
            # Check convergence
            if abs(output.var() - target_variance) < 0.01:
                break
        
        return F.relu(output) if output.dim() > 1 else output


@torch.no_grad()
def apply_neuron_sorting(network: nn.Module, layer_idx: int):
    """
    Apply importance-based neuron sorting to a specific layer.
    
    Args:
        network: The network containing the layers
        layer_idx: Index of the layer to sort
    """
    # Get sparse layers
    sparse_layers = [layer for layer in network if isinstance(layer, (StandardSparseLayer, ExtremaAwareSparseLayer))]
    
    if layer_idx >= len(sparse_layers) - 1:  # Don't sort output layer
        return
    
    current_layer = sparse_layers[layer_idx]
    
    # Calculate importance of each output neuron based on its weight norm
    importance = torch.linalg.norm(current_layer.linear.weight, ord=2, dim=1)
    
    # Get the indices that would sort the neurons by importance (descending)
    perm_indices = torch.argsort(importance, descending=True)
    
    # Apply permutation to current layer (output neurons)
    current_layer.linear.weight.data = current_layer.linear.weight.data[perm_indices, :]
    current_layer.linear.bias.data = current_layer.linear.bias.data[perm_indices]
    current_layer.mask = current_layer.mask[perm_indices, :]
    
    # Apply permutation to next layer (input connections)
    if layer_idx + 1 < len(sparse_layers):
        next_layer = sparse_layers[layer_idx + 1]
        next_layer.linear.weight.data = next_layer.linear.weight.data[:, perm_indices]
        next_layer.mask = next_layer.mask[:, perm_indices]


def sort_all_network_layers(network: nn.Module):
    """
    Apply neuron sorting to all hidden layers in the network.
    
    Args:
        network: The network to sort
    """
    sparse_layers = [layer for layer in network if isinstance(layer, (StandardSparseLayer, ExtremaAwareSparseLayer))]
    
    # Sort every hidden layer, but not the final output layer
    for i in range(len(sparse_layers) - 1):
        apply_neuron_sorting(network, i)


# Export all components
__all__ = [
    'ThresholdConfig',
    'MetricsConfig',
    'ExtremaAwareSparseLayer',
    'TemporaryPatchModule',
    'lsuv_init_layer',
    'apply_neuron_sorting',
    'sort_all_network_layers'
]
