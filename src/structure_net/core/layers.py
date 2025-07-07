#!/usr/bin/env python3
"""
Core Layer Definitions

This module contains the fundamental layer types used throughout structure_net.
All layer definitions follow the canonical standard for maximum compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional


class StandardSparseLayer(nn.Module):
    """
    THE canonical sparse layer used by all systems.
    
    This is the single source of truth for sparse layer behavior.
    All systems must use this exact implementation to ensure compatibility.
    """
    
    def __init__(self, in_features: int, out_features: int, sparsity: float):
        super().__init__()
        
        # Standard structure: nested linear module (matches GPU seed hunter)
        self.linear = nn.Linear(in_features, out_features)
        
        # Standard mask creation and registration
        mask = torch.rand_like(self.linear.weight) < sparsity
        self.register_buffer('mask', mask.float())
        
        # Standard initialization with mask application
        with torch.no_grad():
            self.linear.weight.data *= self.mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        THE canonical forward pass - identical everywhere.
        
        This ensures all systems have identical behavior.
        """
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)
    
    def get_connection_count(self) -> int:
        """Get number of active connections in this layer."""
        return self.mask.sum().item()
    
    def get_sparsity_ratio(self) -> float:
        """Get actual sparsity ratio of this layer."""
        return self.get_connection_count() / self.mask.numel()


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
    
    def create_extrema_aware_mask(self, extrema_to_embed: Dict[str, List[int]]) -> torch.Tensor:
        """Creates a mask with high density in specified regions (from hybrid experiment)."""
        # Start with a base uniform sparse mask
        base_mask = (torch.rand(self.linear.out_features, self.linear.in_features) < self.base_sparsity).float()
        
        if not extrema_to_embed:
            return base_mask

        # Create denser regions for embedding
        patch_mask = torch.zeros_like(base_mask)

        # 1. High-density connections TO revive dead neurons (targets)
        dead_neurons = extrema_to_embed.get('low', [])
        if dead_neurons:
            revival_density = 0.20
            for neuron_idx in dead_neurons:
                if neuron_idx < self.linear.out_features:
                    patch_mask[neuron_idx, :] = (torch.rand(self.linear.in_features) < revival_density).float()

        # 2. High-density connections FROM saturated neurons (sources)
        saturated_neurons = extrema_to_embed.get('high', [])
        if saturated_neurons:
            relief_density = 0.15
            for neuron_idx in saturated_neurons:
                if neuron_idx < self.linear.in_features:
                    patch_mask[:, neuron_idx] = (torch.rand(self.linear.out_features) < relief_density).float()

        # Combine the masks: take the union of all connections
        return torch.maximum(base_mask, patch_mask)


class TemporaryPatchLayer(nn.Module):
    """
    Dense patch layer for fixing specific extrema.
    
    Compatible with canonical standard - can be saved/loaded with standard functions.
    """
    
    def __init__(self, source_neuron_idx: int, patch_size: int = 8, output_size: int = 4):
        super().__init__()
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


# Export layer definitions
__all__ = [
    'StandardSparseLayer',
    'ExtremaAwareSparseLayer', 
    'TemporaryPatchLayer'
]
