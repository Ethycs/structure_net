#!/usr/bin/env python3
"""
Extrema-Aware Sparse Layer Component

Advanced sparse layer that adjusts connectivity based on extrema detection,
helping to revive dead neurons and relieve saturated ones.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional

from .standard_sparse_layer import StandardSparseLayer
from ...core.interfaces import ComponentContract, ComponentVersion, Maturity, ResourceRequirements


class ExtremaAwareSparseLayer(StandardSparseLayer):
    """
    Advanced sparse layer with extrema-guided connectivity.
    
    Extends the canonical StandardSparseLayer with extrema-aware mask creation
    while maintaining full compatibility with the canonical standard.
    """
    
    def get_contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="ExtremaAwareSparseLayer",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"input_tensor"},
            provided_outputs={"output_tensor", "mask", "sparsity_ratio", "extrema_info"},
            optional_inputs={"extrema_to_embed"},
            resources=ResourceRequirements(
                memory_level=ResourceRequirements.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def __init__(self, in_features: int, out_features: int, base_sparsity: float, 
                 extrema_to_embed: Optional[Dict[str, List[int]]] = None):
        # Initialize as standard sparse layer first
        super().__init__(in_features, out_features, base_sparsity)
        
        # Override name for tracking
        self._name = f"ExtremaAwareSparseLayer_{in_features}x{out_features}"
        
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
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get layer configuration and state information."""
        info = super().get_layer_info()
        info.update({
            "type": "ExtremaAwareSparseLayer",
            "base_sparsity": self.base_sparsity,
            "has_extrema_embedding": bool(self.extrema_to_embed),
            "dead_neurons_targeted": len(self.extrema_to_embed.get('low', [])),
            "saturated_neurons_targeted": len(self.extrema_to_embed.get('high', []))
        })
        return info