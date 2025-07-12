#!/usr/bin/env python3
"""
Standard Sparse Layer Component

The canonical sparse layer implementation used throughout structure_net.
This component provides the foundation for sparse connectivity in neural networks.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ...core.base_components import BaseLayer
from ...core.interfaces import ComponentContract, ComponentVersion, Maturity, ResourceRequirements


class StandardSparseLayer(BaseLayer):
    """
    THE canonical sparse layer used by all systems.
    
    This is the single source of truth for sparse layer behavior.
    All systems must use this exact implementation to ensure compatibility.
    """
    
    def get_contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="StandardSparseLayer",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"input_tensor"},
            provided_outputs={"output_tensor", "mask", "sparsity_ratio"},
            resources=ResourceRequirements(
                memory_level=ResourceRequirements.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def __init__(self, in_features: int, out_features: int, sparsity: float):
        super().__init__(f"StandardSparseLayer_{in_features}x{out_features}")
        
        # Type tag for experiment tracking
        self.type = "layer"
        
        # Configuration
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
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
        # Ensure mask is on the same device as the weight
        mask = self.mask.to(self.linear.weight.device)
        return torch.nn.functional.linear(x, self.linear.weight * mask, self.linear.bias)
    
    def get_connection_count(self) -> int:
        """Get number of active connections in this layer."""
        return self.mask.sum().item()
    
    def get_sparsity_ratio(self) -> float:
        """Get actual sparsity ratio of this layer."""
        return self.get_connection_count() / self.mask.numel()
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get layer configuration and state information."""
        return {
            "type": "StandardSparseLayer",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "configured_sparsity": self.sparsity,
            "actual_sparsity": 1.0 - self.get_sparsity_ratio(),
            "connection_count": self.get_connection_count(),
            "total_parameters": self.mask.numel()
        }
    
    def supports_modification(self) -> bool:
        """Check if layer supports dynamic modification."""
        return False
    
    def add_connections(self, num_connections: int) -> int:
        """Add connections to the layer (not supported for standard sparse layer)."""
        raise NotImplementedError("StandardSparseLayer does not support dynamic modification")
    
    def get_analysis_properties(self) -> Dict[str, Any]:
        """Get properties for analysis."""
        return {
            "sparsity_ratio": self.get_sparsity_ratio(),
            "connection_count": self.get_connection_count(),
            "weight_stats": {
                "mean": self.linear.weight.mean().item(),
                "std": self.linear.weight.std().item(),
                "min": self.linear.weight.min().item(),
                "max": self.linear.weight.max().item()
            }
        }