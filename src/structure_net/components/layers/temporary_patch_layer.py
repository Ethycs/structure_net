#!/usr/bin/env python3
"""
Temporary Patch Layer Component

Dense patch layer for fixing specific extrema by targeting problematic neurons
with focused dense connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from ...core.base_components import BaseLayer
from ...core.interfaces import ComponentContract, ComponentVersion, Maturity, ResourceRequirements


class TemporaryPatchLayer(BaseLayer):
    """
    Dense patch layer for fixing specific extrema.
    
    Compatible with canonical standard - can be saved/loaded with standard functions.
    This layer extracts a single neuron's activation and processes it through
    a small dense network to provide targeted correction.
    """
    
    def get_contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="TemporaryPatchLayer",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"input_tensor", "source_neuron_idx"},
            provided_outputs={"output_tensor", "patch_info"},
            resources=ResourceRequirements(
                memory_level=ResourceRequirements.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def __init__(self, source_neuron_idx: int, patch_size: int = 8, output_size: int = 4):
        super().__init__(f"TemporaryPatchLayer_n{source_neuron_idx}")
        
        # Type tag for experiment tracking
        self.type = "layer"
        
        # Configuration
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
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get layer configuration and state information."""
        return {
            "type": "TemporaryPatchLayer",
            "source_neuron_idx": self.source_neuron_idx,
            "patch_size": self.patch_size,
            "output_size": self.output_size,
            "layer1_params": self.layer1.weight.numel() + self.layer1.bias.numel(),
            "layer2_params": self.layer2.weight.numel() + self.layer2.bias.numel(),
            "total_params": sum(p.numel() for p in self.parameters())
        }
    
    def supports_modification(self) -> bool:
        """Check if layer supports dynamic modification."""
        return False
    
    def add_connections(self, num_connections: int) -> int:
        """Add connections to the layer (not supported for patch layer)."""
        raise NotImplementedError("TemporaryPatchLayer does not support dynamic modification")
    
    def get_analysis_properties(self) -> Dict[str, Any]:
        """Get properties for analysis."""
        return {
            "source_neuron": self.source_neuron_idx,
            "patch_complexity": self.patch_size * self.output_size,
            "weight_stats": {
                "layer1_mean": self.layer1.weight.mean().item(),
                "layer1_std": self.layer1.weight.std().item(),
                "layer2_mean": self.layer2.weight.mean().item(),
                "layer2_std": self.layer2.weight.std().item()
            }
        }