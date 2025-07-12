#!/usr/bin/env python3
"""
Structured Linear Layer Component

Sparse linear layer with structured sparsity patterns like block-diagonal,
butterfly, and Toeplitz patterns for efficient computation.
"""

import torch
import numpy as np
from typing import Dict, Any

from .sparse_linear import SparseLinear
from ...core.interfaces import ComponentContract, ComponentVersion, Maturity, ResourceRequirements


class StructuredLinear(SparseLinear):
    """
    Structured sparse linear layer with pattern-based connectivity.
    
    Maintains compatibility with StandardSparseLayer while adding
    structured sparsity patterns.
    """
    
    def get_contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="StructuredLinear",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.EXPERIMENTAL,
            required_inputs={"input_tensor"},
            provided_outputs={"output_tensor", "mask", "sparsity_ratio", "pattern_info"},
            optional_inputs={"pattern"},
            resources=ResourceRequirements(
                memory_level=ResourceRequirements.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def __init__(self, in_features: int, out_features: int, 
                 sparsity: float, pattern: str = "block"):
        # Initialize with standard sparse layer
        super().__init__(in_features, out_features, sparsity=1.0)  # Start fully sparse
        
        # Override name for tracking
        self._name = f"StructuredLinear_{in_features}x{out_features}_{pattern}"
        
        self.pattern = pattern
        self.target_sparsity = sparsity
        
        # Create structured pattern
        self._create_structured_pattern()
    
    def _create_structured_pattern(self):
        """Create structured sparsity pattern."""
        device = self.linear.weight.device
        
        if self.pattern == "block":
            # Block-diagonal pattern
            self._create_block_pattern()
        elif self.pattern == "butterfly":
            # Butterfly pattern (FFT-like)
            self._create_butterfly_pattern()
        elif self.pattern == "toeplitz":
            # Toeplitz pattern (shift-invariant)
            self._create_toeplitz_pattern()
        else:
            # Fall back to random
            self._create_random_pattern()
    
    def _create_block_pattern(self):
        """Create block-diagonal sparsity pattern."""
        in_features = self.linear.in_features
        out_features = self.linear.out_features
        
        # Determine block size
        block_size = max(1, int(np.sqrt(in_features * out_features * (1 - self.target_sparsity))))
        
        # Create block pattern
        self.mask.data.zero_()
        
        for i in range(0, out_features, block_size):
            for j in range(0, in_features, block_size):
                # Fill block
                i_end = min(i + block_size, out_features)
                j_end = min(j + block_size, in_features)
                self.mask.data[i:i_end, j:j_end] = 1.0
        
        # Adjust to match target sparsity
        self._adjust_pattern_sparsity()
    
    def _create_butterfly_pattern(self):
        """Create butterfly (FFT-like) sparsity pattern."""
        # Implement butterfly pattern
        # This is a placeholder - implement based on specific requirements
        self._create_random_pattern()
    
    def _create_toeplitz_pattern(self):
        """Create Toeplitz (shift-invariant) sparsity pattern."""
        # Each row is a shifted version of the first row
        in_features = self.linear.in_features
        out_features = self.linear.out_features
        
        # Create base pattern for first row
        connections_per_row = int(in_features * (1 - self.target_sparsity))
        
        self.mask.data.zero_()
        
        for i in range(out_features):
            # Shift pattern for each row
            start_idx = (i * connections_per_row // out_features) % in_features
            for j in range(connections_per_row):
                idx = (start_idx + j) % in_features
                self.mask.data[i, idx] = 1.0
    
    def _create_random_pattern(self):
        """Fall back to random pattern."""
        mask = torch.rand_like(self.linear.weight) > self.target_sparsity
        self.mask.data = mask.float()
    
    def _adjust_pattern_sparsity(self):
        """Adjust pattern to match target sparsity exactly."""
        current_sparsity = 1 - self.get_sparsity_ratio()
        
        if abs(current_sparsity - self.target_sparsity) > 0.01:
            # Need to adjust
            if current_sparsity < self.target_sparsity:
                # Remove connections
                num_remove = int((self.target_sparsity - current_sparsity) * self.mask.numel())
                self.prune_connections(num_remove, strategy="random")
            else:
                # Add connections
                num_add = int((current_sparsity - self.target_sparsity) * self.mask.numel())
                self.add_connections(num_add, strategy="random")
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get layer configuration and state information."""
        info = super().get_layer_info()
        info.update({
            "type": "StructuredLinear",
            "pattern": self.pattern,
            "target_sparsity": self.target_sparsity,
            "actual_sparsity": 1 - self.get_sparsity_ratio()
        })
        return info