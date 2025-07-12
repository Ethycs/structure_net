#!/usr/bin/env python3
"""
Sparse Linear Layer Component

Sparse linear layer with dynamic growth and pruning capabilities.
Supports adding/removing connections during training for adaptive sparsity.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional

from .standard_sparse_layer import StandardSparseLayer
from ...core.interfaces import ComponentContract, ComponentVersion, Maturity, ResourceRequirements


class SparseLinear(StandardSparseLayer):
    """
    Sparse linear layer with growth capabilities.
    
    Extends StandardSparseLayer to support dynamic growth while maintaining
    full compatibility with existing infrastructure.
    """
    
    def get_contract(self) -> ComponentContract:
        return ComponentContract(
            component_name="SparseLinear",
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"input_tensor"},
            provided_outputs={"output_tensor", "mask", "sparsity_ratio", "growth_capacity"},
            optional_inputs={"growth_strategy", "prune_strategy"},
            resources=ResourceRequirements(
                memory_level=ResourceRequirements.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    def __init__(self, in_features: int, out_features: int, sparsity: float):
        super().__init__(in_features, out_features, sparsity)
        
        # Override name for tracking
        self._name = f"SparseLinear_{in_features}x{out_features}"
        
        # Growth tracking
        self.initial_sparsity = sparsity
        self.growth_history = []
        
        # For gauge-aware operations
        self._importance_scores = None
        
    def add_connections(self, num_new: int, strategy: str = "random"):
        """
        Add new connections to the layer.
        
        Args:
            num_new: Number of connections to add
            strategy: Growth strategy ("random", "importance", "gradient")
        """
        device = self.linear.weight.device
        mask = self.mask.to(device)
        
        # Find currently inactive connections
        inactive_mask = (mask == 0)
        inactive_indices = inactive_mask.nonzero(as_tuple=False)
        
        if len(inactive_indices) == 0:
            return 0  # Already fully connected
        
        # Select which connections to activate
        if strategy == "random":
            # Random selection
            perm = torch.randperm(len(inactive_indices), device=device)
            selected_indices = inactive_indices[perm[:num_new]]
            
        elif strategy == "importance":
            # Select based on gradient magnitude or importance scores
            if self._importance_scores is not None:
                scores = self._importance_scores[inactive_mask]
                _, top_indices = torch.topk(scores, min(num_new, len(scores)))
                selected_indices = inactive_indices[top_indices]
            else:
                # Fall back to random
                perm = torch.randperm(len(inactive_indices), device=device)
                selected_indices = inactive_indices[perm[:num_new]]
                
        elif strategy == "gradient":
            # Select based on gradient information
            if self.linear.weight.grad is not None:
                grad_magnitude = self.linear.weight.grad.abs()
                inactive_grads = grad_magnitude[inactive_mask]
                _, top_indices = torch.topk(inactive_grads, min(num_new, len(inactive_grads)))
                selected_indices = inactive_indices[top_indices]
            else:
                # Fall back to random
                perm = torch.randperm(len(inactive_indices), device=device)
                selected_indices = inactive_indices[perm[:num_new]]
        
        # Activate selected connections
        for idx in selected_indices:
            i, j = idx[0].item(), idx[1].item()
            self.mask[i, j] = 1.0
            
            # Initialize new connection (important for training stability)
            with torch.no_grad():
                # Use Kaiming initialization scaled by current connectivity
                fan_in = self.mask[i, :].sum().item()
                std = (2.0 / fan_in) ** 0.5 if fan_in > 0 else 0
                self.linear.weight.data[i, j] = torch.randn(1, device=device) * std
        
        # Record growth event
        self.growth_history.append({
            'num_added': len(selected_indices),
            'strategy': strategy,
            'new_sparsity': self.get_sparsity_ratio()
        })
        
        return len(selected_indices)
    
    def prune_connections(self, num_remove: int, strategy: str = "magnitude"):
        """
        Remove connections from the layer.
        
        Args:
            num_remove: Number of connections to remove
            strategy: Pruning strategy ("magnitude", "gradient", "random")
        """
        device = self.linear.weight.device
        mask = self.mask.to(device)
        
        # Find currently active connections
        active_mask = (mask == 1)
        active_indices = active_mask.nonzero(as_tuple=False)
        
        if len(active_indices) <= num_remove:
            return 0  # Can't remove that many
        
        # Select which connections to deactivate
        if strategy == "magnitude":
            # Remove smallest magnitude weights
            active_weights = self.linear.weight[active_mask].abs()
            _, bottom_indices = torch.topk(active_weights, num_remove, largest=False)
            selected_indices = active_indices[bottom_indices]
            
        elif strategy == "gradient":
            # Remove based on gradient information
            if self.linear.weight.grad is not None:
                grad_magnitude = self.linear.weight.grad.abs()
                active_grads = grad_magnitude[active_mask]
                _, bottom_indices = torch.topk(active_grads, num_remove, largest=False)
                selected_indices = active_indices[bottom_indices]
            else:
                # Fall back to magnitude
                active_weights = self.linear.weight[active_mask].abs()
                _, bottom_indices = torch.topk(active_weights, num_remove, largest=False)
                selected_indices = active_indices[bottom_indices]
                
        elif strategy == "random":
            perm = torch.randperm(len(active_indices), device=device)
            selected_indices = active_indices[perm[:num_remove]]
        
        # Deactivate selected connections
        for idx in selected_indices:
            i, j = idx[0].item(), idx[1].item()
            self.mask[i, j] = 0.0
            # Zero out the weight
            with torch.no_grad():
                self.linear.weight.data[i, j] = 0.0
        
        return len(selected_indices)
    
    def get_growth_capacity(self) -> int:
        """Get number of connections that can still be added."""
        return (self.mask == 0).sum().item()
    
    def compute_importance_scores(self, method: str = "weight_magnitude"):
        """
        Compute importance scores for all potential connections.
        
        Used for guided growth strategies.
        """
        if method == "weight_magnitude":
            self._importance_scores = self.linear.weight.abs()
        elif method == "gradient_magnitude":
            if self.linear.weight.grad is not None:
                self._importance_scores = self.linear.weight.grad.abs()
            else:
                self._importance_scores = self.linear.weight.abs()
        elif method == "fisher":
            # Approximate Fisher information
            if self.linear.weight.grad is not None:
                self._importance_scores = self.linear.weight.grad.pow(2)
            else:
                self._importance_scores = torch.zeros_like(self.linear.weight)
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get layer configuration and state information."""
        info = super().get_layer_info()
        info.update({
            "type": "SparseLinear",
            "initial_sparsity": self.initial_sparsity,
            "growth_capacity": self.get_growth_capacity(),
            "growth_events": len(self.growth_history),
            "has_importance_scores": self._importance_scores is not None
        })
        return info