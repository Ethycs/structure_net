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
        
        # Type tag for experiment tracking
        self.type = "layer"
        
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


# Export layer definitions
__all__ = [
    'StandardSparseLayer',
    'ExtremaAwareSparseLayer', 
    'TemporaryPatchLayer',
    'SparseLinear',
    'StructuredLinear'
]

class SparseLinear(StandardSparseLayer):
    """
    Sparse linear layer with growth capabilities.
    
    Extends StandardSparseLayer to support dynamic growth while maintaining
    full compatibility with existing infrastructure.
    """
    
    def __init__(self, in_features: int, out_features: int, sparsity: float):
        super().__init__(in_features, out_features, sparsity)
        
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
    
    def apply_gauge_transform(self, permutation: torch.Tensor):
        """
        Apply gauge transformation (permutation) to this layer.
        
        Only valid for hidden layers (not input/output).
        """
        # Permute output dimensions
        self.linear.weight.data = self.linear.weight.data[permutation, :]
        self.mask.data = self.mask.data[permutation, :]
        if self.linear.bias is not None:
            self.linear.bias.data = self.linear.bias.data[permutation]
    
    def get_neuron_importance(self) -> torch.Tensor:
        """
        Get importance scores for each neuron (output dimension).
        
        Used for gauge-aware operations.
        """
        # Combine incoming and outgoing connection strengths
        incoming = self.linear.weight.abs().sum(dim=1)
        outgoing_mask = self.mask.sum(dim=1)
        
        # Weighted combination
        importance = incoming * torch.sqrt(outgoing_mask + 1e-8)
        return importance
    
    def get_growth_statistics(self) -> dict:
        """Get statistics about growth history."""
        if not self.growth_history:
            return {
                'total_growth_events': 0,
                'total_connections_added': 0,
                'initial_sparsity': self.initial_sparsity,
                'current_sparsity': self.get_sparsity_ratio()
            }
        
        total_added = sum(event['num_added'] for event in self.growth_history)
        
        return {
            'total_growth_events': len(self.growth_history),
            'total_connections_added': total_added,
            'initial_sparsity': self.initial_sparsity,
            'current_sparsity': self.get_sparsity_ratio(),
            'growth_history': self.growth_history
        }
    
    def extra_repr(self) -> str:
        """String representation with sparsity info."""
        return (f'in_features={self.linear.in_features}, '
                f'out_features={self.linear.out_features}, '
                f'sparsity={1 - self.get_sparsity_ratio():.2%}, '
                f'active_connections={self.get_connection_count()}')

class StructuredLinear(SparseLinear):
    """
    Structured sparse linear layer with pattern-based connectivity.
    
    Maintains compatibility with StandardSparseLayer while adding
    structured sparsity patterns.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 sparsity: float, pattern: str = "block"):
        # Initialize with standard sparse layer
        super().__init__(in_features, out_features, sparsity=1.0)  # Start fully sparse
        
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

