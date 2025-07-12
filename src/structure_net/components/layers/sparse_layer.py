"""
Sparse Layer Component.

A neural network layer with configurable sparsity that follows
the component architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
import logging

from ...core.base_components import BaseLayer
from ...core.interfaces import (
    ComponentContract, ComponentVersion,
    Maturity, ResourceRequirements, ResourceLevel
)


class SparseLayer(BaseLayer):
    """
    Sparse linear layer with dynamic connection management.
    
    Features:
    - Configurable sparsity levels
    - Dynamic connection addition/removal
    - Gradient-based importance tracking
    - Component-aware design
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 sparsity: float = 0.9,
                 bias: bool = True,
                 structured: bool = False,
                 name: str = None):
        """
        Initialize sparse layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            sparsity: Target sparsity (0 to 1, where 1 = fully sparse)
            bias: Whether to use bias
            structured: Use structured sparsity pattern
            name: Optional custom name
        """
        super().__init__(name or f"SparseLayer_{in_features}x{out_features}")
        
        self.in_features = in_features
        self.out_features = out_features
        self.target_sparsity = sparsity
        self.structured = structured
        
        # Parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Binary mask for sparsity
        self.register_buffer('mask', torch.zeros_like(self.weight))
        
        # Track importance scores for dynamic updates
        self.register_buffer('importance_scores', torch.zeros_like(self.weight))
        
        # Initialize
        self.reset_parameters()
        
        # Define contract
        self._contract = ComponentContract(
            component_name=self.name,
            version=ComponentVersion(1, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={'input'},
            provided_outputs={'output', 'analysis_properties'},
            resources=ResourceRequirements(
                memory_level=ResourceLevel.LOW,
                requires_gpu=False,
                parallel_safe=True
            )
        )
    
    @property
    def contract(self) -> ComponentContract:
        """Return component contract."""
        return self._contract
    
    def reset_parameters(self):
        """Initialize parameters and sparsity mask."""
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize sparsity mask
        self._initialize_mask()
    
    def _initialize_mask(self):
        """Initialize the sparsity mask."""
        num_connections = int((1 - self.target_sparsity) * self.in_features * self.out_features)
        
        if self.structured:
            # Structured sparsity - preserve some structure
            connections_per_output = max(1, num_connections // self.out_features)
            
            for i in range(self.out_features):
                # Connect each output to a subset of inputs
                if connections_per_output < self.in_features:
                    indices = torch.randperm(self.in_features)[:connections_per_output]
                    self.mask[i, indices] = 1
                else:
                    self.mask[i, :] = 1
        else:
            # Random sparsity
            if num_connections < self.mask.numel():
                indices = torch.randperm(self.mask.numel())[:num_connections]
                self.mask.view(-1)[indices] = 1
            else:
                self.mask.fill_(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse connections."""
        # Apply mask to weights
        masked_weight = self.weight * self.mask
        
        # Update importance scores based on gradients if available
        if self.training and self.weight.grad is not None:
            self.importance_scores = 0.9 * self.importance_scores + 0.1 * self.weight.grad.abs()
        
        return F.linear(x, masked_weight, self.bias)
    
    def get_analysis_properties(self) -> Dict[str, torch.Tensor]:
        """Get properties for analysis."""
        return {
            'weight': self.weight,
            'mask': self.mask,
            'bias': self.bias if self.bias is not None else torch.zeros(self.out_features),
            'importance_scores': self.importance_scores,
            'effective_weight': self.weight * self.mask,
            'sparsity': torch.tensor(1.0 - self.mask.mean().item()),
            'dead_neurons': torch.tensor((self.mask.sum(dim=1) == 0).sum().item())
        }
    
    def supports_modification(self) -> bool:
        """Layer supports structural modifications."""
        return True
    
    def add_connections(self, num_connections: int, use_importance: bool = True):
        """
        Add new connections to the layer.
        
        Args:
            num_connections: Number of connections to add
            use_importance: Use importance scores to guide selection
        """
        # Find inactive connections
        inactive_mask = (self.mask == 0)
        num_inactive = inactive_mask.sum().item()
        
        if num_inactive == 0 or num_connections == 0:
            return
        
        num_to_add = min(num_connections, num_inactive)
        
        if use_importance and self.importance_scores.any():
            # Select based on importance scores
            inactive_importance = self.importance_scores * inactive_mask
            _, indices = torch.topk(inactive_importance.view(-1), num_to_add)
        else:
            # Random selection
            inactive_indices = inactive_mask.nonzero(as_tuple=False)
            selected = torch.randperm(len(inactive_indices))[:num_to_add]
            indices = inactive_indices[selected]
            indices = indices[:, 0] * self.in_features + indices[:, 1]
        
        # Activate selected connections
        self.mask.view(-1)[indices] = 1
        
        # Initialize new weights
        for idx in indices:
            i = idx // self.in_features
            j = idx % self.in_features
            
            # Initialize with appropriate scale
            fan_in = self.mask[i].sum().item()
            std = np.sqrt(2.0 / fan_in) if fan_in > 0 else 0.1
            self.weight.data[i, j] = torch.randn(1) * std
        
        self.log(logging.DEBUG, f"Added {num_to_add} connections")
    
    def prune_connections(self, num_connections: int, use_importance: bool = True):
        """
        Remove connections from the layer.
        
        Args:
            num_connections: Number of connections to remove
            use_importance: Use importance scores to guide selection
        """
        # Find active connections
        active_mask = (self.mask == 1)
        num_active = active_mask.sum().item()
        
        if num_active == 0 or num_connections == 0:
            return
        
        # Ensure we don't prune everything
        num_to_prune = min(num_connections, num_active - self.out_features)
        if num_to_prune <= 0:
            return
        
        if use_importance:
            # Prune least important connections
            active_importance = self.importance_scores * active_mask
            
            # Ensure each output keeps at least one connection
            for i in range(self.out_features):
                if self.mask[i].sum() == 1:
                    # Don't prune the last connection
                    active_importance[i] = float('inf')
            
            _, indices = torch.topk(-active_importance.view(-1), num_to_prune)
        else:
            # Random pruning
            active_indices = active_mask.nonzero(as_tuple=False)
            
            # Filter out last connections
            valid_indices = []
            for idx in range(len(active_indices)):
                i = active_indices[idx, 0]
                if self.mask[i].sum() > 1:
                    valid_indices.append(idx)
            
            if len(valid_indices) > 0:
                selected = torch.randperm(len(valid_indices))[:num_to_prune]
                indices = active_indices[valid_indices][selected]
                indices = indices[:, 0] * self.in_features + indices[:, 1]
            else:
                return
        
        # Deactivate selected connections
        self.mask.view(-1)[indices] = 0
        
        # Zero out weights
        for idx in indices:
            i = idx // self.in_features
            j = idx % self.in_features
            self.weight.data[i, j] = 0
        
        self.log(logging.DEBUG, f"Pruned {num_to_prune} connections")
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return int(self.mask.sum().item())
    
    def get_sparsity_stats(self) -> Dict[str, Any]:
        """Get detailed sparsity statistics."""
        total_connections = self.mask.numel()
        active_connections = self.mask.sum().item()
        
        # Per-neuron statistics
        output_connections = self.mask.sum(dim=1)
        input_connections = self.mask.sum(dim=0)
        
        return {
            'total_possible': total_connections,
            'active': int(active_connections),
            'sparsity': 1.0 - (active_connections / total_connections),
            'target_sparsity': self.target_sparsity,
            'output_stats': {
                'mean_connections': output_connections.float().mean().item(),
                'max_connections': output_connections.max().item(),
                'min_connections': output_connections.min().item(),
                'dead_outputs': (output_connections == 0).sum().item()
            },
            'input_stats': {
                'mean_connections': input_connections.float().mean().item(),
                'max_connections': input_connections.max().item(),
                'min_connections': input_connections.min().item(),
                'unused_inputs': (input_connections == 0).sum().item()
            }
        }
    
    def redistribute_connections(self):
        """Redistribute connections to balance the layer."""
        stats = self.get_sparsity_stats()
        
        # Find overconnected and underconnected neurons
        output_connections = self.mask.sum(dim=1)
        mean_connections = output_connections.float().mean()
        
        for i in range(self.out_features):
            current = output_connections[i].item()
            target = int(mean_connections.item())
            
            if current > target + 2:
                # Prune some connections
                excess = current - target
                row_importance = self.importance_scores[i] * self.mask[i]
                _, indices = torch.topk(-row_importance, min(excess, current - 1))
                
                for j in indices:
                    self.mask[i, j] = 0
                    self.weight.data[i, j] = 0
            
            elif current < target - 2 and current > 0:
                # Add some connections
                deficit = target - current
                inactive = (self.mask[i] == 0)
                num_inactive = inactive_mask.sum().item()
                
                if num_inactive > 0:
                    num_to_add = min(deficit, num_inactive)
                    indices = torch.randperm(self.in_features)[inactive][:num_to_add]
                    
                    for j in indices:
                        self.mask[i, j] = 1
                        # Initialize new weight
                        std = np.sqrt(2.0 / target)
                        self.weight.data[i, j] = torch.randn(1) * std