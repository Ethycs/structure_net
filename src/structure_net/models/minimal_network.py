"""
MinimalNetwork - Backward Compatibility Layer

This module provides a `MinimalNetwork` class that is a wrapper around the
new `create_standard_network` factory function. This ensures that older
experiments that rely on `MinimalNetwork` can still run without modification.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from ..core.network_factory import create_standard_network
from ..core.network_analysis import get_network_stats

class MinimalNetwork(nn.Module):
    """
    A backward-compatibility wrapper for the old MinimalNetwork class.
    
    This class uses the new canonical `create_standard_network` factory
    to construct the network, but exposes an interface that is compatible
    with older experiments.
    """
    def __init__(self, 
                 layer_sizes: List[int], 
                 sparsity: float, 
                 activation: str = 'relu', 
                 device: Optional[str] = 'cpu'):
        super().__init__()
        
        self.layer_sizes = layer_sizes
        self.sparsity = sparsity
        self.activation = activation
        self.device = torch.device(device)
        
        # Use the new factory to create the actual network
        self.network = create_standard_network(
            architecture=layer_sizes,
            sparsity=sparsity,
            activation=activation,
            device=device
        )
        
        # For compatibility, connection_masks can be accessed from the sparse layers
        self.connection_masks = [
            layer.mask for layer in self.network if hasattr(layer, 'mask')
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_connectivity_stats(self) -> dict:
        """Provides the connectivity stats expected by older experiments."""
        return get_network_stats(self.network)

    def get_gradient_norm(self) -> float:
        """Computes the L2 norm of all gradients."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def detect_extrema(self, epoch: int) -> dict:
        """
        Placeholder for the old extrema detection method.
        
        In a real scenario, this would need to be implemented to match the
        behavior of the original `detect_extrema` method if it was part of
        the `MinimalNetwork` class. For now, it returns an empty dict.
        """
        # This is a simplified placeholder. The actual extrema detection logic
        # is now in `src/structure_net/evolution/extrema_analyzer.py`.
        # Older experiments might need this method on the network object itself.
        return {'high': [], 'low': []}

    def add_connections(self, connections: List[tuple]):
        """
        Placeholder for adding connections.
        
        This would need to be implemented to modify the masks of the
        StandardSparseLayer instances in self.network.
        """
        print(f"Warning: `add_connections` on the compatibility `MinimalNetwork` is not fully implemented.")
        pass

    def state_dict_sparse(self):
        """Returns a state dict that is compatible with the old format."""
        return self.network.state_dict()

    def load_state_dict_sparse(self, state_dict):
        """Loads a state dict from the old format."""
        self.network.load_state_dict(state_dict)
