"""
Modern Multi-Scale Network Implementation

This module defines a modern, composable version of a multi-scale network.
The network itself is a standard nn.Module. The multi-scale growth and
evolution logic is handled externally by the ComposableEvolutionSystem.
"""

import torch
import torch.nn as nn
from typing import List, Dict

from ..core.layers import StandardSparseLayer

class ModernMultiScaleNetwork(nn.Module):
    """
    A network architecture that can be grown in a multi-scale fashion
    by an external evolution system.
    """
    def __init__(self, initial_architecture: List[int], initial_sparsity: float):
        super().__init__()
        
        self.architecture = initial_architecture
        self.sparsity = initial_sparsity
        
        self.layers = nn.ModuleList()
        self._build_from_architecture()

    def _build_from_architecture(self):
        """Builds the network layers from the current architecture spec."""
        self.layers.clear()
        for i in range(len(self.architecture) - 1):
            in_features = self.architecture[i]
            out_features = self.architecture[i+1]
            self.layers.append(
                StandardSparseLayer(in_features, out_features, self.sparsity)
            )
            if i < len(self.architecture) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input if it's not already
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        for layer in self.layers:
            x = layer(x)
        return x

    def get_current_architecture(self) -> List[int]:
        """Returns the current architecture of the network."""
        return self.architecture

    def get_stats(self) -> Dict[str, int]:
        """Returns basic statistics about the network."""
        total_params = sum(p.numel() for p in self.parameters())
        active_params = sum(layer.get_connection_count() for layer in self.layers if isinstance(layer, StandardSparseLayer))
        return {
            "total_parameters": total_params,
            "active_connections": active_params,
            "depth": len([l for l in self.layers if isinstance(l, StandardSparseLayer)])
        }
