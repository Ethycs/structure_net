"""
Modern Multi-Scale Network Implementation

This module defines a modern, composable version of a multi-scale network.
The network itself is a standard nn.Module. The multi-scale growth and
evolution logic is handled externally by the ComposableEvolutionSystem.

DEPRECATED: This module is deprecated. Please use the new component-based architecture:
    from structure_net.components.models import MultiScaleModel
    
The new MultiScaleModel provides enhanced functionality:
- Full component architecture integration with contracts
- Dynamic multi-scale block management
- Scale importance tracking and adaptation
- Growth potential analysis
- Integrated scale-specific pruning and growth

Migration example:
    # Old way (deprecated):
    from structure_net.models import ModernMultiScaleNetwork
    model = ModernMultiScaleNetwork([784, 256, 10], initial_sparsity=0.9)
    
    # New way:
    from structure_net.components.models import MultiScaleModel
    model = MultiScaleModel([784, 256, 10], scales=[1, 2, 4], initial_sparsity=0.9)
"""

import torch
import torch.nn as nn
from typing import List, Dict
import warnings

from ..core.layers import StandardSparseLayer

# Issue deprecation warning on import
warnings.warn(
    "ModernMultiScaleNetwork is deprecated. Please use structure_net.components.models.MultiScaleModel instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

class ModernMultiScaleNetwork(nn.Module):
    """
    A network architecture that can be grown in a multi-scale fashion
    by an external evolution system.
    
    DEPRECATED: Use MultiScaleModel from structure_net.components.models instead.
    """
    def __init__(self, initial_architecture: List[int], initial_sparsity: float):
        warnings.warn(
            "ModernMultiScaleNetwork is deprecated. Use MultiScaleModel from structure_net.components.models instead.",
            DeprecationWarning,
            stacklevel=2
        )
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
