"""
Structure Net: Multi-Scale Snapshots Neural Network

A PyTorch implementation of neural networks that grow dynamically during training
based on extrema detection and multi-scale snapshot preservation.
"""

from .models.multi_scale_network import MultiScaleNetwork, create_multi_scale_network
from .core.minimal_network import MinimalNetwork, create_minimal_network
from .core.growth_scheduler import GrowthScheduler, StructuralLimits
from .core.connection_router import ConnectionRouter, ParsimonousRouter
from .snapshots.snapshot_manager import SnapshotManager

__version__ = "0.1.0"
__author__ = "Ethycs"

__all__ = [
    "MultiScaleNetwork",
    "create_multi_scale_network",
    "MinimalNetwork", 
    "create_minimal_network",
    "GrowthScheduler",
    "StructuralLimits",
    "ConnectionRouter",
    "ParsimonousRouter",
    "SnapshotManager"
]
