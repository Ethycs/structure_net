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

# Import canonical standard (THE foundation)
from .core.model_io import (
    StandardSparseLayer,
    ExtremaAwareSparseLayer,
    TemporaryPatchLayer,
    create_standard_network,
    save_model_seed,
    load_model_seed,
    get_network_stats,
    sort_all_network_layers
)

# Import evolution system (built on canonical standard)
from .evolution.network_evolver import OptimalGrowthEvolver
from .evolution.extrema_analyzer import analyze_layer_extrema, detect_network_extrema
from .evolution.information_theory import analyze_information_flow, estimate_mi_proxy

# Import seed search (uses canonical standard)
from .seed_search.gpu_seed_hunter import GPUSeedHunter
from .seed_search.architecture_generator import ArchitectureGenerator

__version__ = "0.2.0"
__author__ = "Ethycs"

__all__ = [
    # Legacy components
    "MultiScaleNetwork",
    "create_multi_scale_network",
    "MinimalNetwork", 
    "create_minimal_network",
    "GrowthScheduler",
    "StructuralLimits",
    "ConnectionRouter",
    "ParsimonousRouter",
    "SnapshotManager",
    
    # Canonical Standard (THE foundation)
    "StandardSparseLayer",
    "ExtremaAwareSparseLayer", 
    "TemporaryPatchLayer",
    "create_standard_network",
    "save_model_seed",
    "load_model_seed",
    "get_network_stats",
    "sort_all_network_layers",
    
    # Evolution System
    "OptimalGrowthEvolver",
    "analyze_layer_extrema",
    "detect_network_extrema",
    "analyze_information_flow",
    "estimate_mi_proxy",
    
    # Seed Search
    "GPUSeedHunter",
    "ArchitectureGenerator"
]
