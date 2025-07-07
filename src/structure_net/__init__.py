"""
Structure Net: Multi-Scale Snapshots Neural Network

A PyTorch implementation of neural networks that grow dynamically during training
based on extrema detection and multi-scale snapshot preservation.
"""

# Legacy components temporarily disabled due to dependencies on deleted modules
# from .models.multi_scale_network import MultiScaleNetwork, create_multi_scale_network
# from .snapshots.snapshot_manager import SnapshotManager

# Import canonical standard (THE foundation) - now modular
from .core import (
    StandardSparseLayer,
    ExtremaAwareSparseLayer,
    TemporaryPatchLayer,
    create_standard_network,
    create_extrema_aware_network,
    create_evolvable_network,
    load_pretrained_into_canonical,
    save_model_seed,
    load_model_seed,
    test_save_load_compatibility,
    get_network_stats,
    apply_neuron_sorting,
    sort_all_network_layers,
    validate_model_quality,
    validate_models_in_directory,
    delete_invalid_models,
    cleanup_data_directory,
    lsuv_init_layer,
    lsuv_init_network,
    lsuv_init_new_layers_only,
    create_lsuv_initialized_network,
    analyze_network_variance_flow
)

# Import evolution system (built on canonical standard)
from .evolution.network_evolver import OptimalGrowthEvolver
from .evolution.extrema_analyzer import analyze_layer_extrema, detect_network_extrema
from .evolution.integrated_growth_system import (
    StructureNetGrowthSystem,
    ThresholdConfig,
    MetricsConfig,
    ExactMutualInformation,
    analyze_and_grow_network
)
# MI analysis removed - using direct extrema-driven approach
# from .evolution.information_theory import analyze_information_flow, estimate_mi_proxy

# Import seed search (uses canonical standard)
from .seed_search.gpu_seed_hunter import GPUSeedHunter
from .seed_search.architecture_generator import ArchitectureGenerator

__version__ = "0.2.0"
__author__ = "Ethycs"

__all__ = [
    # Canonical Standard (THE foundation) - modular
    "StandardSparseLayer",
    "ExtremaAwareSparseLayer", 
    "TemporaryPatchLayer",
    "create_standard_network",
    "create_extrema_aware_network",
    "create_evolvable_network",
    "load_pretrained_into_canonical",
    "save_model_seed",
    "load_model_seed",
    "test_save_load_compatibility",
    "get_network_stats",
    "apply_neuron_sorting",
    "sort_all_network_layers",
    "validate_model_quality",
    "validate_models_in_directory",
    "delete_invalid_models",
    "cleanup_data_directory",
    
    # LSUV Initialization (sparse-optimized)
    "lsuv_init_layer",
    "lsuv_init_network",
    "lsuv_init_new_layers_only",
    "create_lsuv_initialized_network",
    "analyze_network_variance_flow",
    
    # Evolution System
    "OptimalGrowthEvolver",
    "analyze_layer_extrema",
    "detect_network_extrema",
    
    # Integrated Growth System
    "StructureNetGrowthSystem",
    "ThresholdConfig",
    "MetricsConfig",
    "ExactMutualInformation",
    "analyze_and_grow_network",
    
    # Seed Search
    "GPUSeedHunter",
    "ArchitectureGenerator"
]
