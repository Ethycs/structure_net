"""
Structure Net: Multi-Scale Snapshots Neural Network

A PyTorch implementation of neural networks that grow dynamically during training
based on extrema detection and multi-scale snapshot preservation.
"""

# Configure environment before any torch imports
from .config import setup_cuda_devices
setup_cuda_devices()

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
from .components.evolvers.optimal_growth_evolver import OptimalGrowthEvolver
# Removed deprecated IntegratedGrowthSystem - use evolution.components instead
from .core.config_schemas import ThresholdConfig, MetricsConfig
# Import adaptive learning rate strategies
from .components.strategies.adaptive_learning_rates import (
    ExponentialBackoffScheduler,
    LayerwiseAdaptiveRates,
    SoftClampingScheduler,
    ScaleDependentRates,
    GrowthPhaseScheduler,
    AdaptiveLearningRateManager,
    create_adaptive_training_loop,
    # Advanced combination systems
    ExtremaPhaseScheduler,
    LayerAgeAwareLR,
    MultiScaleLearning,
    UnifiedAdaptiveLearning
)
# MI analysis removed - using direct extrema-driven approach
# from .evolution.information_theory import analyze_information_flow, estimate_mi_proxy

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
    
    # Configuration classes
    "ThresholdConfig",
    "MetricsConfig",
    
    # Adaptive Learning Rate Strategies
    "ExponentialBackoffScheduler",
    "LayerwiseAdaptiveRates",
    "SoftClampingScheduler",
    "ScaleDependentRates",
    "GrowthPhaseScheduler",
    "AdaptiveLearningRateManager",
    "create_adaptive_training_loop",
    # Advanced combination systems
    "ExtremaPhaseScheduler",
    "LayerAgeAwareLR",
    "MultiScaleLearning",
    "UnifiedAdaptiveLearning",
    
    # Seed Search
    "ArchitectureGenerator"
]
