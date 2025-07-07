"""
Core components for structure_net - The Canonical Standard

This module contains the canonical model I/O system that serves as the
single source of truth for all sparse network operations.
"""

from .model_io import (
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
    cleanup_data_directory
)

__version__ = "2.0.0"
__author__ = "Structure Net Team"

__all__ = [
    # Core layer types
    "StandardSparseLayer",
    "ExtremaAwareSparseLayer", 
    "TemporaryPatchLayer",
    
    # Network creation functions
    "create_standard_network",
    "create_extrema_aware_network",
    "create_evolvable_network",
    "load_pretrained_into_canonical",
    
    # I/O functions
    "save_model_seed",
    "load_model_seed",
    "test_save_load_compatibility",
    
    # Analysis functions
    "get_network_stats",
    "apply_neuron_sorting",
    "sort_all_network_layers",
    
    # Validation functions
    "validate_model_quality",
    "validate_models_in_directory", 
    "delete_invalid_models",
    "cleanup_data_directory"
]
