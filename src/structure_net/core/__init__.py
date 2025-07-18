"""
Core components for Structure Net - Modular Architecture

This module provides the fundamental building blocks for sparse neural networks.
The core has been modularized into focused components:

- layers.py: Core layer definitions (StandardSparseLayer, ExtremaAwareSparseLayer, etc.)
- network_factory.py: Network creation functions
- network_analysis.py: Network statistics and analysis
- io_operations.py: Model saving and loading
- validation.py: Model quality validation and cleanup
- lsuv.py: LSUV initialization system

New Component Architecture:
- interfaces.py: Core interfaces and contracts for self-aware components
- base_components.py: Base implementations for all component types
- compatibility.py: Component compatibility and registry system
"""

# Import core layer types
from .layers import (
    StandardSparseLayer,
    ExtremaAwareSparseLayer,
    TemporaryPatchLayer
)

# Import network creation functions
from .network_factory import (
    create_standard_network,
    create_extrema_aware_network,
    create_evolvable_network,
    load_pretrained_into_canonical
)

# Import network analysis functions
from .network_analysis import (
    get_network_stats,
    apply_neuron_sorting,
    sort_all_network_layers
)

# Import I/O operations
from .io_operations import (
    save_model_seed,
    load_model_seed,
    test_save_load_compatibility
)

# Import validation functions
from .validation import (
    validate_model_quality,
    validate_models_in_directory,
    delete_invalid_models,
    cleanup_data_directory
)

# Import LSUV initialization
from .lsuv import (
    lsuv_init_layer,
    lsuv_init_network,
    lsuv_init_new_layers_only,
    create_lsuv_initialized_network,
    analyze_network_variance_flow
)

# Import New Component Architecture (optional imports to avoid breaking existing code)
try:
    from .interfaces import (
        Maturity, ComponentVersion, ResourceLevel, ResourceRequirements,
        ComponentContract, EvolutionContext, AnalysisReport, EvolutionPlan,
        IComponent, ILayer, IModel, ITrainer, IMetric, IAnalyzer,
        IStrategy, IEvolver, IScheduler, IOrchestrator
    )
    from .base_components import (
        BaseComponent, BaseLayer, BaseModel, BaseTrainer, BaseMetric,
        BaseAnalyzer, BaseStrategy, BaseEvolver, BaseScheduler, BaseOrchestrator
    )
    from .compatibility import (
        CompatibilityLevel, CompatibilityIssue, ComponentRegistry, CompatibilityManager
    )
    _COMPONENT_ARCHITECTURE_AVAILABLE = True
except ImportError:
    _COMPONENT_ARCHITECTURE_AVAILABLE = False

__version__ = "2.0.0"
__author__ = "Structure Net Team"

# Build __all__ list dynamically
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
    
    # Network analysis functions
    "get_network_stats",
    "apply_neuron_sorting",
    "sort_all_network_layers",
    
    # I/O operations
    "save_model_seed",
    "load_model_seed",
    "test_save_load_compatibility",
    
    # Validation functions
    "validate_model_quality",
    "validate_models_in_directory", 
    "delete_invalid_models",
    "cleanup_data_directory",
    
    # LSUV initialization
    "lsuv_init_layer",
    "lsuv_init_network",
    "lsuv_init_new_layers_only",
    "create_lsuv_initialized_network",
    "analyze_network_variance_flow"
]

# Add component architecture exports if available
if _COMPONENT_ARCHITECTURE_AVAILABLE:
    __all__.extend([
        # New Component Architecture
        "Maturity", "ComponentVersion", "ResourceLevel", "ResourceRequirements",
        "ComponentContract", "EvolutionContext", "AnalysisReport", "EvolutionPlan",
        "IComponent", "ILayer", "IModel", "ITrainer", "IMetric", "IAnalyzer",
        "IStrategy", "IEvolver", "IScheduler", "IOrchestrator",
        "BaseComponent", "BaseLayer", "BaseModel", "BaseTrainer", "BaseMetric",
        "BaseAnalyzer", "BaseStrategy", "BaseEvolver", "BaseScheduler", "BaseOrchestrator",
        "CompatibilityLevel", "CompatibilityIssue", "ComponentRegistry", "CompatibilityManager"
    ])
