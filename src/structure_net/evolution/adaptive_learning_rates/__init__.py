"""
Adaptive Learning Rate Strategies for Structure Net

This package implements sophisticated differential learning rate strategies
that adapt based on network growth phase, layer position, connection age,
and scale-dependent factors.

The package is organized into several modules:
- base: Base classes and interfaces
- phase_schedulers: Phase-based learning rate strategies
- layer_schedulers: Layer-wise adaptive strategies
- connection_schedulers: Connection-level adaptive strategies
- unified_manager: Comprehensive management system

DEPRECATED: This entire package is deprecated. Please use the new component-based
implementations in:
- structure_net.components.schedulers - For individual scheduler components
- structure_net.components.orchestrators - For unified adaptive learning rate management
"""

import warnings

warnings.warn(
    "The adaptive_learning_rates package is deprecated. "
    "Please use structure_net.components.schedulers and "
    "structure_net.components.orchestrators instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import new components for compatibility
from ...components.schedulers import (
    MultiScaleLearningScheduler as _MultiScaleLearningScheduler,
    LayerAgeAwareScheduler as _LayerAgeAwareScheduler,
    ExtremaPhaseScheduler as _ExtremaPhaseScheduler
)
from ...components.orchestrators import (
    AdaptiveLearningRateOrchestrator as _AdaptiveLearningRateOrchestrator
)

# Import old implementations
from .base import BaseLearningRateScheduler, LearningRateStrategy
from .phase_schedulers import (
    ExtremaPhaseScheduler,
    GrowthPhaseScheduler,
    ExponentialBackoffScheduler,
    WarmupScheduler,
    CosineAnnealingScheduler,
    AdaptivePhaseScheduler
)
from .layer_schedulers import (
    LayerAgeAwareLR,
    CascadingDecayScheduler,
    LayerwiseAdaptiveRates,
    ProgressiveFreezingScheduler,
    AgeBasedScheduler as LayerAgeBasedScheduler,
    ComponentSpecificScheduler,
    PretrainedNewLayerScheduler,
    LARSScheduler,
    SedimentaryLearningScheduler
)
from .connection_schedulers import (
    MultiScaleLearning,
    SoftClampingScheduler,
    SparsityAwareScheduler,
    AgeBasedScheduler,
    ScaleDependentRates,
    ConnectionStrengthScheduler,
    GradientBasedScheduler,
    ExtremaProximityScheduler
)
from .unified_manager import AdaptiveLearningRateManager, UnifiedAdaptiveLearning
from .factory import (
    create_adaptive_manager, 
    create_adaptive_training_loop,
    create_basic_manager,
    create_advanced_manager,
    create_comprehensive_manager,
    create_ultimate_manager,
    create_custom_manager,
    create_preset_manager,
    create_structure_net_manager,
    create_transfer_learning_manager,
    create_continual_learning_manager,
    create_scheduler_presets
)

# Provide migration guidance
def _migration_guide():
    """Print migration guide for users."""
    print("""
    === Adaptive Learning Rate Migration Guide ===
    
    The adaptive_learning_rates package has been refactored into components.
    
    Migration examples:
    
    1. MultiScaleLearning -> MultiScaleLearningScheduler
       OLD: from structure_net.evolution.adaptive_learning_rates import MultiScaleLearning
       NEW: from structure_net.components.schedulers import MultiScaleLearningScheduler
    
    2. LayerAgeAwareLR -> LayerAgeAwareScheduler
       OLD: from structure_net.evolution.adaptive_learning_rates import LayerAgeAwareLR
       NEW: from structure_net.components.schedulers import LayerAgeAwareScheduler
    
    3. ExtremaPhaseScheduler -> ExtremaPhaseScheduler (component version)
       OLD: from structure_net.evolution.adaptive_learning_rates import ExtremaPhaseScheduler
       NEW: from structure_net.components.schedulers import ExtremaPhaseScheduler
    
    4. UnifiedAdaptiveLearning -> AdaptiveLearningRateOrchestrator
       OLD: from structure_net.evolution.adaptive_learning_rates import UnifiedAdaptiveLearning
       NEW: from structure_net.components.orchestrators import AdaptiveLearningRateOrchestrator
    
    The new components offer:
    - Better modularity and testability
    - Formal contracts defining inputs/outputs
    - Resource requirement declarations
    - Improved state management
    """)

__all__ = [
    # Base classes
    'BaseLearningRateScheduler',
    'LearningRateStrategy',
    
    # Phase schedulers
    'ExtremaPhaseScheduler',
    'GrowthPhaseScheduler', 
    'ExponentialBackoffScheduler',
    'WarmupScheduler',
    'CosineAnnealingScheduler',
    'AdaptivePhaseScheduler',
    
    # Layer schedulers
    'LayerAgeAwareLR',
    'CascadingDecayScheduler',
    'LayerwiseAdaptiveRates',
    'ProgressiveFreezingScheduler',
    'LayerAgeBasedScheduler',
    'ComponentSpecificScheduler',
    'PretrainedNewLayerScheduler',
    'LARSScheduler',
    'SedimentaryLearningScheduler',
    
    # Connection schedulers
    'MultiScaleLearning',
    'SoftClampingScheduler',
    'SparsityAwareScheduler',
    'AgeBasedScheduler',
    'ScaleDependentRates',
    'ConnectionStrengthScheduler',
    'GradientBasedScheduler',
    'ExtremaProximityScheduler',
    
    # Unified system
    'AdaptiveLearningRateManager',
    'UnifiedAdaptiveLearning',
    
    # Factory functions
    'create_adaptive_manager',
    'create_adaptive_training_loop',
    'create_basic_manager',
    'create_advanced_manager',
    'create_comprehensive_manager',
    'create_ultimate_manager',
    'create_custom_manager',
    'create_preset_manager',
    'create_structure_net_manager',
    'create_transfer_learning_manager',
    'create_continual_learning_manager',
    'create_scheduler_presets',
    
    # Migration helper
    '_migration_guide'
]