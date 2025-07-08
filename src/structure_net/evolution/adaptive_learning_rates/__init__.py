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
"""

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
    'create_scheduler_presets'
]
