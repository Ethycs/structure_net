#!/usr/bin/env python3
"""
DEPRECATED: Adaptive Learning Rate Strategies for Structure Net

This module has been DEPRECATED and replaced by the modular adaptive_learning_rates package.

Please use the new modular system instead:

OLD (DEPRECATED):
    from src.structure_net.evolution.adaptive_learning_rates import AdaptiveLearningRateManager
    
NEW (RECOMMENDED):
    from src.structure_net.evolution.adaptive_learning_rates import AdaptiveLearningRateManager

The new modular system provides:
- Better organization and maintainability
- Composable components
- Enhanced functionality
- Easier testing and debugging
- Future-proof architecture

Migration Guide:
1. Replace imports from this module with imports from adaptive_learning_rates package
2. Use factory functions for easy setup: create_adaptive_manager(), create_comprehensive_manager()
3. Leverage the new composable architecture for custom configurations

For detailed migration instructions, see:
docs/adaptive_learning_rates_migration.md
"""

import warnings
from typing import *

# Import everything from the new modular system
from .adaptive_learning_rates import *

# Issue deprecation warning
warnings.warn(
    "adaptive_learning_rates.py is deprecated. "
    "Please use the modular adaptive_learning_rates package instead. "
    "See docs/adaptive_learning_rates_migration.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# For backward compatibility, re-export everything
# This allows existing code to continue working while encouraging migration

__all__ = [
    # Legacy classes (now imported from modular system)
    'ExtremaPhaseScheduler',
    'LayerAgeAwareLR', 
    'MultiScaleLearning',
    'CascadingDecayScheduler',
    'AgeBasedScheduler',
    'ComponentSpecificScheduler',
    'PretrainedNewLayerScheduler',
    'GrowthAwareScheduler',
    'WarmupScheduler',
    'LARSScheduler',
    'ProgressiveFreezingScheduler',
    'SparsityAwareScheduler',
    'SedimentaryLearningScheduler',
    'UnifiedAdaptiveLearning',
    'ExponentialBackoffScheduler',
    'LayerwiseAdaptiveRates',
    'SoftClampingScheduler',
    'ScaleDependentRates',
    'GrowthPhaseScheduler',
    'AdaptiveLearningRateManager',
    
    # Legacy functions (now imported from modular system)
    'create_optimal_lr_schedule',
    'differential_decay_after_events',
    'create_comprehensive_adaptive_manager',
    'create_adaptive_training_loop',
]

# Legacy function aliases for backward compatibility
def create_optimal_lr_schedule(*args, **kwargs):
    """DEPRECATED: Use create_adaptive_manager() from adaptive_learning_rates package."""
    warnings.warn("create_optimal_lr_schedule is deprecated. Use create_adaptive_manager() instead.", 
                  DeprecationWarning, stacklevel=2)
    from .adaptive_learning_rates import create_adaptive_manager
    return create_adaptive_manager(*args, **kwargs)

def create_comprehensive_adaptive_manager(*args, **kwargs):
    """DEPRECATED: Use create_comprehensive_manager() from adaptive_learning_rates package."""
    warnings.warn("create_comprehensive_adaptive_manager is deprecated. Use create_comprehensive_manager() instead.", 
                  DeprecationWarning, stacklevel=2)
    from .adaptive_learning_rates import create_comprehensive_manager
    return create_comprehensive_manager(*args, **kwargs)

def differential_decay_after_events(*args, **kwargs):
    """DEPRECATED: Use GrowthAwareScheduler from adaptive_learning_rates package."""
    warnings.warn("differential_decay_after_events is deprecated. Use GrowthAwareScheduler instead.", 
                  DeprecationWarning, stacklevel=2)
    from .adaptive_learning_rates import GrowthAwareScheduler
    scheduler = GrowthAwareScheduler()
    return scheduler.adjust_lr_after_growth(*args, **kwargs)
