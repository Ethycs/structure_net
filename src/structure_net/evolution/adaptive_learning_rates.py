#!/usr/bin/env python3
"""
DEPRECATED: Adaptive Learning Rate Strategies for Structure Net

⚠️  THIS FILE HAS BEEN DEPRECATED ⚠️

This monolithic module has been replaced by the modular adaptive_learning_rates package.

MIGRATION REQUIRED:

OLD (DEPRECATED):
    from ..evolution.adaptive_learning_rates import AdaptiveLearningRateManager
    
NEW (RECOMMENDED):
    from ..evolution.adaptive_learning_rates import AdaptiveLearningRateManager

The new modular system provides:
✅ Better organization and maintainability
✅ Composable components  
✅ Enhanced functionality
✅ Easier testing and debugging
✅ Future-proof architecture

QUICK MIGRATION EXAMPLES:

# OLD WAY (still works but deprecated)
from ..evolution.adaptive_learning_rates import AdaptiveLearningRateManager
manager = AdaptiveLearningRateManager(network, base_lr=0.001)

# NEW WAY (recommended)
from ..evolution.adaptive_learning_rates import create_comprehensive_manager
manager = create_comprehensive_manager(network, base_lr=0.001)

# CUSTOM CONFIGURATIONS
from ..evolution.adaptive_learning_rates import (
    ExtremaPhaseScheduler, LayerAgeAwareLR, MultiScaleLearning
)

For detailed migration instructions, see:
docs/adaptive_learning_rates_migration.md

The original 1000+ line monolithic implementation has been moved to:
src/structure_net/evolution/adaptive_learning_rates_deprecated.py
"""

import warnings
from typing import *

# Issue strong deprecation warning
warnings.warn(
    "\n" + "="*80 + "\n"
    "⚠️  DEPRECATION WARNING ⚠️\n"
    "The monolithic adaptive_learning_rates.py has been DEPRECATED.\n"
    "Please migrate to the modular adaptive_learning_rates package.\n"
    "\n"
    "Quick fix: Replace imports with:\n"
    "from ..evolution.adaptive_learning_rates import ...\n"
    "\n"
    "See docs/adaptive_learning_rates_migration.md for full guide.\n"
    "="*80,
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new modular system for backward compatibility
try:
    from .adaptive_learning_rates import *
    print("✅ Successfully imported from modular adaptive_learning_rates package")
except ImportError as e:
    print(f"❌ Failed to import from modular package: {e}")
    print("Falling back to deprecated implementation...")
    
    # Fallback to deprecated implementation if modular version not available
    from .adaptive_learning_rates_deprecated import *

# Re-export everything for backward compatibility
__all__ = [
    # Core classes
    'AdaptiveLearningRateManager',
    'UnifiedAdaptiveLearning',
    
    # Phase schedulers
    'ExtremaPhaseScheduler',
    'GrowthPhaseScheduler', 
    'ExponentialBackoffScheduler',
    'WarmupScheduler',
    
    # Layer schedulers
    'LayerAgeAwareLR',
    'CascadingDecayScheduler',
    'LayerwiseAdaptiveRates',
    'ProgressiveFreezingScheduler',
    'AgeBasedScheduler',
    'ComponentSpecificScheduler',
    'PretrainedNewLayerScheduler',
    'LARSScheduler',
    'SedimentaryLearningScheduler',
    
    # Connection schedulers
    'MultiScaleLearning',
    'SoftClampingScheduler',
    'SparsityAwareScheduler',
    'ScaleDependentRates',
    
    # Factory functions
    'create_adaptive_manager',
    'create_comprehensive_manager',
    'create_adaptive_training_loop',
    
    # Legacy functions (deprecated)
    'create_optimal_lr_schedule',
    'create_comprehensive_adaptive_manager',
    'differential_decay_after_events',
]

def show_migration_guide():
    """Show detailed migration guide."""
    print("""
🔄 ADAPTIVE LEARNING RATES MIGRATION GUIDE
==========================================

The monolithic adaptive_learning_rates.py (1000+ lines) has been replaced 
with a modular package for better maintainability and composability.

STEP 1: Update Imports
---------------------
OLD:
    from ..evolution.adaptive_learning_rates import AdaptiveLearningRateManager
    
NEW:
    from ..evolution.adaptive_learning_rates import AdaptiveLearningRateManager

STEP 2: Use Factory Functions (Recommended)
------------------------------------------
Instead of manually configuring the manager:

OLD:
    manager = AdaptiveLearningRateManager(
        network=network,
        base_lr=0.001,
        enable_exponential_backoff=True,
        enable_layerwise_rates=True,
        # ... many parameters
    )

NEW:
    from ..evolution.adaptive_learning_rates import create_comprehensive_manager
    manager = create_comprehensive_manager(network, base_lr=0.001)

STEP 3: Custom Configurations
----------------------------
For custom setups, import specific components:

    from ..evolution.adaptive_learning_rates import (
        ExtremaPhaseScheduler,
        LayerAgeAwareLR,
        MultiScaleLearning,
        create_custom_manager
    )
    
    manager = create_custom_manager(
        network=network,
        schedulers=[ExtremaPhaseScheduler(), LayerAgeAwareLR()],
        base_lr=0.001
    )

BENEFITS OF NEW SYSTEM:
✅ Modular components (easier to test and debug)
✅ Composable architecture (mix and match strategies)
✅ Better organization (separate files for different concerns)
✅ Enhanced functionality (new strategies and combinations)
✅ Future-proof (easy to add new research directions)

MIGRATION TIMELINE:
- Phase 1: Backward compatibility maintained (current)
- Phase 2: Deprecation warnings (6 months)
- Phase 3: Remove deprecated file (12 months)

For questions, see: docs/adaptive_learning_rates_migration.md
""")

# Show migration guide when imported
if __name__ != "__main__":
    print("\n📖 For migration guide, call: show_migration_guide()")
