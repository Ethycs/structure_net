# Adaptive Learning Rates Migration Guide

## Overview

The monolithic `adaptive_learning_rates.py` file (1000+ lines) has been **deprecated** and replaced with a modular package for better maintainability, composability, and future extensibility.

## 🚨 Migration Required

### Quick Migration

**OLD (Deprecated):**
```python
from src.structure_net.evolution.adaptive_learning_rates import AdaptiveLearningRateManager
```

**NEW (Recommended):**
```python
from src.structure_net.evolution.adaptive_learning_rates import AdaptiveLearningRateManager
```

> **Note:** The import path is the same, but now it imports from the modular package instead of the monolithic file.

## 📁 New Modular Structure

The new system is organized into focused modules:

```
adaptive_learning_rates/
├── __init__.py              # Main exports and API
├── base.py                  # Base classes and interfaces
├── phase_schedulers.py      # Phase-based strategies
├── layer_schedulers.py      # Layer-wise strategies  
├── connection_schedulers.py # Connection-level strategies
├── unified_manager.py       # Comprehensive management
├── factory.py              # Easy setup functions
└── README.md               # Package documentation
```

## 🔄 Migration Examples

### 1. Basic Manager Setup

**OLD WAY:**
```python
from src.structure_net.evolution.adaptive_learning_rates import AdaptiveLearningRateManager

manager = AdaptiveLearningRateManager(
    network=network,
    base_lr=0.001,
    enable_exponential_backoff=True,
    enable_layerwise_rates=True,
    enable_soft_clamping=True,
    enable_scale_dependent=True,
    enable_phase_based=True
)
```

**NEW WAY (Recommended):**
```python
from src.structure_net.evolution.adaptive_learning_rates import create_comprehensive_manager

# Much simpler setup with same functionality
manager = create_comprehensive_manager(network, base_lr=0.001)
```

### 2. Custom Configurations

**OLD WAY:**
```python
from src.structure_net.evolution.adaptive_learning_rates import (
    ExtremaPhaseScheduler,
    LayerAgeAwareLR,
    MultiScaleLearning,
    AdaptiveLearningRateManager
)

# Manual configuration was complex
manager = AdaptiveLearningRateManager(
    network=network,
    base_lr=0.001,
    enable_extrema_phase=True,
    enable_layer_age_aware=True,
    enable_multi_scale=True
)
```

**NEW WAY (Modular):**
```python
from src.structure_net.evolution.adaptive_learning_rates import (
    ExtremaPhaseScheduler,
    LayerAgeAwareLR, 
    MultiScaleLearning,
    create_custom_manager
)

# Compose exactly what you need
schedulers = [
    ExtremaPhaseScheduler(explosive_threshold=0.1),
    LayerAgeAwareLR(decay_constant=50.0),
    MultiScaleLearning()
]

manager = create_custom_manager(
    network=network,
    schedulers=schedulers,
    base_lr=0.001
)
```

### 3. Factory Functions

The new system provides convenient factory functions:

```python
from src.structure_net.evolution.adaptive_learning_rates import (
    create_basic_manager,           # Basic strategies only
    create_advanced_manager,        # Basic + advanced features
    create_comprehensive_manager,   # Most strategies enabled
    create_ultimate_manager,        # Everything enabled
    create_structure_net_manager,   # Optimized for structure_net
    create_transfer_learning_manager, # For transfer learning
    create_continual_learning_manager # For continual learning
)

# Choose the right level for your needs
manager = create_structure_net_manager(network, base_lr=0.001)
```

### 4. Training Loop Integration

**OLD WAY:**
```python
from src.structure_net.evolution.adaptive_learning_rates import create_adaptive_training_loop

# Function existed but was monolithic
network, history = create_adaptive_training_loop(
    network, train_loader, val_loader, epochs=50
)
```

**NEW WAY (Enhanced):**
```python
from src.structure_net.evolution.adaptive_learning_rates import create_adaptive_training_loop

# Same interface, but now modular and more configurable
network, history = create_adaptive_training_loop(
    network=network,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    strategy='comprehensive',  # or 'basic', 'advanced', 'ultimate'
    base_lr=0.001
)
```

## 🎯 Benefits of New System

### 1. **Better Organization**
- **Before:** 1000+ lines in one file
- **After:** Organized into focused modules (~200 lines each)

### 2. **Composable Architecture**
- **Before:** Monolithic configuration with boolean flags
- **After:** Mix and match specific components

### 3. **Enhanced Functionality**
- **Before:** Fixed set of strategies
- **After:** Easy to add new strategies and combinations

### 4. **Easier Testing**
- **Before:** Hard to test individual strategies
- **After:** Each component can be tested in isolation

### 5. **Future-Proof**
- **Before:** Adding features required modifying large file
- **After:** Add new modules without touching existing code

## 📋 Migration Checklist

### Phase 1: Immediate (Backward Compatible)
- [ ] Update imports to use modular package
- [ ] Test that existing code still works
- [ ] No functional changes required

### Phase 2: Optimization (Recommended)
- [ ] Replace manual configuration with factory functions
- [ ] Use `create_comprehensive_manager()` for most cases
- [ ] Use `create_custom_manager()` for specific needs

### Phase 3: Advanced (Optional)
- [ ] Leverage new modular components
- [ ] Create custom scheduler combinations
- [ ] Integrate with composable evolution system

## 🔧 Specific Component Migrations

### ExtremaPhaseScheduler
```python
# OLD: Part of monolithic system
from src.structure_net.evolution.adaptive_learning_rates import ExtremaPhaseScheduler

# NEW: Dedicated module
from src.structure_net.evolution.adaptive_learning_rates.phase_schedulers import ExtremaPhaseScheduler
# OR (recommended)
from src.structure_net.evolution.adaptive_learning_rates import ExtremaPhaseScheduler
```

### LayerAgeAwareLR
```python
# OLD: Part of monolithic system  
from src.structure_net.evolution.adaptive_learning_rates import LayerAgeAwareLR

# NEW: Dedicated module
from src.structure_net.evolution.adaptive_learning_rates.layer_schedulers import LayerAgeAwareLR
# OR (recommended)
from src.structure_net.evolution.adaptive_learning_rates import LayerAgeAwareLR
```

### MultiScaleLearning
```python
# OLD: Part of monolithic system
from src.structure_net.evolution.adaptive_learning_rates import MultiScaleLearning

# NEW: Dedicated module
from src.structure_net.evolution.adaptive_learning_rates.connection_schedulers import MultiScaleLearning
# OR (recommended)
from src.structure_net.evolution.adaptive_learning_rates import MultiScaleLearning
```

## 🚀 Advanced Usage

### Custom Scheduler Development
```python
from src.structure_net.evolution.adaptive_learning_rates.base import BaseLearningRateScheduler

class MyCustomScheduler(BaseLearningRateScheduler):
    def __init__(self, custom_param=1.0):
        super().__init__()
        self.custom_param = custom_param
    
    def get_learning_rate(self, context):
        # Your custom logic here
        return context.base_lr * self.custom_param
    
    def configure(self, config):
        self.custom_param = config.get('custom_param', self.custom_param)

# Use in custom manager
from src.structure_net.evolution.adaptive_learning_rates import create_custom_manager

manager = create_custom_manager(
    network=network,
    schedulers=[MyCustomScheduler(custom_param=2.0)],
    base_lr=0.001
)
```

### Integration with Composable Evolution
```python
from src.structure_net.evolution.components import ComposableEvolutionSystem
from src.structure_net.evolution.adaptive_learning_rates import create_structure_net_manager

# Create evolution system
evolution_system = ComposableEvolutionSystem()

# Add adaptive learning rate component
lr_manager = create_structure_net_manager(network, base_lr=0.001)
evolution_system.add_component(lr_manager)

# Now learning rates adapt automatically during evolution
```

## 📊 Performance Comparison

| Aspect | Old Monolithic | New Modular |
|--------|---------------|-------------|
| **File Size** | 1000+ lines | ~200 lines per module |
| **Import Time** | Slow (loads everything) | Fast (loads only needed) |
| **Memory Usage** | High (all strategies loaded) | Low (only used strategies) |
| **Testability** | Hard (monolithic) | Easy (isolated components) |
| **Extensibility** | Difficult | Simple |
| **Maintainability** | Poor | Excellent |

## 🛠️ Troubleshooting

### Import Errors
```python
# If you get import errors, check the path:
try:
    from src.structure_net.evolution.adaptive_learning_rates import AdaptiveLearningRateManager
    print("✅ Modular system imported successfully")
except ImportError:
    print("❌ Modular system not found, using deprecated fallback")
    from src.structure_net.evolution.adaptive_learning_rates_deprecated import AdaptiveLearningRateManager
```

### Configuration Issues
```python
# If old configuration doesn't work:
# OLD (might break)
manager = AdaptiveLearningRateManager(network, enable_all_strategies=True)

# NEW (guaranteed to work)
from src.structure_net.evolution.adaptive_learning_rates import create_comprehensive_manager
manager = create_comprehensive_manager(network, base_lr=0.001)
```

### Performance Issues
```python
# If performance is slower than expected:
# Use specific factory for your use case
from src.structure_net.evolution.adaptive_learning_rates import (
    create_basic_manager,      # Fastest, basic features
    create_advanced_manager,   # Balanced performance/features  
    create_comprehensive_manager, # Full features, slower
)

# Choose based on your needs
manager = create_basic_manager(network, base_lr=0.001)  # For speed
manager = create_comprehensive_manager(network, base_lr=0.001)  # For features
```

## 📅 Migration Timeline

### Phase 1: Backward Compatibility (Current)
- ✅ Monolithic file deprecated but still works
- ✅ Modular system available
- ✅ All existing code continues to work

### Phase 2: Deprecation Warnings (6 months)
- ⚠️ Deprecation warnings when using old system
- 📖 Migration guides and examples provided
- 🔧 Tools to help with migration

### Phase 3: Removal (12 months)
- ❌ Monolithic file removed
- ✅ Only modular system available
- 🚀 Full benefits of new architecture

## 🆘 Getting Help

### Documentation
- **Package README:** `src/structure_net/evolution/adaptive_learning_rates/README.md`
- **API Reference:** Auto-generated from docstrings
- **Examples:** `examples/adaptive_learning_rates_example.py`

### Migration Support
- **Migration Helper:** `show_migration_guide()` function
- **Compatibility Layer:** Automatic fallback to deprecated version
- **Factory Functions:** Easy setup for common use cases

### Community
- **Issues:** Report problems on GitHub
- **Discussions:** Ask questions in discussions
- **Examples:** Check examples directory for patterns

## ✅ Migration Complete!

Once you've migrated to the modular system, you'll have:

- ✅ **Better Performance:** Only load what you need
- ✅ **Easier Debugging:** Isolated components
- ✅ **More Flexibility:** Compose custom solutions
- ✅ **Future-Proof:** Easy to extend and maintain
- ✅ **Better Testing:** Test components individually

The modular adaptive learning rates system is designed to grow with your research needs while maintaining the sophisticated functionality of the original system.
