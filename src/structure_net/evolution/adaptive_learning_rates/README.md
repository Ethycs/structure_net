# Adaptive Learning Rates for Structure Net

This package provides a comprehensive, modular system for adaptive learning rate strategies in neural network training, specifically designed for Structure Net's dynamic growth capabilities.

## Overview

The adaptive learning rates system has been completely refactored from a single monolithic file into a well-organized, modular package with clear separation of concerns:

- **Base classes and interfaces** for consistent API design
- **Phase-based schedulers** that adapt to training phases
- **Layer-wise schedulers** that consider network depth and architecture
- **Connection-level schedulers** for fine-grained control
- **Unified management system** that combines multiple strategies
- **Factory functions** for easy creation and configuration

## Quick Start

```python
from src.structure_net.evolution.adaptive_learning_rates import (
    create_adaptive_manager,
    create_adaptive_training_loop
)

# Create a network
network = MyNetwork()

# Create adaptive learning rate manager
manager = create_adaptive_manager(
    network=network,
    base_lr=0.001,
    strategy='comprehensive'  # or 'basic', 'advanced', 'ultimate'
)

# Use in training loop
trained_network, history = create_adaptive_training_loop(
    network=network,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    strategy='comprehensive'
)
```

## Architecture

### Base Classes (`base.py`)

- `BaseLearningRateScheduler`: Abstract base for all schedulers
- `BasePhaseScheduler`: Base for phase-based strategies
- `BaseLayerScheduler`: Base for layer-wise strategies  
- `BaseConnectionScheduler`: Base for connection-level strategies
- `LearningRateStrategy`: Enum for predefined strategies
- `AdaptiveOptimizerWrapper`: Wrapper for PyTorch optimizers

### Phase Schedulers (`phase_schedulers.py`)

Adapt learning rates based on training phases:

- `ExtremaPhaseScheduler`: Uses extrema patterns to detect growth phases
- `GrowthPhaseScheduler`: Early/middle/late phase adaptation
- `ExponentialBackoffScheduler`: Aggressive early → gentle late
- `WarmupScheduler`: Gradual warm-up for new components
- `CosineAnnealingScheduler`: Smooth cosine decay
- `AdaptivePhaseScheduler`: Multi-signal phase detection

### Layer Schedulers (`layer_schedulers.py`)

Adapt learning rates based on layer properties:

- `LayerAgeAwareLR`: Combines layer depth with connection age
- `CascadingDecayScheduler`: Exponential decay by depth
- `LayerwiseAdaptiveRates`: Different rates for early/middle/late layers
- `ProgressiveFreezingScheduler`: Gradually freeze layers over time
- `ComponentSpecificScheduler`: Different rates for different components
- `PretrainedNewLayerScheduler`: Transfer learning optimization
- `LARSScheduler`: Layer-wise Adaptive Rate Scaling
- `SedimentaryLearningScheduler`: Geological-inspired stratification

### Connection Schedulers (`connection_schedulers.py`)

Fine-grained control at the connection level:

- `MultiScaleLearning`: Scale-dependent rates based on birth time
- `SoftClampingScheduler`: Gradual freezing instead of hard clamps
- `SparsityAwareScheduler`: Adapt based on connection density
- `AgeBasedScheduler`: Older connections learn slower
- `ScaleDependentRates`: Coarse/medium/fine scale adaptation
- `ConnectionStrengthScheduler`: Adapt based on weight magnitudes
- `GradientBasedScheduler`: Adapt based on gradient patterns
- `ExtremaProximityScheduler`: Boost learning near extrema

### Unified Manager (`unified_manager.py`)

- `AdaptiveLearningRateManager`: Main interface combining all strategies
- `UnifiedAdaptiveLearning`: Ultimate system combining all techniques

### Factory Functions (`factory.py`)

Convenient creation functions:

- `create_adaptive_manager()`: Main factory with strategy presets
- `create_basic_manager()`: Essential strategies only
- `create_advanced_manager()`: With extrema detection
- `create_comprehensive_manager()`: Most features enabled
- `create_ultimate_manager()`: All features enabled
- `create_preset_manager()`: Use predefined configurations
- `create_adaptive_training_loop()`: Complete training setup

## Strategies

### Basic Strategy
Essential adaptive learning rate techniques:
- Exponential backoff
- Layer-wise adaptive rates
- Soft clamping
- Scale-dependent rates
- Phase-based adjustment

### Advanced Strategy
Adds sophisticated detection:
- Extrema phase detection
- Layer-age aware learning
- All basic features

### Comprehensive Strategy
Most features enabled:
- Multi-scale learning
- Advanced connection tracking
- All advanced features

### Ultimate Strategy
Everything enabled:
- Unified adaptive system
- All available schedulers
- Maximum sophistication

## Presets

Pre-configured scheduler combinations:

- **Conservative**: Slow, stable learning
- **Aggressive**: Fast, dynamic adaptation
- **Balanced**: Middle ground approach
- **Extrema Focused**: Optimized for extrema detection
- **Fine Tuning**: For refinement phases

## Usage Examples

### Basic Usage

```python
# Create manager with basic strategy
manager = create_basic_manager(network, base_lr=0.001)

# Create optimizer
optimizer = manager.create_adaptive_optimizer()

# Training loop
for epoch in range(epochs):
    manager.update_learning_rates(optimizer, epoch)
    # ... training code ...
```

### Custom Configuration

```python
# Custom scheduler configurations
custom_configs = {
    'exponential_backoff': {
        'initial_lr': 0.5,
        'decay_rate': 0.92
    },
    'layerwise_rates': {
        'early_rate': 0.005,
        'middle_rate': 0.01,
        'late_rate': 0.002
    }
}

manager = create_adaptive_manager(
    network=network,
    base_lr=0.001,
    strategy='basic',
    custom_configs=custom_configs
)
```

### Preset Usage

```python
# Use a preset configuration
manager = create_preset_manager(
    network=network,
    preset_name='aggressive',
    base_lr=0.001
)
```

### Individual Schedulers

```python
from src.structure_net.evolution.adaptive_learning_rates import (
    ExponentialBackoffScheduler,
    LayerwiseAdaptiveRates
)

# Use individual schedulers
backoff = ExponentialBackoffScheduler(base_lr=0.001)
layerwise = LayerwiseAdaptiveRates(base_lr=0.001, total_layers=5)

for epoch in range(epochs):
    backoff.update_epoch(epoch)
    lr = backoff.get_learning_rate()
    # Use lr...
```

## Integration with Structure Net

The adaptive learning rates system is designed to work seamlessly with Structure Net's dynamic growth:

```python
from src.structure_net.evolution.adaptive_learning_rates import (
    create_structure_net_manager
)

# Optimized for Structure Net
manager = create_structure_net_manager(
    network=structure_net,
    base_lr=0.001,
    enable_extrema=True
)
```

## Advanced Features

### Extrema Detection Integration

When extrema detection is available, the system can automatically adjust learning rates based on network extrema patterns:

```python
# Extrema-aware training
manager.update_learning_rates(
    optimizer, 
    epoch,
    network=network,
    data_loader=train_loader,
    device='cuda'
)
```

### Connection-Level Tracking

Track individual connections for fine-grained control:

```python
# Register new connections
manager.unified_system.register_new_connection("layer_2_conn_5", layer_idx=2)

# Update connection ages
manager.unified_system.update_connection_age("layer_2_conn_5")
```

### State Management

Save and load manager state for checkpointing:

```python
# Save state
state = manager.save_state()

# Load state
manager.load_state(state)
```

## Benefits of the Refactored System

1. **Modularity**: Each scheduler type is in its own module
2. **Extensibility**: Easy to add new schedulers
3. **Composability**: Mix and match different strategies
4. **Maintainability**: Clear separation of concerns
5. **Testability**: Individual components can be tested in isolation
6. **Documentation**: Each module is well-documented
7. **Type Safety**: Comprehensive type hints throughout
8. **Performance**: Optimized for minimal overhead

## Migration from Old System

The old monolithic `adaptive_learning_rates.py` file has been replaced with this modular system. Key changes:

- Import paths have changed to use the new package structure
- Factory functions provide easier creation
- Strategy enums replace string-based configuration
- Better error handling and validation
- Comprehensive configuration options

## Examples

See `examples/refactored_adaptive_learning_rates_example.py` for comprehensive usage examples demonstrating all features of the refactored system.

## Contributing

When adding new schedulers:

1. Inherit from the appropriate base class
2. Implement required abstract methods
3. Add comprehensive docstrings
4. Include configuration validation
5. Add to the appropriate module
6. Update `__init__.py` imports
7. Add factory function if needed
8. Include usage examples

## Performance Considerations

- Schedulers are designed for minimal computational overhead
- Connection-level tracking scales with network size
- Use appropriate strategies for your use case
- Monitor memory usage with large networks
- Consider disabling unused features for production

## Future Enhancements

Planned improvements:
- GPU acceleration for large-scale tracking
- Automatic hyperparameter tuning
- Integration with popular training frameworks
- Advanced visualization tools
- Distributed training support
