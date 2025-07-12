# Integrated Growth System Migration Guide

This guide helps you migrate from the old hardcoded `IntegratedGrowthSystem` to the new composable evolution architecture.

## Overview

The `IntegratedGrowthSystem` has been completely refactored to use the new composable evolution architecture as its backend. This provides:

- **Backward Compatibility**: Your existing code continues to work without changes
- **Better Performance**: Optimized components and reduced overhead
- **Enhanced Modularity**: Mix and match evolution components
- **Future-Proof Design**: Easy to extend and customize

## Migration Strategies

### Strategy 1: No Changes Required (Recommended for Quick Migration)

Your existing code will automatically use the new composable backend:

```python
# This code continues to work exactly as before
from structure_net.evolution.integrated_growth_system import IntegratedGrowthSystem

system = IntegratedGrowthSystem(network, config)
grown_network = system.grow_network(train_loader, val_loader, growth_iterations=3)
```

**What happens**: The system now uses `integrated_growth_system_v2.py` which delegates to the composable system while maintaining the exact same API.

### Strategy 2: Gradual Migration (Recommended for New Features)

Start using the new composable API for new code:

```python
# New composable approach
from structure_net.evolution.components import (
    create_standard_evolution_system,
    NetworkContext
)

def new_evolution_approach(network, train_loader, val_loader, device):
    system = create_standard_evolution_system()
    context = NetworkContext(network, train_loader, device)
    evolved_context = system.evolve_network(context, num_iterations=3)
    return evolved_context.network
```

### Strategy 3: Full Migration (Recommended for Maximum Benefits)

Completely migrate to the composable system:

```python
# Before
from structure_net.evolution.integrated_growth_system import IntegratedGrowthSystem
from structure_net.evolution.advanced_layers import ThresholdConfig

def old_experiment():
    config = ThresholdConfig()
    system = IntegratedGrowthSystem(network, config)
    grown_network = system.grow_network(train_loader, val_loader)
    return grown_network

# After
from structure_net.evolution.components import (
    ComposableEvolutionSystem,
    NetworkContext,
    StandardExtremaAnalyzer,
    ExtremaGrowthStrategy,
    InformationFlowGrowthStrategy
)

def new_experiment():
    # Create custom system
    system = ComposableEvolutionSystem()
    system.add_component(StandardExtremaAnalyzer(max_batches=5))
    system.add_component(ExtremaGrowthStrategy(extrema_threshold=0.25))
    system.add_component(InformationFlowGrowthStrategy())
    
    # Run evolution
    context = NetworkContext(network, train_loader, device)
    evolved_context = system.evolve_network(context, num_iterations=3)
    return evolved_context.network
```

## Component Mapping

### Old Tournament Strategies → New Components

| Old Tournament Strategy | New Component | Configuration |
|------------------------|---------------|---------------|
| "Add Layer at Bottleneck" | `InformationFlowGrowthStrategy` | `bottleneck_threshold=0.1` |
| "Add Patches to Extrema" | `ExtremaGrowthStrategy` | `extrema_threshold=0.3` |
| "Add 2-Layer Residual Block" | `ResidualBlockGrowthStrategy` | `num_layers=2` |
| "Add 3-Layer Residual Block" | `ResidualBlockGrowthStrategy` | `num_layers=3` |
| "Hybrid: Add Layer & Patches" | `HybridGrowthStrategy` | Combines multiple strategies |

### Old Configuration → New Configuration

```python
# Old configuration
from structure_net.evolution.advanced_layers import ThresholdConfig, MetricsConfig

threshold_config = ThresholdConfig(
    activation_threshold=0.01,
    gradient_threshold=1e-6,
    adaptive=True
)

# New configuration (component-specific)
from structure_net.evolution.components import StandardExtremaAnalyzer

analyzer = StandardExtremaAnalyzer()
analyzer.configure({
    'dead_threshold': 0.01,        # equivalent to activation_threshold
    'saturated_multiplier': 2.5,   # adaptive threshold calculation
    'max_batches': 5               # analysis depth
})
```

## Migration Tools

### Automatic Migration Helper

Use the migration helper to analyze your code:

```python
from structure_net.evolution.migration_helper import MigrationHelper

helper = MigrationHelper()
helper.analyze_existing_code("my_experiment.py")
helper.suggest_migration()
helper.print_migration_report()
```

### Quick Migration Check

```python
from structure_net.evolution.migration_helper import quick_migration_check

result = quick_migration_check("my_experiment.py")
if result['needs_migration']:
    print(f"Migration complexity: {result['complexity']}")
    print(f"Suggestions available: {result['suggestions_count']}")
```

## Performance Comparison

The migration helper can compare performance between old and new systems:

```python
helper = MigrationHelper()
comparison = helper.compare_performance(
    network, train_loader, val_loader, 
    config=config, iterations=2
)

print(f"Time improvement: {comparison['comparison']['time_improvement']:+.1f}%")
print(f"Accuracy difference: {comparison['comparison']['accuracy_difference']:+.3f}")
```

## Common Migration Patterns

### Pattern 1: Simple Growth Experiment

```python
# Before
system = IntegratedGrowthSystem(network, config)
grown_network = system.grow_network(train_loader, val_loader, growth_iterations=3)

# After (equivalent)
system = create_standard_evolution_system()
context = NetworkContext(network, train_loader, device)
evolved_context = system.evolve_network(context, num_iterations=3)
grown_network = evolved_context.network
```

### Pattern 2: Custom Tournament

```python
# Before
tournament = ParallelGrowthTournament(network, threshold_config, metrics_config)
results = tournament.run_tournament(train_loader, val_loader)
winner = results['winner']

# After
system = ComposableEvolutionSystem()
system.add_component(StandardExtremaAnalyzer())
system.add_component(ExtremaGrowthStrategy())
system.add_component(InformationFlowGrowthStrategy())

context = NetworkContext(network, train_loader, device)
evolved_context = system.evolve_network(context, num_iterations=1)
```

### Pattern 3: Threshold Management

```python
# Before
threshold_manager = AdaptiveThresholdManager(threshold_config)
stats = threshold_manager.compute_network_stats(network, data_loader)
threshold_manager.update_thresholds(stats)

# After (automatic)
# Threshold management is now handled automatically by individual components
# Configure thresholds through component configuration:
analyzer = StandardExtremaAnalyzer()
analyzer.configure({'dead_threshold': 0.005})  # More sensitive
```

## Benefits of Migration

### Immediate Benefits (No Code Changes)

- ✅ **Better Performance**: Optimized backend with reduced overhead
- ✅ **Enhanced Monitoring**: Built-in metrics for all components
- ✅ **Improved Reliability**: Better error handling and validation

### Full Migration Benefits

- ✅ **Modularity**: Mix and match components for custom evolution strategies
- ✅ **Configurability**: Fine-tune each component individually
- ✅ **Extensibility**: Easy to add new analyzers and strategies
- ✅ **Testability**: Test components in isolation
- ✅ **Future-Proof**: Architecture designed for research evolution

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # If you get import errors, ensure you're using the right module
   from structure_net.evolution.integrated_growth_system import IntegratedGrowthSystem  # Works
   from structure_net.evolution.components import create_standard_evolution_system     # New way
   ```

2. **Performance Differences**
   ```python
   # If performance differs, check component configuration
   system = create_standard_evolution_system()
   config = system.get_configuration()
   print("Current configuration:", config)
   ```

3. **Missing Features**
   ```python
   # If you need specific old tournament behavior, create custom components
   class CustomStrategy(GrowthStrategy):
       def analyze_growth_potential(self, context):
           # Your custom logic here
           pass
   ```

### Getting Help

1. **Use Migration Helper**: Automatic analysis and suggestions
2. **Check Documentation**: `docs/composable_evolution_guide.md`
3. **Run Examples**: `examples/composable_evolution_example.py`
4. **Performance Comparison**: Use migration helper's comparison tools

## Migration Checklist

### Phase 1: Assessment
- [ ] Run migration helper on your codebase
- [ ] Review migration complexity assessment
- [ ] Identify critical code paths

### Phase 2: Testing
- [ ] Test existing code with new backend (no changes needed)
- [ ] Compare performance between old and new systems
- [ ] Validate results match expectations

### Phase 3: Migration
- [ ] Update imports for new code
- [ ] Migrate to composable API gradually
- [ ] Configure components for your specific needs
- [ ] Update documentation and examples

### Phase 4: Optimization
- [ ] Fine-tune component configurations
- [ ] Add custom components if needed
- [ ] Monitor performance improvements
- [ ] Share learnings with team

## Example: Complete Migration

Here's a complete example showing before and after:

```python
#!/usr/bin/env python3
"""
Complete Migration Example
"""

# ============================================================================
# BEFORE: Old Hardcoded System
# ============================================================================

def old_approach():
    from structure_net.evolution.integrated_growth_system import IntegratedGrowthSystem
    from structure_net.evolution.advanced_layers import ThresholdConfig, MetricsConfig
    
    # Configuration
    threshold_config = ThresholdConfig(
        activation_threshold=0.01,
        gradient_threshold=1e-6,
        adaptive=True
    )
    metrics_config = MetricsConfig()
    
    # Create system
    system = IntegratedGrowthSystem(network, threshold_config, metrics_config)
    
    # Run growth
    grown_network = system.grow_network(
        train_loader, val_loader, 
        growth_iterations=3,
        epochs_per_iteration=20
    )
    
    return grown_network

# ============================================================================
# AFTER: New Composable System
# ============================================================================

def new_approach():
    from structure_net.evolution.components import (
        ComposableEvolutionSystem,
        NetworkContext,
        StandardExtremaAnalyzer,
        NetworkStatsAnalyzer,
        ExtremaGrowthStrategy,
        InformationFlowGrowthStrategy,
        StandardNetworkTrainer
    )
    
    # Create composable system
    system = ComposableEvolutionSystem()
    
    # Add analyzers
    extrema_analyzer = StandardExtremaAnalyzer()
    extrema_analyzer.configure({
        'dead_threshold': 0.01,
        'saturated_multiplier': 2.5,
        'max_batches': 5
    })
    system.add_component(extrema_analyzer)
    system.add_component(NetworkStatsAnalyzer())
    
    # Add growth strategies
    extrema_strategy = ExtremaGrowthStrategy()
    extrema_strategy.configure({
        'extrema_threshold': 0.3,
        'dead_neuron_threshold': 5
    })
    system.add_component(extrema_strategy)
    system.add_component(InformationFlowGrowthStrategy())
    
    # Add trainer
    trainer = StandardNetworkTrainer()
    trainer.configure({'learning_rate': 0.001})
    system.add_component(trainer)
    
    # Create context and run evolution
    device = next(network.parameters()).device
    context = NetworkContext(network, train_loader, device)
    evolved_context = system.evolve_network(context, num_iterations=3)
    
    return evolved_context.network

# ============================================================================
# MIGRATION BENEFITS COMPARISON
# ============================================================================

def compare_approaches():
    """Compare old vs new approaches."""
    
    print("OLD APPROACH:")
    print("✓ Simple API")
    print("✗ Hardcoded strategies")
    print("✗ Limited configurability")
    print("✗ Difficult to extend")
    print("✗ Monolithic architecture")
    
    print("\nNEW APPROACH:")
    print("✓ Modular components")
    print("✓ Individual configuration")
    print("✓ Easy to extend")
    print("✓ Better monitoring")
    print("✓ Future-proof design")
    print("✓ Backward compatible")

if __name__ == "__main__":
    compare_approaches()
```

## Conclusion

The migration to the composable evolution system provides significant benefits while maintaining full backward compatibility. You can:

1. **Start immediately** with no code changes (automatic backend migration)
2. **Migrate gradually** by using new APIs for new features
3. **Fully migrate** to gain maximum benefits of the modular architecture

The new system is designed to grow with your research needs while providing a solid, tested foundation for network evolution experiments.
