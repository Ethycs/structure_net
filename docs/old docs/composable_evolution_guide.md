# Composable Evolution System Guide

The Structure Net project now features a completely redesigned, interface-based composable evolution system that eliminates hardcoded strategies and enables flexible, modular network evolution.

## Overview

### What Changed

**Before (Hardcoded):**
```python
# Old system with hardcoded tournament strategies
tournament = ParallelGrowthTournament(network, config)
results = tournament.run_tournament(data_loader)
winner = results['winner']  # Fixed set of strategies
```

**After (Composable):**
```python
# New composable system
system = ComposableEvolutionSystem()
system.add_component(StandardExtremaAnalyzer())
system.add_component(ExtremaGrowthStrategy())
system.add_component(InformationFlowGrowthStrategy())

context = NetworkContext(network, data_loader, device)
evolved_context = system.evolve_network(context, num_iterations=5)
```

### Key Benefits

1. **Modularity**: Each component has a single responsibility
2. **Composability**: Mix and match components without custom code
3. **Configurability**: Configure each component individually
4. **Testability**: Test components in isolation
5. **Extensibility**: Add new strategies by implementing interfaces
6. **Monitoring**: Built-in metrics for each component

## Core Architecture

### Interfaces

The system is built around clean interfaces that define contracts:

```python
# Core interfaces
NetworkComponent      # Base interface for all components
NetworkAnalyzer      # Analyzes network state
GrowthStrategy       # Determines and applies growth actions
LearningRateStrategy # Manages learning rate adaptation
NetworkTrainer       # Handles training process
NetworkEvolutionSystem # Coordinates complete evolution
```

### Data Structures

```python
NetworkContext      # Contains network, data, device, metadata
AnalysisResult      # Results from network analysis
GrowthAction        # Specific action to take (add layer, patches, etc.)
ActionType          # Enumeration of possible actions
```

## Components

### Analyzers

Analyzers examine networks and provide insights:

#### StandardExtremaAnalyzer
```python
analyzer = StandardExtremaAnalyzer(
    dead_threshold=0.01,        # Threshold for dead neurons
    saturated_multiplier=2.5,   # Multiplier for saturation detection
    max_batches=5              # Batches to analyze
)

# Configure after creation
analyzer.configure({
    'dead_threshold': 0.005,    # More sensitive
    'max_batches': 10          # More data
})
```

#### NetworkStatsAnalyzer
```python
analyzer = NetworkStatsAnalyzer()
# Provides basic network statistics and architecture info
```

#### SimpleInformationFlowAnalyzer
```python
analyzer = SimpleInformationFlowAnalyzer(
    min_bottleneck_severity=0.05  # Minimum severity to report bottlenecks
)
```

### Growth Strategies

Strategies determine how to grow networks:

#### ExtremaGrowthStrategy
```python
strategy = ExtremaGrowthStrategy(
    extrema_threshold=0.3,           # Threshold for adding layers
    dead_neuron_threshold=5,         # Min dead neurons for patches
    saturated_neuron_threshold=5,    # Min saturated neurons for patches
    patch_size=3                     # Number of connections to add
)
```

#### InformationFlowGrowthStrategy
```python
strategy = InformationFlowGrowthStrategy(
    bottleneck_threshold=0.1,    # Min severity for bottleneck action
    efficiency_threshold=0.7     # Min efficiency before adding skip connections
)
```

#### ResidualBlockGrowthStrategy
```python
strategy = ResidualBlockGrowthStrategy(
    num_layers=2,                # Layers in residual block
    activation_threshold=0.2     # Activation threshold for block addition
)
```

#### HybridGrowthStrategy
```python
# Combine multiple strategies
hybrid = HybridGrowthStrategy([
    ExtremaGrowthStrategy(extrema_threshold=0.2),
    InformationFlowGrowthStrategy(),
    ResidualBlockGrowthStrategy(num_layers=3)
])
```

### Evolution Systems

#### ComposableEvolutionSystem

The main evolution coordinator:

```python
system = ComposableEvolutionSystem()

# Add components
system.add_component(StandardExtremaAnalyzer())
system.add_component(ExtremaGrowthStrategy())
system.add_component(StandardNetworkTrainer())

# Configure entire system
system.configure({
    'analyzers': {
        'analyzer_0': {'max_batches': 8}
    },
    'strategies': {
        'strategy_0': {'extrema_threshold': 0.25}
    }
})

# Run evolution
context = NetworkContext(network, data_loader, device)
evolved_context = system.evolve_network(context, num_iterations=5)

# Get results
summary = system.get_evolution_summary()
metrics = system.get_metrics()
```

## Usage Patterns

### 1. Quick Start with Preconfigured Systems

```python
from src.structure_net.evolution.components import (
    create_standard_evolution_system,
    create_extrema_focused_system,
    create_hybrid_system,
    NetworkContext
)

# Use preconfigured system
system = create_standard_evolution_system()

# Create context
context = NetworkContext(
    network=your_network,
    data_loader=your_data_loader,
    device=torch.device('cuda')
)

# Evolve
evolved_context = system.evolve_network(context, num_iterations=3)
```

### 2. Custom System Composition

```python
from src.structure_net.evolution.components import *

# Build custom system
system = ComposableEvolutionSystem()

# Add specific analyzers
system.add_component(StandardExtremaAnalyzer(max_batches=10))
system.add_component(SimpleInformationFlowAnalyzer())

# Add specific strategies
system.add_component(ExtremaGrowthStrategy(extrema_threshold=0.2))

# Add trainer
system.add_component(StandardNetworkTrainer(learning_rate=0.001))

# Run evolution
evolved_context = system.evolve_network(context, num_iterations=5)
```

### 3. Component Configuration

```python
# Configure individual components
extrema_analyzer = StandardExtremaAnalyzer()
extrema_analyzer.configure({
    'dead_threshold': 0.005,
    'saturated_multiplier': 3.0,
    'max_batches': 8
})

extrema_strategy = ExtremaGrowthStrategy()
extrema_strategy.configure({
    'extrema_threshold': 0.15,
    'dead_neuron_threshold': 3,
    'patch_size': 5
})

# Add to system
system.add_component(extrema_analyzer)
system.add_component(extrema_strategy)
```

### 4. Monitoring and Metrics

```python
# Get component metrics
all_metrics = system.get_metrics()
for key, value in all_metrics.items():
    print(f"{key}: {value}")

# Get evolution summary
summary = system.get_evolution_summary()
print(f"Growth events: {summary['metrics']['total_growth_events']}")
print(f"Average iteration time: {summary['metrics']['average_iteration_time']:.1f}s")

# Get individual component metrics
for analyzer in system.analyzers:
    if hasattr(analyzer, 'get_metrics'):
        print(f"{analyzer.get_name()}: {analyzer.get_metrics()}")
```

### 5. Serialization and Configuration

```python
# Serialize system configuration
config = system.get_configuration()
with open('evolution_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Serialize entire system
system_data = system.to_dict()
with open('evolution_system.json', 'w') as f:
    json.dump(system_data, f, indent=2)

# Configure from file
with open('evolution_config.json', 'r') as f:
    config = json.load(f)
system.configure(config)
```

## Creating Custom Components

### Custom Analyzer

```python
from src.structure_net.evolution.interfaces import NetworkAnalyzer, AnalysisResult

class CustomAnalyzer(NetworkAnalyzer):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def can_apply(self, context: NetworkContext) -> bool:
        return self.validate_context(context)
    
    def apply(self, context: NetworkContext) -> bool:
        result = self.analyze(context)
        context.metadata['custom_analysis'] = result
        return True
    
    def analyze(self, context: NetworkContext) -> AnalysisResult:
        # Your custom analysis logic
        metrics = {'custom_metric': 0.5}
        recommendations = ['custom_recommendation']
        
        return AnalysisResult(
            analyzer_name=self.get_name(),
            metrics=metrics,
            recommendations=recommendations,
            confidence=0.8
        )
```

### Custom Growth Strategy

```python
from src.structure_net.evolution.interfaces import GrowthStrategy, GrowthAction, ActionType

class CustomGrowthStrategy(GrowthStrategy):
    def can_apply(self, context: NetworkContext) -> bool:
        return 'custom_analysis' in context.metadata
    
    def analyze_growth_potential(self, context: NetworkContext) -> AnalysisResult:
        # Analyze if growth is needed
        metrics = {'needs_growth': True}
        return AnalysisResult(
            analyzer_name=self.get_name(),
            metrics=metrics,
            confidence=0.7
        )
    
    def calculate_growth_action(self, analysis: AnalysisResult, context: NetworkContext) -> Optional[GrowthAction]:
        if analysis.metrics.get('needs_growth'):
            return GrowthAction(
                action_type=ActionType.ADD_LAYER,
                position=1,
                size=64,
                reason="Custom growth logic",
                confidence=0.7
            )
        return None
    
    def execute_growth_action(self, action: GrowthAction, context: NetworkContext) -> bool:
        # Your custom growth implementation
        return True
```

## Integration with Existing Code

### Replacing Tournament System

**Old:**
```python
tournament = ParallelGrowthTournament(network, threshold_config, metrics_config)
results = tournament.run_tournament(train_loader, val_loader)
winner = results['winner']
```

**New:**
```python
system = ComposableEvolutionSystem()
system.add_component(StandardExtremaAnalyzer())
system.add_component(ExtremaGrowthStrategy())
system.add_component(InformationFlowGrowthStrategy())

context = NetworkContext(network, train_loader, device)
evolved_context = system.evolve_network(context, num_iterations=1)
```

### Replacing Integrated Growth System

**Old:**
```python
integrated_system = IntegratedGrowthSystem(network, config)
grown_network = integrated_system.grow_network(train_loader, val_loader, growth_iterations=3)
```

**New:**
```python
system = create_standard_evolution_system()
context = NetworkContext(network, train_loader, device, metadata={'val_loader': val_loader})
evolved_context = system.evolve_network(context, num_iterations=3)
grown_network = evolved_context.network
```

## Best Practices

### 1. Component Selection

- **For extrema-heavy problems**: Use `create_extrema_focused_system()`
- **For general purpose**: Use `create_standard_evolution_system()`
- **For complex scenarios**: Use `create_hybrid_system()` or build custom

### 2. Configuration

- Configure components individually for fine-tuning
- Use lower thresholds for more aggressive growth
- Use higher batch counts for more reliable analysis

### 3. Monitoring

- Always check component metrics after evolution
- Monitor growth events and iteration times
- Use component-specific metrics for debugging

### 4. Performance

- Limit analysis batches for faster iteration
- Use fewer components for faster evolution
- Monitor memory usage with large networks

## Migration Guide

### Step 1: Identify Current Usage

Find existing tournament or integrated system usage:
```python
# Look for these patterns
ParallelGrowthTournament
IntegratedGrowthSystem
tournament.run_tournament()
integrated_system.grow_network()
```

### Step 2: Choose Replacement

- **Tournament → ComposableEvolutionSystem**: Direct replacement
- **Integrated → Preconfigured systems**: Use factory functions
- **Custom logic → Custom components**: Implement interfaces

### Step 3: Update Imports

```python
# Old
from src.structure_net.evolution.integrated_growth_system import IntegratedGrowthSystem

# New
from src.structure_net.evolution.components import create_standard_evolution_system
```

### Step 4: Update Code

```python
# Old
system = IntegratedGrowthSystem(network, config)
result = system.grow_network(train_loader, val_loader, growth_iterations=3)

# New
system = create_standard_evolution_system()
context = NetworkContext(network, train_loader, device)
evolved_context = system.evolve_network(context, num_iterations=3)
```

## Examples

See `examples/composable_evolution_example.py` for comprehensive demonstrations of:

1. Basic composable system usage
2. Preconfigured system comparison
3. Component configuration
4. Custom hybrid strategies
5. Monitoring and metrics

## Troubleshooting

### Common Issues

1. **"Component not applicable"**: Check `can_apply()` conditions
2. **"No growth occurred"**: Lower thresholds or check analysis results
3. **"Analysis failed"**: Verify data loader and network compatibility
4. **"Configuration error"**: Check component configuration parameters

### Debugging

```python
# Check component applicability
for component in system.get_components():
    print(f"{component.get_name()}: {component.can_apply(context)}")

# Check analysis results
for analyzer in system.analyzers:
    if analyzer.can_apply(context):
        result = analyzer.analyze(context)
        print(f"{analyzer.get_name()}: {result.metrics}")

# Check component metrics
metrics = system.get_metrics()
for key, value in metrics.items():
    if 'failed' in key or 'error' in key:
        print(f"Issue: {key} = {value}")
```

## Future Extensions

The composable system is designed for easy extension:

1. **New Analyzers**: Implement `NetworkAnalyzer` interface
2. **New Strategies**: Implement `GrowthStrategy` interface
3. **Learning Rate Strategies**: Implement `LearningRateStrategy` interface
4. **Custom Trainers**: Implement `NetworkTrainer` interface
5. **Component Factories**: Create factory functions for common configurations

The interface-based design ensures that new components integrate seamlessly with existing ones, enabling continuous evolution of the evolution system itself!
