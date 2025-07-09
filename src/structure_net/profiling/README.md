# Structure Net Profiling System

A comprehensive, modular profiling system for monitoring performance, memory usage, and execution patterns across all components of the structure_net system.

## 🎯 Key Features

- **Toggleable Profiling**: Minimal overhead when disabled, comprehensive monitoring when enabled
- **Granular Control**: Profile individual functions, methods, or entire components
- **Component-Specific Profilers**: Specialized profilers for evolution, metrics, training, and network operations
- **Adaptive Overhead Management**: Automatically adjusts profiling detail to maintain performance
- **Multiple Output Formats**: JSON, CSV, and integration with Weights & Biases
- **Memory & GPU Profiling**: Track memory usage and GPU utilization
- **Real-time Monitoring**: Live performance reports and metrics aggregation

## 🚀 Quick Start

### Basic Usage

```python
from structure_net.profiling import create_standard_profiler, profile_function

# Create a profiler
profiler = create_standard_profiler()
profiler.start_session("my_experiment")

# Profile a function with decorator
@profile_function(component="evolution")
def my_evolution_function():
    # Your code here
    pass

# Profile with context manager
with profiler.profile_operation("data_loading", "data"):
    # Load and process data
    pass

# Get results
profiler.end_session()
print(profiler.get_performance_report())
```

### Evolution-Focused Profiling

```python
from structure_net.profiling import create_evolution_focused_profiler

# Create specialized evolution profiler
profiler = create_evolution_focused_profiler()
evolution_profiler = profiler.get_profiler("evolution_profiler")

# Track growth events
evolution_profiler.track_growth_event(
    strategy_name="ExtremaGrowthStrategy",
    growth_type="add_layer",
    network_before=old_network,
    network_after=new_network,
    performance_improvement=0.05
)

# Get evolution-specific metrics
metrics = evolution_profiler.get_specialized_metrics()
rankings = evolution_profiler.get_strategy_ranking()
```

## 📊 Profiler Types

### 1. Standard Profiler
General-purpose profiler suitable for most use cases.

```python
profiler = create_standard_profiler(
    level=ProfilerLevel.BASIC,
    enable_memory=True,
    enable_compute=False
)
```

### 2. Lightweight Profiler
Minimal overhead profiler for production environments.

```python
profiler = create_lightweight_profiler()
# < 1% overhead, basic time tracking only
```

### 3. Comprehensive Profiler
Full-featured profiler with all monitoring capabilities.

```python
profiler = create_comprehensive_profiler(
    enable_wandb=True  # Integrate with Weights & Biases
)
```

### 4. Evolution-Focused Profiler
Specialized for network evolution experiments.

```python
profiler = create_evolution_focused_profiler(
    level=ProfilerLevel.DETAILED
)
```

### 5. Research Profiler
Optimized for research experiments with comprehensive data collection.

```python
profiler = create_research_profiler("my_experiment")
```

## 🎛️ Configuration Options

### Profiler Levels

- **DISABLED**: No profiling (0% overhead)
- **BASIC**: Time tracking only (< 1% overhead)
- **DETAILED**: Time + memory tracking (< 5% overhead)
- **COMPREHENSIVE**: All metrics (< 10% overhead)

### Configuration Example

```python
from structure_net.profiling import ProfilerConfig, ProfilerLevel

config = ProfilerConfig(
    level=ProfilerLevel.DETAILED,
    profile_memory=True,
    profile_compute=True,
    profile_io=False,
    output_dir="my_profiling_results",
    auto_save=True,
    save_interval=50,
    max_overhead_percent=5.0,
    adaptive_sampling=True
)
```

## 🎨 Decorators

### Function Profiling

```python
@profile_function(component="evolution", tags=["growth"])
def grow_network(network):
    # Function automatically profiled
    pass
```

### Method Profiling

```python
class MyEvolutionSystem:
    @profile_method(component="evolution")
    def analyze_network(self, network):
        # Method automatically profiled
        pass
```

### Component Profiling

```python
@profile_component(component_name="evolution")
class MyEvolutionSystem:
    # All public methods automatically profiled
    def method1(self): pass
    def method2(self): pass
```

### Conditional Profiling

```python
@profile_if_enabled(condition=lambda: os.getenv('PROFILE') == '1')
def conditional_function():
    # Only profiled if condition is met
    pass
```

## 📈 Specialized Profilers

### Evolution Profiler

Tracks evolution-specific metrics:

```python
evolution_profiler = EvolutionProfiler()

# Track growth events
evolution_profiler.track_growth_event(
    strategy_name="AddLayerStrategy",
    growth_type="layer_addition",
    network_before=old_net,
    network_after=new_net,
    performance_improvement=0.03
)

# Track analyzer performance
evolution_profiler.track_analyzer_execution(
    analyzer_name="ExtremaAnalyzer",
    execution_time=0.15,
    analysis_results=results,
    network_size=1000
)

# Get strategy rankings
rankings = evolution_profiler.get_strategy_ranking()
```

### Metrics Available

- **Growth Events**: Strategy performance, improvement tracking
- **Analyzer Performance**: Execution times, efficiency analysis
- **Architecture Changes**: Network structure evolution
- **Strategy Rankings**: Performance-based strategy comparison

## 🔧 Advanced Usage

### Custom Profiler Setup

```python
from structure_net.profiling import create_custom_profiler

profiler_configs = {
    "evolution": ProfilerConfig(level=ProfilerLevel.COMPREHENSIVE),
    "metrics": ProfilerConfig(level=ProfilerLevel.BASIC),
    "training": ProfilerConfig(level=ProfilerLevel.DETAILED)
}

profiler = create_custom_profiler(profiler_configs)
```

### Integration with Weights & Biases

```python
import wandb

# Initialize wandb
run = wandb.init(project="structure_net")

# Integrate with profiler
profiler.integrate_with_wandb(run)

# Profiling data automatically logged to wandb
```

### Manual Profiling

```python
# Start operation
op_id = profiler.start_operation("my_operation", "component")

# Your code here
result = expensive_function()

# End operation with custom metrics
profiler.end_operation(op_id, 
                      custom_metric=42,
                      result_size=len(result))
```

## 📊 Reports and Analysis

### Performance Report

```python
# Get human-readable report
report = profiler.get_performance_report()
print(report)
```

Example output:
```
🔬 PROFILING PERFORMANCE REPORT
==================================================
Session: evolution_experiment_1
Duration: 45.23s
Total Operations: 1,247
Total Time: 42.15s
Memory Delta: 156.3MB

📊 Component Breakdown:
  evolution: 45 ops, 15.234s total, 0.3385s avg
  training: 120 ops, 12.456s total, 0.1038s avg
  metrics: 89 ops, 8.123s total, 0.0913s avg

⏱️  Top Operations by Total Time:
  network_growth: 8.234s (12 calls, 0.6862s avg)
  extrema_analysis: 4.567s (45 calls, 0.1015s avg)
  training_epoch: 3.891s (15 calls, 0.2594s avg)
```

### Aggregated Metrics

```python
# Get detailed metrics
aggregated = profiler.get_aggregated_metrics()

print(f"Total operations: {aggregated['total_operations']}")
print(f"Component breakdown: {aggregated['component_breakdown']}")
print(f"Operation breakdown: {aggregated['operation_breakdown']}")
```

### Evolution-Specific Analysis

```python
evolution_profiler = profiler.get_profiler("evolution_profiler")

# Strategy performance
rankings = evolution_profiler.get_strategy_ranking()
for rank in rankings:
    print(f"{rank['strategy_name']}: {rank['average_improvement']:.3f}")

# Analyzer efficiency
efficiency = evolution_profiler.get_analyzer_efficiency_report()
print(f"Most efficient: {efficiency['most_efficient']}")
print(f"Least efficient: {efficiency['least_efficient']}")
```

## 💾 Data Export

### JSON Export

```python
# Auto-save enabled by default
profiler.save_all_results()

# Manual save
profiler.save_results("custom_filename.json")
```

### Session Data

```python
# End session with data export
session_data = profiler.end_session(save_results=True)

# Session data structure:
{
    "session_id": "experiment_123",
    "session_duration": 45.23,
    "profilers": {
        "evolution_profiler": {
            "summary_stats": {...},
            "specialized_metrics": {...}
        }
    }
}
```

## ⚡ Performance Considerations

### Overhead Management

The profiling system automatically manages overhead:

- **Adaptive Sampling**: Reduces detail if overhead exceeds threshold
- **Level Adjustment**: Automatically downgrades profiling level
- **Selective Profiling**: Only profile when conditions are met

### Best Practices

1. **Use appropriate profiling levels**:
   - Production: `ProfilerLevel.BASIC` or `DISABLED`
   - Development: `ProfilerLevel.DETAILED`
   - Research: `ProfilerLevel.COMPREHENSIVE`

2. **Configure overhead limits**:
   ```python
   config.max_overhead_percent = 2.0  # Production
   config.max_overhead_percent = 10.0  # Research
   ```

3. **Use conditional profiling**:
   ```python
   @profile_if_enabled(condition=lambda: DEBUG_MODE)
   def debug_function():
       pass
   ```

## 🔍 Troubleshooting

### Common Issues

1. **High Overhead**:
   - Reduce profiling level
   - Enable adaptive sampling
   - Use selective profiling

2. **Missing Data**:
   - Check if profiler is enabled
   - Verify profiling level
   - Ensure session is started

3. **Import Errors**:
   - Install required dependencies: `psutil`, `torch`
   - Optional: `pynvml` for GPU profiling

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check profiler status
status = profiler.get_status()
print(f"Profiling enabled: {status['is_profiling']}")
print(f"Active profilers: {status['profilers_count']}")
```

## 📚 Examples

See `examples/profiling_system_example.py` for comprehensive usage examples including:

- Basic function and method profiling
- Evolution-specific profiling
- Training profiling
- Comprehensive profiling with all features
- Profiler comparison and overhead analysis

## 🤝 Contributing

When adding new profilers:

1. Inherit from `BaseProfiler`
2. Implement `get_specialized_metrics()`
3. Add to factory functions
4. Update documentation

Example:
```python
class MyCustomProfiler(BaseProfiler):
    def __init__(self, config):
        super().__init__("my_profiler", config)
    
    def get_specialized_metrics(self):
        return {"custom_metric": "value"}
```

## 📄 License

This profiling system is part of the structure_net project and follows the same license terms.
