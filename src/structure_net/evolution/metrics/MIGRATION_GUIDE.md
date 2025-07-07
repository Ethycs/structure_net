# Migration Guide: Monolithic to Modular Metrics System

This guide helps you migrate from the old monolithic metrics system to the new modular architecture while taking advantage of enhanced features.

## 🔄 Quick Migration (Zero Code Changes)

Your existing code will continue to work without any changes:

```python
# This still works exactly as before
from structure_net.evolution.complete_metrics_system import CompleteMetricsSystem

metrics_system = CompleteMetricsSystem(network, threshold_config, metrics_config)
results = metrics_system.compute_all_metrics(data_loader)
```

The system will automatically use the new modular backend with improved performance.

## 🚀 Recommended Migration (Enhanced Features)

For new projects or when refactoring, use the new modular imports:

### Before (Old Import)
```python
from structure_net.evolution.complete_metrics_system import (
    CompleteMetricsSystem,
    MetricPerformanceAnalyzer
)
```

### After (New Modular Import)
```python
from structure_net.evolution.metrics import (
    CompleteMetricsSystem,
    ThresholdConfig,
    MetricsConfig
)
from structure_net.evolution.autocorrelation import PerformanceAnalyzer
```

## 📊 Enhanced Usage Patterns

### 1. Basic Metrics Computation (Improved Performance)

```python
# Old way (still works)
metrics_system = CompleteMetricsSystem(network, threshold_config, metrics_config)
results = metrics_system.compute_all_metrics(data_loader)

# New way (same API, better performance)
from structure_net.evolution.metrics import CompleteMetricsSystem, ThresholdConfig, MetricsConfig

threshold_config = ThresholdConfig(
    activation_threshold=0.01,
    weight_threshold=0.001,
    adaptive=True
)

metrics_config = MetricsConfig(
    compute_mi=True,
    compute_activity=True,
    compute_sensli=True,
    compute_graph=True,
    sensli_optimization=True  # NEW: Optimized SensLI computation
)

metrics_system = CompleteMetricsSystem(network, threshold_config, metrics_config)
results = metrics_system.compute_all_metrics(data_loader, num_batches=10)

# NEW: Access computation statistics
stats = results['computation_stats']
print(f"Cache hit rate: {stats['mi_analyzer']['cache_hit_rate']:.1%}")
```

### 2. Individual Analyzer Usage (New Capability)

```python
# NEW: Use individual analyzers for focused analysis
from structure_net.evolution.metrics import (
    MutualInformationAnalyzer,
    ActivityAnalyzer,
    SensitivityAnalyzer,
    GraphAnalyzer
)

# Focused MI analysis
mi_analyzer = MutualInformationAnalyzer(threshold_config)
mi_results = mi_analyzer.compute_metrics(layer1_activations, layer2_activations)

# Activity analysis with temporal tracking
activity_analyzer = ActivityAnalyzer(threshold_config)
activity_results = activity_analyzer.compute_metrics(activations, layer_idx=0)
temporal_metrics = activity_analyzer.compute_temporal_metrics(layer_idx=0)
```

### 3. Autocorrelation Framework (Revolutionary New Feature)

```python
# NEW: Meta-learning for growth strategy discovery
from structure_net.evolution.autocorrelation import PerformanceAnalyzer

# Initialize the autocorrelation framework
performance_analyzer = PerformanceAnalyzer()

# During training loop
for epoch in range(num_epochs):
    # Train your model...
    train_acc, val_acc, train_loss, val_loss = train_epoch(model, data_loader)
    
    # Collect performance data
    performance_metrics = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    # NEW: Collect checkpoint data for correlation analysis
    performance_analyzer.collect_checkpoint_data(
        network, data_loader, epoch, performance_metrics
    )
    
    # Compute comprehensive metrics
    complete_metrics = metrics_system.compute_all_metrics(data_loader)
    
    # NEW: Update autocorrelation framework
    performance_analyzer.update_metrics_from_complete_system(epoch, complete_metrics)

# NEW: Discover which metrics predict performance
correlation_results = performance_analyzer.analyze_metric_correlations()

# NEW: Get AI-driven growth recommendations
current_metrics = complete_metrics['summary']
recommendations = performance_analyzer.get_growth_recommendations(current_metrics)

for rec in recommendations:
    print(f"Action: {rec['action']} (confidence: {rec['confidence']:.2f})")
    print(f"Reason: {rec['reason']}")
```

## 🔧 Advanced Features Migration

### 1. Caching and Performance Monitoring

```python
# NEW: Access detailed performance statistics
results = metrics_system.compute_all_metrics(data_loader)
comp_stats = results['computation_stats']

for analyzer_name, stats in comp_stats.items():
    print(f"{analyzer_name}:")
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Average time: {stats['avg_time_per_call']:.4f}s")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")

# NEW: Clear caches when needed
metrics_system.clear_caches()
```

### 2. Strategy Learning and Effectiveness Tracking

```python
# NEW: Record strategy outcomes for learning
performance_analyzer.record_strategy_outcome(
    strategy_name="add_layer_at_bottleneck",
    metrics_before=metrics_before,
    metrics_after=metrics_after,
    performance_improvement=0.05  # 5% improvement
)

# NEW: Get strategy effectiveness summary
effectiveness = performance_analyzer.get_strategy_effectiveness_summary()
for strategy, stats in effectiveness.items():
    print(f"{strategy}: {stats['success_rate']:.1%} success rate")
```

### 3. Cross-Experiment Learning

```python
# NEW: Save learned patterns for reuse
performance_analyzer.save_learned_patterns('growth_patterns.pkl')

# NEW: Load patterns in new experiments
new_analyzer = PerformanceAnalyzer()
new_analyzer.load_learned_patterns('growth_patterns.pkl')
```

## 🎯 Configuration Migration

### Old Configuration
```python
# Old threshold configuration (still works)
class ThresholdConfig:
    def __init__(self):
        self.activation_threshold = 0.01
        self.weight_threshold = 0.001
        self.persistence_ratio = 0.1
```

### New Enhanced Configuration
```python
# NEW: Enhanced configuration with validation
from structure_net.evolution.metrics import ThresholdConfig, MetricsConfig

threshold_config = ThresholdConfig(
    activation_threshold=0.01,
    weight_threshold=0.001,
    persistence_ratio=0.1,
    adaptive=True  # NEW: Adaptive thresholding
)

metrics_config = MetricsConfig(
    compute_mi=True,
    compute_activity=True,
    compute_sensli=True,
    compute_graph=True,
    
    # NEW: Performance settings
    max_batches=10,
    sample_size=1000,
    enable_caching=True,
    
    # NEW: Advanced settings
    mi_method='auto',  # 'auto', 'exact', 'knn', 'advanced'
    graph_sampling=True,
    sensli_optimization=True  # Use optimized precomputed data
)
```

## 🚨 Breaking Changes (None!)

There are **no breaking changes**. All existing code will continue to work exactly as before. The new modular system is a drop-in replacement with enhanced capabilities.

## 📈 Performance Improvements You'll Get Automatically

1. **Single-Pass Data Collection**: Eliminates redundant forward/backward passes
2. **Intelligent Caching**: Automatic caching with hit rate monitoring
3. **Optimized SensLI**: Reuses precomputed activation/gradient data
4. **Memory Efficiency**: Better memory management for large networks

## 🔍 Troubleshooting Common Migration Issues

### Issue 1: Import Warnings
```python
# You might see deprecation warnings like:
# DeprecationWarning: CompleteMIAnalyzer is deprecated. Use MutualInformationAnalyzer...

# Solution: Update to new imports (optional)
from structure_net.evolution.metrics import MutualInformationAnalyzer
```

### Issue 2: Missing Autocorrelation Framework
```python
# If you see: ImportError: cannot import name 'PerformanceAnalyzer'

# Solution: The autocorrelation framework is optional
try:
    from structure_net.evolution.autocorrelation import PerformanceAnalyzer
    AUTOCORR_AVAILABLE = True
except ImportError:
    AUTOCORR_AVAILABLE = False
    print("Autocorrelation framework not available")
```

### Issue 3: Configuration Validation Errors
```python
# New configuration classes have validation

# This will raise an error:
# ThresholdConfig(activation_threshold=-0.01)  # Negative threshold

# Solution: Use valid values
ThresholdConfig(activation_threshold=0.01)  # Positive threshold
```

## 📚 Learning Resources

### Examples
- `examples/modular_metrics_example.py` - Comprehensive demonstration
- `examples/autocorrelation_growth_example.py` - Meta-learning example

### Documentation
- `src/structure_net/evolution/metrics/TODO.md` - Future enhancements
- `docs/autocorrelation_framework.md` - Detailed framework documentation

### API Reference
```python
# Get help on any component
help(CompleteMetricsSystem)
help(PerformanceAnalyzer)
help(MutualInformationAnalyzer)
```

## 🎉 Benefits You'll Gain

### Immediate Benefits (No Code Changes)
- ✅ Better performance through optimized data collection
- ✅ Improved memory efficiency
- ✅ Enhanced error handling and validation
- ✅ Detailed computation statistics

### Enhanced Benefits (With New APIs)
- 🧠 Meta-learning discovers which metrics predict success
- 🎯 AI-driven growth recommendations
- 📊 Individual analyzer usage for focused analysis
- 🔄 Cross-experiment learning and pattern transfer
- ⚡ Advanced caching and performance monitoring

### Future Benefits (Planned)
- 🤖 Fully autonomous network growth
- 📈 Real-time adaptation during training
- 🔬 Advanced pattern discovery methods
- 📊 Interactive visualization dashboard

## 🚀 Next Steps

1. **Keep using your existing code** - it will work better automatically
2. **Try the new autocorrelation framework** for meta-learning capabilities
3. **Explore individual analyzers** for focused analysis
4. **Check out the examples** to see new possibilities
5. **Contribute to the TODO list** to help shape the future

The modular metrics system represents a fundamental shift toward evidence-based network growth that learns and improves with every experiment!
