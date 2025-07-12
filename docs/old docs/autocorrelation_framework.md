# Autocorrelation Framework for Meta-Learning Growth

## Overview

The Autocorrelation Framework is a revolutionary meta-learning approach that discovers which metrics actually predict neural network learning success. Instead of using fixed rules for network growth, this framework learns the "laws of neural network growth" from your own experimental data through statistical analysis and correlation discovery.

## Key Innovation

**Traditional Approach**: "Try different growth strategies and see what works"

**Autocorrelation Approach**: "Learn which network conditions predict when each strategy will succeed, then automatically apply the best strategy for the current situation"

## Core Components

### 1. MetricPerformanceAnalyzer

The heart of the framework that collects comprehensive metrics and discovers predictive patterns:

```python
from structure_net.evolution.complete_metrics_system import MetricPerformanceAnalyzer

analyzer = MetricPerformanceAnalyzer()

# Collect metrics at each checkpoint
analyzer.collect_checkpoint_data(network, dataloader, epoch, performance_metrics)

# Discover which metrics predict future performance
correlation_results = analyzer.analyze_metric_correlations()

# Get learned recommendations
recommendations = analyzer.get_growth_recommendations(current_metrics)
```

### 2. Enhanced IntegratedGrowthSystem

The growth system now learns and adapts based on discovered patterns:

```python
from structure_net.evolution.integrated_growth_system import IntegratedGrowthSystem

# Initialize with autocorrelation framework
growth_system = IntegratedGrowthSystem(network, threshold_config, metrics_config)

# Growth loop automatically learns and adapts
final_network = growth_system.grow_network(
    train_loader, val_loader,
    growth_iterations=5,
    epochs_per_iteration=20
)
```

## What the Framework Discovers

### 1. Metric Importance Ranking
- Which of the 50+ available metrics actually correlate with future performance
- Linear relationships (Pearson correlation)
- Monotonic relationships (Spearman correlation) 
- Non-linear relationships (Mutual Information)

### 2. Critical Thresholds
- Specific metric values where performance changes dramatically
- Example: "Growth only helps when MI efficiency < 0.3"
- Automatic threshold detection using quantile analysis

### 3. Interaction Effects
- Metric combinations that predict performance
- Example: "Low MI + High betweenness = guaranteed improvement"
- Pairwise interaction analysis

### 4. Temporal Patterns
- Leading indicators that change 5-10 epochs before performance drops
- Optimal timing for growth interventions
- Autocorrelation analysis at different time lags

### 5. Strategy Effectiveness
- Which strategies work under which metric conditions
- Success rates and average improvements for each strategy
- Automatic strategy weighting based on learned patterns

## Comprehensive Metrics Collected

The framework collects over 50 different metrics across multiple categories:

### Mutual Information Metrics
- `mi_efficiency_mean`: Average information flow efficiency
- `mi_efficiency_min`: Worst bottleneck severity
- `total_information_loss`: Cumulative information loss
- `mi_bottleneck_severity`: Number of severe bottlenecks

### Graph-Based Metrics
- `algebraic_connectivity`: Network connectivity strength
- `spectral_gap`: Information mixing efficiency
- `betweenness_max`: Maximum bottleneck centrality
- `num_components`: Network fragmentation
- `percolation_distance`: Distance to percolation threshold

### Activity Metrics
- `active_neuron_ratio`: Fraction of active neurons
- `dead_neuron_count`: Number of inactive neurons
- `activation_variance`: Activation distribution spread
- `extrema_ratio`: Fraction of extreme activations

### Gradient Flow Metrics
- `gradient_norm_mean`: Average gradient magnitude
- `gradient_correlation`: Gradient flow stability
- `sensitivity_stability`: Gradient sensitivity consistency

## Example Usage

### Basic Usage

```python
import torch
from structure_net.core.network_factory import create_standard_network
from structure_net.evolution.integrated_growth_system import IntegratedGrowthSystem
from structure_net.evolution.advanced_layers import ThresholdConfig, MetricsConfig

# Create network
network = create_standard_network(784, [128, 64], 10, sparsity=0.1)

# Configure framework
threshold_config = ThresholdConfig()
threshold_config.adaptive = True

metrics_config = MetricsConfig()
metrics_config.compute_mi = True
metrics_config.compute_activity = True
metrics_config.compute_sensli = True
metrics_config.compute_graph = True

# Initialize growth system with autocorrelation
growth_system = IntegratedGrowthSystem(network, threshold_config, metrics_config)

# Run growth with automatic learning
final_network = growth_system.grow_network(
    train_loader, val_loader,
    growth_iterations=5,
    epochs_per_iteration=20
)
```

### Advanced Analysis

```python
# Access the performance analyzer
analyzer = growth_system.performance_analyzer

# Get correlation results
if analyzer.correlation_results:
    # Find top predictive metrics
    top_metrics = analyzer._find_top_predictive_metrics(
        analyzer.correlation_results, top_n=10
    )
    
    for metric_info in top_metrics:
        print(f"{metric_info['metric']}: {metric_info['val_correlation']:.3f}")

# Get strategy effectiveness
effectiveness = analyzer.get_strategy_effectiveness_summary()
for strategy, stats in effectiveness.items():
    print(f"{strategy}: {stats['success_rate']:.1%} success rate")

# Find critical thresholds
threshold_info = analyzer.find_critical_thresholds('mi_efficiency_mean')
if threshold_info:
    print(f"Critical threshold: {threshold_info['critical_value']:.3f}")
    print(f"Performance jump: {threshold_info['performance_jump']:.3f}")
```

## Key Insights the Framework Reveals

### 1. Which Metrics Actually Matter
Instead of guessing, you'll know exactly which metrics predict success:
- MI efficiency might be the king predictor
- Or maybe spectral gap is what really matters
- The framework discovers this from your data

### 2. Precise Growth Conditions
Learn exactly when each strategy works:
- "Add layers when MI efficiency < 0.3 AND spectral gap > 0.1"
- "Patches work best when dead neuron count > 50"
- "Hybrid strategies excel when betweenness variance > 0.05"

### 3. Early Warning System
Identify problems before they hurt performance:
- "When algebraic connectivity drops below 0.2, performance will decline in 5 epochs"
- "Rising extrema ratio predicts upcoming saturation"

### 4. Optimal Timing
Know exactly when to intervene:
- "Growth is most effective after 3 epochs of declining MI efficiency"
- "Wait for percolation distance to exceed 0.1 before adding connections"

## Integration with Existing Systems

The framework seamlessly integrates with your existing structure_net components:

- **Compatible with all existing layers**: StandardSparseLayer, ExtremaAwareSparseLayer
- **Works with existing metrics**: All previous metrics are included and enhanced
- **Extends existing growth strategies**: Tournament system now learns optimal strategy selection
- **Maintains existing APIs**: Drop-in replacement for existing growth systems

## Performance Benefits

### Automatic Strategy Selection
- No more manual tuning of growth strategies
- System learns which strategy works best for current network state
- Continuous improvement as more data is collected

### Predictive Growth
- Intervene before problems occur
- Optimal timing based on learned patterns
- Reduced wasted computation on ineffective strategies

### Self-Improving System
- Gets better with more experiments
- Transfers learning across different architectures
- Builds universal growth rules from your data

## Advanced Features

### Cross-Experiment Learning
```python
# Save learned patterns
analyzer.save_learned_patterns('growth_patterns.pkl')

# Load patterns for new experiments
analyzer.load_learned_patterns('growth_patterns.pkl')
```

### Custom Metric Integration
```python
# Add your own metrics to the analysis
def custom_metric(network, activations):
    return some_computation(network, activations)

analyzer.register_custom_metric('my_metric', custom_metric)
```

### Real-time Adaptation
```python
# Enable real-time strategy adaptation
growth_system.enable_realtime_adaptation()

# System will continuously update strategy weights during training
```

## Visualization and Insights

The framework provides rich visualization capabilities:

```python
# Generate insight dashboard
dashboard = analyzer.create_insight_dashboard()

# Plot correlation heatmaps
analyzer.plot_correlation_matrix()

# Show strategy effectiveness over time
analyzer.plot_strategy_effectiveness()

# Display critical threshold analysis
analyzer.plot_critical_thresholds()
```

## Future Directions

### Planned Enhancements
1. **Random Forest Analysis**: Use ensemble methods for non-linear pattern discovery
2. **Temporal Sequence Modeling**: LSTM-based prediction of optimal growth timing
3. **Multi-Objective Optimization**: Balance multiple performance metrics simultaneously
4. **Transfer Learning**: Apply learned patterns across different domains
5. **Causal Analysis**: Discover causal relationships, not just correlations

### Research Applications
- **Architecture Search**: Discover optimal architectures based on learned growth patterns
- **Hyperparameter Optimization**: Learn which hyperparameters matter for growth
- **Domain Adaptation**: Transfer growth knowledge across different datasets
- **Theoretical Insights**: Discover fundamental principles of neural network growth

## Getting Started

1. **Install Dependencies**: Ensure you have pandas, scikit-learn, and scipy
2. **Run the Example**: Execute `examples/autocorrelation_growth_example.py`
3. **Analyze Results**: Examine the learned patterns and strategy effectiveness
4. **Integrate**: Replace your existing growth system with the autocorrelation framework
5. **Experiment**: Try different datasets and architectures to build your knowledge base

## Conclusion

The Autocorrelation Framework transforms neural network growth from an art to a science. By learning the relationships between network metrics and performance outcomes, it creates a self-improving system that gets better with every experiment.

This is not just an incremental improvement—it's a fundamental shift toward evidence-based network growth that could reveal the underlying "laws" governing how neural networks should evolve.
