# Refactoring Plan: Metrics to Components Architecture

## Overview
This plan outlines the migration of the metrics system from `src/structure_net/evolution/metrics/` to the new component architecture in `src/structure_net/components/`.

## Key Concept: Metrics vs Analyzers

```
┌─────────────────────────────────────────────────────────────┐
│                     Current (Monolithic)                     │
├─────────────────────────────────────────────────────────────┤
│  MutualInformationAnalyzer                                  │
│  └── Computes MI, entropy, normalized MI, etc. all together │
│                                                              │
│  ActivityAnalyzer                                            │
│  └── Computes dead neurons, patterns, statistics together   │
└─────────────────────────────────────────────────────────────┘

                            ↓ Refactor ↓

┌─────────────────────────────────────────────────────────────┐
│                    New (Component-Based)                     │
├─────────────────────────────────────────────────────────────┤
│  METRICS (Low-level, focused measurements)                  │
│  ├── LayerMIMetric: Just mutual information                 │
│  ├── EntropyMetric: Just entropy calculations               │
│  ├── DeadNeuronMetric: Just dead neuron detection           │
│  ├── SparsityMetric: Just sparsity ratio                    │
│  └── GradientMetric: Just gradient statistics               │
│                                                              │
│  ANALYZERS (High-level, combine multiple metrics)           │
│  ├── InformationFlowAnalyzer                                │
│  │   └── Uses: LayerMI + Entropy + Gradient metrics         │
│  ├── NetworkHealthAnalyzer                                  │
│  │   └── Uses: DeadNeuron + Sparsity + Activity metrics     │
│  └── StructuralAnalyzer                                     │
│      └── Uses: Connectivity + Clustering + Path metrics     │
│                                                              │
│  TRAINERS (Use metrics/analysis during training)            │
│  ├── AdaptiveTrainer                                        │
│  │   └── Adjusts training based on real-time metrics        │
│  └── MetricAwareTrainer                                     │
│      └── Modifies loss/optimization using metrics           │
│                                                              │
│  SCHEDULERS (Decide when to trigger evolution)              │
│  ├── MetricBasedScheduler                                   │
│  │   └── Triggers when metrics cross thresholds             │
│  └── AnalysisBasedScheduler                                 │
│      └── Triggers based on analyzer recommendations         │
└─────────────────────────────────────────────────────────────┘
```

## Component Interaction Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    Evolution Cycle Flow                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. TRAINER executes training step                           │
│      ↓                                                        │
│  2. METRICS collect measurements during/after step           │
│      ↓                                                        │
│  3. ANALYZERS combine metrics to generate insights           │
│      ↓                                                        │
│  4. SCHEDULER checks if evolution should trigger             │
│      ↓ (if yes)                                               │
│  5. STRATEGY proposes changes based on analysis              │
│      ↓                                                        │
│  6. EVOLVER applies changes to model                         │
│      ↓                                                        │
│  7. TRAINER adapts to new model structure                    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Current Structure Analysis

### Existing Metrics Files:
1. **base.py** - Base classes and interfaces
2. **mutual_information.py** - Mutual information analyzer
3. **activity_analysis.py** - Activity pattern analyzer  
4. **sensitivity_analysis.py** - Sensitivity (SensLI) analyzer
5. **graph_analysis.py** - Graph-theoretic analyzer
6. **topological_analysis.py** - Topological analyzer
7. **homological_analysis.py** - Homological analyzer
8. **compactification_metrics.py** - Compactification analyzer
9. **catastrophe_analysis.py** - Catastrophe theory analyzer
10. **integrated_system.py** - Complete metrics system orchestrator

## Target Component Architecture

### Component Categories:
- **Metrics** (IMetric): Low-level, focused measurements on layers/models
  - Examples: sparsity ratio, weight magnitude, activation variance
- **Analyzers** (IAnalyzer): High-level analysis that COMBINES multiple metrics
  - Examples: network health analysis combining sparsity, activity, and connectivity metrics
- **Orchestrators** (IOrchestrator): Coordinate metrics, analyzers, and other components

## Migration Mapping

### 1. Base Classes → Core Infrastructure
```
base.py → Already compatible with new architecture
- ThresholdConfig → Keep as configuration dataclass
- MetricsConfig → Keep as configuration dataclass  
- MetricResult → Keep as result dataclass
- BaseMetricAnalyzer → Split into BaseMetric and BaseAnalyzer
```

### 2. Low-Level Measurements → Metric Components

These will be broken down into focused, single-purpose metrics:

#### From MutualInformationAnalyzer → Multiple Metrics:
- `components/metrics/layer_mi_metric.py` - Layer-to-layer MI
- `components/metrics/neuron_mi_metric.py` - Neuron-level MI
- `components/metrics/entropy_metric.py` - Entropy measurements

#### From ActivityAnalyzer → Multiple Metrics:
- `components/metrics/activation_metric.py` - Activation statistics
- `components/metrics/dead_neuron_metric.py` - Dead neuron detection
- `components/metrics/activity_pattern_metric.py` - Activity patterns

#### From SensitivityAnalyzer → Multiple Metrics:
- `components/metrics/gradient_metric.py` - Gradient statistics
- `components/metrics/sensitivity_metric.py` - Local sensitivity
- `components/metrics/lipschitz_metric.py` - Lipschitz constants

#### From GraphAnalyzer → Multiple Metrics:
- `components/metrics/connectivity_metric.py` - Connection statistics
- `components/metrics/clustering_metric.py` - Clustering coefficients
- `components/metrics/path_length_metric.py` - Path lengths

### 3. High-Level Analysis → Analyzer Components

These combine multiple metrics to provide insights:

#### Network Health Analyzer → `components/analyzers/network_health_analyzer.py`
```python
# Combines: sparsity, activity, connectivity metrics
# Provides: Overall network health assessment
```

#### Information Flow Analyzer → `components/analyzers/information_flow_analyzer.py`
```python
# FROM: Parts of MutualInformationAnalyzer
# Combines: layer MI, entropy, gradient flow metrics
# Provides: Information bottleneck detection, flow analysis
```

#### Structural Analyzer → `components/analyzers/structural_analyzer.py`
```python
# FROM: Parts of GraphAnalyzer
# Combines: connectivity, clustering, path metrics
# Provides: Network topology insights, critical paths
```

#### Sensitivity Analyzer → `components/analyzers/sensitivity_analyzer.py`
```python
# FROM: SensitivityAnalyzer (enhanced)
# Combines: gradient, sensitivity, Lipschitz metrics
# Provides: Robustness analysis, vulnerable regions
```

#### Topological Analyzer → `components/analyzers/topological_analyzer.py`
```python
# FROM: TopologicalAnalyzer
# Combines: persistence, extrema, connectivity metrics
# Provides: Topological signatures, critical points
```

#### Homological Analyzer → `components/analyzers/homological_analyzer.py`
```python
# FROM: HomologicalAnalyzer
# Combines: topological metrics, chain complexes
# Provides: Homological features, Betti numbers
```

#### Compactification Analyzer → `components/analyzers/compactification_analyzer.py`
```python
# FROM: CompactificationAnalyzer  
# Combines: compression, memory, performance metrics
# Provides: Compactification effectiveness, memory profiles
```

#### Catastrophe Analyzer → `components/analyzers/catastrophe_analyzer.py`
```python
# FROM: CatastropheAnalyzer
# Combines: stability, bifurcation, phase metrics
# Provides: Catastrophe detection, phase transitions
```

### 4. System Integration → Orchestrator, Trainers, and Schedulers

#### Complete Metrics System → `components/orchestrators/metrics_orchestrator.py`
```python
# FROM: CompleteMetricsSystem
# TO: MetricsOrchestrator(BaseOrchestrator)
# Coordinates metrics, analyzers, trainers, and schedulers
```

### 5. Integration with Trainers and Schedulers

#### Metric-Aware Trainer → `components/trainers/adaptive_trainer.py`
```python
# Uses metrics during training to adapt behavior
# Example: Adjusts batch size based on memory metrics
# Example: Modifies loss function based on dead neuron metrics
```

#### Analysis-Driven Scheduler → `components/schedulers/metric_based_scheduler.py`
```python
# Triggers evolution based on metric thresholds
# Example: Trigger growth when information bottlenecks detected
# Example: Trigger pruning when dead neuron ratio exceeds threshold
```

#### Performance-Based Scheduler → `components/schedulers/performance_scheduler.py`
```python
# Uses analyzer outputs to determine evolution timing
# Example: Schedule intervention when network health degrades
# Example: Delay evolution during critical learning phases
```

## Implementation Steps

### Phase 1: Core Infrastructure (Week 1)
1. Create base metric implementations that extend new architecture
2. Update configuration classes to be component-aware
3. Create metric result adapters for compatibility

### Phase 2: Metric Components (Week 1-2)
1. Convert each basic analyzer to IMetric components:
   - Mutual Information
   - Activity Analysis
   - Sensitivity Analysis
   - Graph Analysis
2. Implement contracts with proper data dependencies
3. Add performance tracking and caching

### Phase 3: Analyzer Components (Week 2)
1. Convert high-level analyzers to IAnalyzer components:
   - Topological Analyzer
   - Homological Analyzer
   - Compactification Analyzer
   - Catastrophe Analyzer
2. Define required metrics for each analyzer
3. Implement analysis report integration

### Phase 4: Orchestration (Week 3)
1. Create MetricsOrchestrator to replace CompleteMetricsSystem
2. Implement backward compatibility layer
3. Add composition validation
4. Create migration helpers

### Phase 5: Testing & Validation (Week 3)
1. Create comprehensive test suite
2. Validate backward compatibility
3. Performance benchmarking
4. Documentation updates

## Code Example: Separating Metrics and Analyzers

### Before (Current Monolithic Approach):
```python
class MutualInformationAnalyzer(BaseMetricAnalyzer):
    def __init__(self, threshold_config: ThresholdConfig):
        super().__init__(threshold_config)
        
    def compute_metrics(self, layer1_activations, layer2_activations):
        # Computes MI, entropy, normalized MI, etc. all together
        return {
            'mutual_information': mi_value,
            'normalized_mi': normalized_mi,
            'entropy_x': h_x,
            'entropy_y': h_y,
            'conditional_entropy': h_y_given_x,
            'information_gain': gain
        }
```

### After (Component Architecture):

#### Step 1: Create Focused Metrics
```python
# components/metrics/layer_mi_metric.py
from src.structure_net.core import BaseMetric, ComponentContract

class LayerMIMetric(BaseMetric):
    """Computes mutual information between two layers"""
    
    def __init__(self, name: str = None):
        super().__init__(name or "LayerMIMetric")
        self._measurement_schema = {
            "mutual_information": float,
            "normalized_mi": float
        }
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(2, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs={"layer.activations"},
            provided_outputs={"metrics.layer_mi", "metrics.normalized_mi"},
            resources=ResourceRequirements(memory_level=ResourceLevel.LOW)
        )
    
    def _compute_metric(self, target: ILayer, context: EvolutionContext) -> Dict[str, Any]:
        activations = context.get('layer.activations')
        # Focused computation of just MI
        return {"mutual_information": mi, "normalized_mi": mi / max_entropy}
```

```python
# components/metrics/entropy_metric.py
class EntropyMetric(BaseMetric):
    """Computes entropy of layer activations"""
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            provided_outputs={"metrics.entropy", "metrics.conditional_entropy"}
        )
```

#### Step 2: Create High-Level Analyzer
```python
# components/analyzers/information_flow_analyzer.py
from src.structure_net.core import BaseAnalyzer, AnalysisReport

class InformationFlowAnalyzer(BaseAnalyzer):
    """Analyzes information flow through network using multiple metrics"""
    
    def __init__(self, name: str = None):
        super().__init__(name or "InformationFlowAnalyzer")
        self._required_metrics = {
            "metrics.layer_mi",
            "metrics.entropy", 
            "metrics.gradient_flow"
        }
    
    @property
    def contract(self) -> ComponentContract:
        return ComponentContract(
            component_name=self.name,
            version=ComponentVersion(2, 0, 0),
            maturity=Maturity.STABLE,
            required_inputs=self._required_metrics.union({"model"}),
            provided_outputs={
                "analysis.information_bottlenecks",
                "analysis.information_flow_map",
                "analysis.layer_importance"
            },
            resources=ResourceRequirements(memory_level=ResourceLevel.MEDIUM)
        )
    
    def _perform_analysis(self, model: IModel, report: AnalysisReport, 
                         context: EvolutionContext) -> Dict[str, Any]:
        # Get metrics from the report
        layer_mi_data = report.get("metrics.LayerMIMetric", {})
        entropy_data = report.get("metrics.EntropyMetric", {})
        gradient_data = report.get("metrics.GradientMetric", {})
        
        # Combine metrics to identify information bottlenecks
        bottlenecks = self._identify_bottlenecks(layer_mi_data, entropy_data)
        
        # Create information flow map
        flow_map = self._create_flow_map(layer_mi_data, gradient_data)
        
        # Determine layer importance based on information flow
        layer_importance = self._compute_layer_importance(flow_map, bottlenecks)
        
        return {
            "information_bottlenecks": bottlenecks,
            "information_flow_map": flow_map,
            "layer_importance": layer_importance,
            "recommendations": self._generate_recommendations(bottlenecks)
        }
```

#### Step 3: Orchestrator Coordinates Everything
```python
# components/orchestrators/metrics_orchestrator.py
class MetricsOrchestrator(BaseOrchestrator):
    def __init__(self, model: IModel, metrics: List[IMetric], 
                 analyzers: List[IAnalyzer], trainer: ITrainer,
                 scheduler: IScheduler):
        super().__init__("MetricsOrchestrator")
        self.model = model
        self.metrics = metrics
        self.analyzers = analyzers
        self.trainer = trainer
        self.scheduler = scheduler
        
    def run_cycle(self, context: EvolutionContext) -> Dict[str, Any]:
        # Step 1: Run all metrics
        metric_results = {}
        for metric in self.metrics:
            results = metric.analyze(self.model, context)
            metric_results[metric.name] = results
        
        # Step 2: Create analysis report with metric results
        report = AnalysisReport()
        for name, data in metric_results.items():
            report.add_metric_data(name, data)
        
        # Step 3: Run analyzers that combine metrics
        analyzer_results = {}
        for analyzer in self.analyzers:
            results = analyzer.analyze(self.model, report, context)
            analyzer_results[analyzer.name] = results
            report.add_analyzer_data(analyzer.name, results)
        
        # Step 4: Check if evolution should trigger
        if self.scheduler.should_trigger(context, report):
            # Evolution cycle would run here
            pass
        
        return {
            "metrics": metric_results,
            "analysis": analyzer_results,
            "report": report,
            "should_evolve": self.scheduler.should_trigger(context, report)
        }
```

#### Step 4: Metric-Aware Trainer
```python
# components/trainers/adaptive_trainer.py
class AdaptiveTrainer(BaseTrainer):
    """Trainer that adapts based on real-time metrics"""
    
    def __init__(self, base_lr: float = 0.001, name: str = None):
        super().__init__(name or "AdaptiveTrainer")
        self.base_lr = base_lr
        self._required_metrics = {"metrics.dead_neurons", "metrics.gradient_variance"}
    
    def train_step(self, model: IModel, batch: Any, context: EvolutionContext) -> Dict[str, float]:
        # Get current metrics from context
        dead_neuron_ratio = context.get("metrics.dead_neurons", {}).get("ratio", 0)
        grad_variance = context.get("metrics.gradient_variance", {}).get("variance", 1.0)
        
        # Adapt learning rate based on metrics
        if dead_neuron_ratio > 0.2:  # Many dead neurons
            lr_scale = 1.5  # Increase LR to revive neurons
        elif grad_variance < 0.01:  # Low gradient variance
            lr_scale = 2.0  # Increase LR to escape plateau
        else:
            lr_scale = 1.0
        
        # Adjust optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * lr_scale
        
        # Regular training step
        loss = self._compute_loss(model, batch)
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item(), "lr_scale": lr_scale}
```

#### Step 5: Metric-Based Scheduler
```python
# components/schedulers/metric_based_scheduler.py
class MetricBasedScheduler(BaseScheduler):
    """Triggers evolution based on metric thresholds"""
    
    def __init__(self, thresholds: Dict[str, float], name: str = None):
        super().__init__(name or "MetricBasedScheduler")
        self.thresholds = thresholds
        self._cooldown_steps = 100
        self._last_trigger = -self._cooldown_steps
    
    def should_trigger(self, context: EvolutionContext, report: AnalysisReport) -> bool:
        # Check cooldown
        if context.step - self._last_trigger < self._cooldown_steps:
            return False
        
        # Check metric thresholds
        triggers = []
        
        # Dead neuron threshold
        dead_neurons = report.get("metrics.DeadNeuronMetric", {}).get("ratio", 0)
        if dead_neurons > self.thresholds.get("max_dead_neurons", 0.3):
            triggers.append("high_dead_neurons")
        
        # Information bottleneck threshold
        bottlenecks = report.get("analysis.InformationFlowAnalyzer", {}).get("bottlenecks", [])
        if len(bottlenecks) > self.thresholds.get("max_bottlenecks", 2):
            triggers.append("information_bottlenecks")
        
        # Network health threshold
        health_score = report.get("analysis.NetworkHealthAnalyzer", {}).get("health_score", 1.0)
        if health_score < self.thresholds.get("min_health_score", 0.7):
            triggers.append("poor_network_health")
        
        if triggers:
            self._last_trigger = context.step
            self.log(logging.INFO, f"Evolution triggered by: {triggers}")
            return True
        
        return False
```

## Backward Compatibility Strategy

### 1. Compatibility Layer
Create adapters that maintain the old API while using new components:

```python
# In evolution/metrics/__init__.py
class MutualInformationAnalyzer:
    """Backward compatibility wrapper"""
    def __init__(self, threshold_config):
        self._metric = MutualInformationMetric(threshold_config)
        
    def compute_metrics(self, layer1_activations, layer2_activations):
        context = EvolutionContext()
        context['activations.layer1'] = layer1_activations
        context['activations.layer2'] = layer2_activations
        return self._metric.analyze(None, context)
```

### 2. Import Redirection
Update `__init__.py` files to redirect imports:

```python
# Maintain old imports
from .mutual_information import MutualInformationAnalyzer

# But internally use new components
from src.structure_net.components.metrics.mutual_information_metric import (
    MutualInformationMetric as _MutualInformationMetric
)
```

## Benefits of Migration

1. **Self-Aware Components**: Each metric declares its capabilities
2. **Composition Validation**: Automatic compatibility checking
3. **Resource Management**: Better resource allocation
4. **Performance Tracking**: Built-in performance monitoring
5. **Flexible Composition**: Mix and match metrics as needed
6. **Evolution Integration**: Metrics can participate in evolution cycles

## Testing Strategy

1. **Unit Tests**: Test each component individually
2. **Integration Tests**: Test metric combinations
3. **Compatibility Tests**: Ensure old API still works
4. **Performance Tests**: Benchmark against old implementation
5. **Contract Tests**: Validate component contracts

## Documentation Updates

1. Update component documentation
2. Create migration guide for users
3. Update examples to show both old and new usage
4. Add contract documentation
5. Create composition examples

## Rollout Plan

1. **Alpha**: Internal testing with parallel implementations
2. **Beta**: Selected users test new components
3. **Release**: Full release with compatibility layer
4. **Deprecation**: Mark old system as deprecated
5. **Removal**: Remove old system in future version (6+ months)

## Success Criteria

- [ ] All metrics migrated to component architecture
- [ ] 100% backward compatibility maintained
- [ ] Performance equal or better than original
- [ ] Comprehensive test coverage (>90%)
- [ ] Documentation complete
- [ ] No breaking changes for users
- [ ] Component contracts validated
- [ ] Successful composition examples working

## Timeline

- **Week 1**: Core infrastructure and basic metrics
- **Week 2**: Analyzers and advanced metrics
- **Week 3**: Orchestration and testing
- **Week 4**: Documentation and release preparation

## Next Steps

1. Review and approve this plan
2. Create component directory structure
3. Begin Phase 1 implementation
4. Set up testing infrastructure
5. Create tracking issues for each component migration