# Profiling Data Handling in Structure Net

## Overview

Structure Net includes a comprehensive profiling system that can track performance, memory usage, and execution patterns. This guide explains how profiling data is collected, stored, and integrated with the data system.

## Current Profiling Architecture

### 1. Profiling Components

```
src/structure_net/profiling/
├── core/
│   ├── base_profiler.py      # Base classes and data structures
│   ├── profiler_manager.py   # Global profiling management
│   ├── decorators.py         # Profiling decorators
│   └── context_manager.py    # Context-based profiling
├── components/
│   └── evolution_profiler.py # Evolution-specific profiling
└── factory.py               # Profiler factory functions
```

### 2. Profiling Data Structure

```python
@dataclass
class ProfiledOperation:
    """Core profiling data structure."""
    name: str
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]
    
    # Memory metrics
    memory_before: Optional[float]
    memory_after: Optional[float]
    memory_peak: Optional[float]
    
    # GPU metrics
    gpu_memory_before: Optional[float]
    gpu_memory_after: Optional[float]
    gpu_utilization: Optional[float]
    
    # Custom metrics
    custom_metrics: Dict[str, Any]
    
    # Context
    component: Optional[str]
    tags: List[str]
```

## Integration with Data System

### 1. Adding Profiling to Experiments

```python
from structure_net.profiling import create_standard_profiler, profile_operation
from data_factory import ExperimentSearcher

# Initialize profiler
profiler = create_standard_profiler(
    output_dir="/data/profiling",
    profile_memory=True,
    profile_compute=True
)

# Profile an experiment
with profile_operation("experiment_training") as prof:
    result = run_experiment(...)
    
    # Add profiling data to experiment metadata
    prof.custom_metrics['accuracy'] = result.accuracy
    prof.custom_metrics['parameters'] = result.parameters

# Index with profiling data
searcher = ExperimentSearcher()
searcher.index_experiment(
    experiment_id=result.id,
    experiment_data=result.to_dict(),
    additional_metadata={
        'duration_seconds': prof.duration,
        'memory_peak_mb': prof.memory_peak / 1024**2,
        'gpu_memory_peak_mb': prof.gpu_memory_peak / 1024**2
    }
)
```

### 2. Enhanced Experiment Schema with Profiling

Create a new schema that includes profiling data:

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ProfilingMetrics(BaseModel):
    """Profiling metrics for an experiment."""
    duration_seconds: float = Field(..., description="Total execution time")
    memory_peak_mb: float = Field(..., description="Peak memory usage in MB")
    memory_avg_mb: float = Field(..., description="Average memory usage in MB")
    gpu_memory_peak_mb: Optional[float] = Field(None, description="Peak GPU memory in MB")
    gpu_utilization_avg: Optional[float] = Field(None, description="Average GPU utilization %")
    
    # Detailed breakdowns
    phase_durations: Dict[str, float] = Field(default_factory=dict, description="Duration by phase")
    memory_timeline: Optional[Dict[str, Any]] = Field(None, description="Memory usage over time")
    
class ProfiledExperiment(TrainingExperiment):
    """Experiment with integrated profiling data."""
    profiling: Optional[ProfilingMetrics] = Field(None, description="Profiling metrics")
```

### 3. Profiling Data Storage

#### Option 1: Integrated with Experiment Data

```python
# Store profiling as part of experiment
experiment_data = {
    'experiment_id': 'exp_001',
    'architecture': [...],
    'performance': {...},
    'profiling': {
        'duration_seconds': 234.5,
        'memory_peak_mb': 1024.3,
        'gpu_memory_peak_mb': 4096.7,
        'phase_durations': {
            'data_loading': 12.3,
            'training': 210.5,
            'evaluation': 11.7
        }
    }
}
```

#### Option 2: Separate Profiling Database

```python
# Create dedicated profiling storage
class ProfilingStorage:
    def __init__(self, storage_path: str = "/data/profiling"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
    def save_profile(self, experiment_id: str, profile_data: Dict[str, Any]):
        """Save profiling data for an experiment."""
        profile_file = self.storage_path / f"{experiment_id}_profile.json"
        with open(profile_file, 'w') as f:
            json.dump(profile_data, f, indent=2)
    
    def load_profile(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load profiling data for an experiment."""
        profile_file = self.storage_path / f"{experiment_id}_profile.json"
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                return json.load(f)
        return None
```

### 4. ChromaDB Integration for Profiling Search

```python
from data_factory.search import ExperimentSearcher

class ProfilingSearcher(ExperimentSearcher):
    """Extended searcher with profiling-aware search."""
    
    def search_by_performance_profile(
        self,
        max_duration: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        min_efficiency: Optional[float] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search experiments by performance profile."""
        where_clause = {}
        
        if max_duration:
            where_clause['duration_seconds'] = {'$lte': max_duration}
        
        if max_memory_mb:
            where_clause['memory_peak_mb'] = {'$lte': max_memory_mb}
        
        if min_efficiency:
            # Efficiency = accuracy / (duration * memory)
            where_clause['efficiency_score'] = {'$gte': min_efficiency}
        
        # Create query embedding for efficient experiments
        query_exp = {
            'profiling': {
                'duration_seconds': max_duration or 100,
                'memory_peak_mb': max_memory_mb or 1000
            },
            'final_performance': {'accuracy': 0.95}
        }
        
        return self.search_similar_experiments(
            query_experiment=query_exp,
            n_results=n_results,
            filters=where_clause
        )
```

## Profiling in Ultimate Stress Test v2

Based on the current implementation, here's how to enhance profiling:

```python
# In evaluate_competitor_task
def evaluate_competitor_task(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """Enhanced with profiling."""
    from structure_net.profiling import profile_operation
    
    device = config.get('device', 'cpu')
    enable_profiling = config.get('enable_profiling', False)
    
    if enable_profiling:
        profiler = create_lightweight_profiler()
    
    metrics = {}
    
    # Profile data loading
    with profile_operation("data_loading", enabled=enable_profiling) as prof:
        dataset_config = get_dataset_config(config['dataset'])
        train_loader, test_loader = create_dataset(
            config['dataset'],
            batch_size=config['batch_size']
        )
        if enable_profiling:
            metrics['data_loading_time'] = prof.duration
    
    # Profile model creation
    with profile_operation("model_creation", enabled=enable_profiling) as prof:
        if 'seed_path' in config and config['seed_path']:
            model, _ = load_model_seed(config['seed_path'], device=device)
        else:
            model = create_standard_network(
                architecture=config['architecture'],
                sparsity=config.get('sparsity', 0.02),
                device=device
            )
        if enable_profiling:
            metrics['model_creation_time'] = prof.duration
            metrics['model_memory_mb'] = prof.memory_peak / 1024**2
    
    # Profile training
    with profile_operation("training", enabled=enable_profiling) as prof:
        # ... training code ...
        if enable_profiling:
            metrics['training_time'] = prof.duration
            metrics['training_memory_peak_mb'] = prof.memory_peak / 1024**2
            if torch.cuda.is_available():
                metrics['gpu_memory_peak_mb'] = prof.gpu_memory_peak / 1024**2
    
    # Add profiling summary to metrics
    if enable_profiling:
        metrics['total_time'] = sum([
            metrics.get('data_loading_time', 0),
            metrics.get('model_creation_time', 0),
            metrics.get('training_time', 0)
        ])
        metrics['efficiency'] = metrics['accuracy'] / metrics['total_time']
    
    return model, metrics
```

## Best Practices

### 1. Selective Profiling

```python
# Only profile when needed
if config.enable_profiling and config.profiling_level >= ProfilerLevel.DETAILED:
    with profile_operation("expensive_operation") as prof:
        result = expensive_operation()
else:
    result = expensive_operation()
```

### 2. Hierarchical Profiling

```python
with profile_operation("experiment") as exp_prof:
    with profile_operation("training") as train_prof:
        for epoch in range(epochs):
            with profile_operation(f"epoch_{epoch}") as epoch_prof:
                # Training code
                pass
```

### 3. Memory Profiling

```python
from structure_net.profiling import profile_memory_intensive

@profile_memory_intensive(threshold_mb=1000)
def large_model_training():
    """Will profile if memory usage exceeds 1GB."""
    # Training code
    pass
```

### 4. Async Profiling

```python
from structure_net.profiling import profile_async

@profile_async
async def async_experiment():
    """Profile async operations."""
    result = await some_async_operation()
    return result
```

## Data Export and Analysis

### 1. Export Profiling Data

```python
def export_profiling_data(output_file: str = "profiling_export.csv"):
    """Export all profiling data to CSV."""
    import pandas as pd
    from data_factory import get_chroma_client
    
    client = get_chroma_client()
    collection = client.collection
    
    # Get all experiments with profiling data
    results = collection.get(
        where={"duration_seconds": {"$exists": True}},
        include=['metadatas']
    )
    
    # Convert to DataFrame
    data = []
    for i, metadata in enumerate(results['metadatas']):
        if 'duration_seconds' in metadata:
            data.append({
                'experiment_id': results['ids'][i],
                'duration': metadata.get('duration_seconds'),
                'memory_peak_mb': metadata.get('memory_peak_mb'),
                'gpu_memory_peak_mb': metadata.get('gpu_memory_peak_mb'),
                'accuracy': metadata.get('accuracy'),
                'parameters': metadata.get('parameters'),
                'efficiency': metadata.get('efficiency_score')
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} profiled experiments to {output_file}")
```

### 2. Profiling Analysis

```python
def analyze_profiling_trends():
    """Analyze profiling data trends."""
    df = pd.read_csv("profiling_export.csv")
    
    # Performance vs resource usage
    efficiency_correlation = df['accuracy'].corr(df['duration'])
    memory_correlation = df['accuracy'].corr(df['memory_peak_mb'])
    
    # Find most efficient experiments
    df['efficiency_score'] = df['accuracy'] / (df['duration'] * df['memory_peak_mb'])
    top_efficient = df.nlargest(10, 'efficiency_score')
    
    print(f"Efficiency Analysis:")
    print(f"  Accuracy-Duration correlation: {efficiency_correlation:.3f}")
    print(f"  Accuracy-Memory correlation: {memory_correlation:.3f}")
    print(f"\nTop 10 Most Efficient Experiments:")
    print(top_efficient[['experiment_id', 'accuracy', 'duration', 'memory_peak_mb', 'efficiency_score']])
```

## Integration with Existing Tools

### 1. WandB Integration

```python
if config.integrate_with_wandb:
    import wandb
    
    # Log profiling metrics
    wandb.log({
        "profiling/duration": prof.duration,
        "profiling/memory_peak_mb": prof.memory_peak / 1024**2,
        "profiling/gpu_utilization": prof.gpu_utilization
    })
```

### 2. StandardizedLogger Integration

```python
from structure_net.logging.standardized_logging import StandardizedLogger

logger = StandardizedLogger()

# Log with profiling data
logger.log_experiment_result(
    experiment,
    custom_fields={
        'profiling': {
            'duration_seconds': prof.duration,
            'memory_peak_mb': prof.memory_peak / 1024**2
        }
    }
)
```

## Summary

The profiling system in Structure Net provides:

1. **Comprehensive Metrics**: Time, memory, GPU usage, custom metrics
2. **Flexible Integration**: Works with existing logging and data systems
3. **Searchable Data**: Profiling metrics can be indexed in ChromaDB
4. **Low Overhead**: Minimal impact when disabled
5. **Export Options**: CSV, JSON, or direct database queries

Use profiling to:
- Identify performance bottlenecks
- Track resource usage trends
- Find efficient architectures
- Optimize training pipelines
- Debug memory issues