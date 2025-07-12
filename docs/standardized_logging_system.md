# Standardized Logging System for Structure Net

## Overview

The Structure Net Standardized Logging System implements a robust, schema-validated logging framework that adopts the WandB artifact standard. This system ensures all experimental data follows consistent JSON schemas, provides offline resilience, and integrates seamlessly with WandB for experiment tracking and visualization.

## Key Features

### 🔒 **Pydantic Schema Validation**
- All logged data is validated against strict Pydantic schemas
- Prevents invalid data from entering the system
- Automatic type checking and constraint validation
- Clear error messages for debugging

### 📦 **WandB Artifact Integration**
- Each experiment result becomes a versioned WandB artifact
- Automatic deduplication via content hashing
- Immutable experiment records with full lineage tracking
- Seamless integration with WandB's visualization tools

### 🔄 **Offline Resilience**
- Local queue system for offline operation
- Automatic retry mechanism for failed uploads
- Data never lost even during network outages
- Background daemon for continuous uploading

### 🧮 **Advanced Metrics Integration**
- Built-in support for homological analysis metrics
- Topological data analysis integration
- Compactification metrics tracking
- Extensible schema for custom metrics

### 🌱 **Growth Event Tracking**
- Specialized schemas for network growth events
- Performance impact tracking
- Architecture evolution monitoring
- Comprehensive growth analytics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Standardized Logger                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Pydantic      │  │   Local Queue   │  │    WandB     │ │
│  │   Validation    │  │     System      │  │ Integration  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Homological    │  │  Topological    │  │Compactification│
│  │   Metrics       │  │    Metrics      │  │   Metrics    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Data Schemas

### Core Schemas

#### ExperimentResult
The main schema that encompasses all experiment data:

```python
class ExperimentResult(BaseModel):
    schema_version: str = "1.0"
    experiment_id: str
    timestamp: str
    config: ExperimentConfig
    metrics: MetricsData
    homological_metrics: Optional[HomologicalMetrics] = None
    topological_metrics: Optional[TopologicalMetrics] = None
    compactification_metrics: Optional[CompactificationMetrics] = None
    growth_events: Optional[List[GrowthEvent]] = None
    custom_metrics: Optional[Dict[str, Any]] = None
```

#### ExperimentConfig
Configuration parameters for experiments:

```python
class ExperimentConfig(BaseModel):
    experiment_id: str
    experiment_type: str
    dataset: str
    model_type: str
    batch_size: int = Field(..., gt=0)
    learning_rate: float = Field(..., gt=0.0)
    epochs: int = Field(..., gt=0)
    seed_architecture: Optional[List[int]] = None
    sparsity: Optional[float] = Field(None, ge=0.0, le=1.0)
    growth_enabled: Optional[bool] = None
    device: str
    random_seed: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
```

#### MetricsData
Core performance metrics:

```python
class MetricsData(BaseModel):
    accuracy: float = Field(..., ge=0.0, le=1.0)
    loss: float = Field(..., ge=0.0)
    epoch: int = Field(..., ge=0)
    iteration: Optional[int] = Field(None, ge=0)
    learning_rate: Optional[float] = Field(None, gt=0.0)
    total_parameters: Optional[int] = Field(None, ge=0)
    active_connections: Optional[int] = Field(None, ge=0)
    sparsity: Optional[float] = Field(None, ge=0.0, le=1.0)
    growth_occurred: Optional[bool] = None
    architecture: Optional[List[int]] = None
    extrema_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
```

### Specialized Metrics Schemas

#### HomologicalMetrics
For homological analysis results:

```python
class HomologicalMetrics(BaseModel):
    rank: int = Field(..., ge=0)
    betti_numbers: List[int]
    information_efficiency: float = Field(..., ge=0.0, le=1.0)
    kernel_dimension: int = Field(..., ge=0)
    image_dimension: int = Field(..., ge=0)
    bottleneck_severity: float = Field(..., ge=0.0, le=1.0)
```

#### TopologicalMetrics
For topological data analysis:

```python
class TopologicalMetrics(BaseModel):
    extrema_count: int = Field(..., ge=0)
    extrema_density: float = Field(..., ge=0.0)
    persistence_entropy: float = Field(..., ge=0.0)
    connectivity_density: float = Field(..., ge=0.0, le=1.0)
    topological_complexity: float = Field(..., ge=0.0)
```

#### CompactificationMetrics
For network compactification analysis:

```python
class CompactificationMetrics(BaseModel):
    compression_ratio: float = Field(..., ge=0.0, le=1.0)
    patch_count: int = Field(..., ge=0)
    memory_efficiency: float = Field(..., ge=0.0, le=1.0)
    reconstruction_error: float = Field(..., ge=0.0)
    information_preservation: float = Field(..., ge=0.0, le=1.0)
```

#### GrowthEvent
For tracking network growth events:

```python
class GrowthEvent(BaseModel):
    epoch: int = Field(..., ge=0)
    iteration: Optional[int] = Field(None, ge=0)
    growth_type: str
    growth_location: Optional[str] = None
    connections_added: Optional[int] = Field(None, ge=0)
    accuracy_before: float = Field(..., ge=0.0, le=1.0)
    accuracy_after: float = Field(..., ge=0.0, le=1.0)
    performance_delta: Optional[float] = None
    architecture_before: Optional[List[int]] = None
    architecture_after: Optional[List[int]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
```

## Usage Guide

### Basic Setup

```python
from src.logging.standardized_logging import (
    initialize_logging,
    LoggingConfig,
    log_experiment,
    log_metrics,
    log_growth_event
)

# Initialize the logging system
config = LoggingConfig(
    project_name="my_structure_net_project",
    enable_wandb=True,
    auto_upload=True
)

logger = initialize_logging(config)
```

### Logging Experiments

#### Complete Experiment Result

```python
from src.logging.standardized_logging import (
    ExperimentResult,
    ExperimentConfig,
    MetricsData
)

# Create experiment configuration
config = ExperimentConfig(
    experiment_id="exp_001",
    experiment_type="growth_tracking",
    dataset="MNIST",
    model_type="sparse_mlp",
    batch_size=64,
    learning_rate=0.001,
    epochs=100,
    device="cuda"
)

# Create metrics
metrics = MetricsData(
    accuracy=0.95,
    loss=0.1,
    epoch=50,
    total_parameters=100000,
    sparsity=0.02
)

# Create and log experiment
experiment = ExperimentResult(
    experiment_id="exp_001",
    config=config,
    metrics=metrics
)

result_hash = log_experiment(experiment)
print(f"Logged experiment: {result_hash}")
```

#### Quick Metrics Logging

```python
# Log metrics quickly
metrics_data = {
    "accuracy": 0.92,
    "loss": 0.15,
    "epoch": 25,
    "learning_rate": 0.001
}

result_hash = log_metrics("exp_002", metrics_data)
```

#### Growth Event Logging

```python
from src.logging.standardized_logging import GrowthEvent

# Log a growth event
growth_event = GrowthEvent(
    epoch=30,
    growth_type="add_connections",
    connections_added=100,
    accuracy_before=0.85,
    accuracy_after=0.88,
    architecture_before=[784, 256, 128, 10],
    architecture_after=[784, 256, 128, 10]
)

event_hash = log_growth_event("exp_003", growth_event)
```

### Advanced Metrics Integration

#### Homological Analysis

```python
from src.structure_net.evolution.metrics import create_homological_analyzer
from src.logging.standardized_logging import HomologicalMetrics

# Analyze weight matrix
analyzer = create_homological_analyzer()
results = analyzer.compute_metrics(weight_matrix)

# Create homological metrics
homological_metrics = HomologicalMetrics(
    rank=results['rank'],
    betti_numbers=results['betti_numbers'],
    information_efficiency=results['information_efficiency'],
    kernel_dimension=results['kernel_dimension'],
    image_dimension=results['image_dimension'],
    bottleneck_severity=results['bottleneck_severity']
)

# Include in experiment result
experiment = ExperimentResult(
    experiment_id="exp_004",
    config=config,
    metrics=metrics,
    homological_metrics=homological_metrics
)
```

#### Topological Analysis

```python
from src.structure_net.evolution.metrics import create_topological_analyzer
from src.logging.standardized_logging import TopologicalMetrics

# Analyze topology
analyzer = create_topological_analyzer()
results = analyzer.compute_metrics(weight_matrix)

# Create topological metrics
topological_metrics = TopologicalMetrics(
    extrema_count=results['extrema_count'],
    extrema_density=results['extrema_density'],
    persistence_entropy=results['topological_signature'].persistence_entropy,
    connectivity_density=results['topological_signature'].connectivity_density,
    topological_complexity=results['topological_signature'].topological_complexity
)
```

#### Compactification Analysis

```python
from src.structure_net.evolution.metrics import create_compactification_analyzer
from src.logging.standardized_logging import CompactificationMetrics

# Analyze compactification
analyzer = create_compactification_analyzer()
results = analyzer.compute_metrics(compact_data)

# Create compactification metrics
compactification_metrics = CompactificationMetrics(
    compression_ratio=results['compression_stats'].compression_ratio,
    patch_count=results['patch_effectiveness'].patch_count,
    memory_efficiency=results['memory_profile'].memory_efficiency,
    reconstruction_error=results['reconstruction_metrics']['reconstruction_error'],
    information_preservation=results['patch_effectiveness'].information_preservation
)
```

## Queue System

### Directory Structure

```
experiment_queue/     # Pending uploads
├── a1b2c3d4.json    # Queued experiment
├── e5f6g7h8.json    # Queued experiment
└── ...

experiment_sent/      # Successfully uploaded
├── a1b2c3d4.json    # Sent to WandB
└── ...

experiment_rejected/  # Failed validation
├── invalid_20240101_120000.json
└── ...
```

### Queue Management

```python
# Check queue status
status = logger.get_queue_status()
print(f"Queued: {status['queued']}")
print(f"Sent: {status['sent']}")
print(f"Rejected: {status['rejected']}")

# Upload all queued experiments
stats = logger.upload_all_queued()
print(f"Upload stats: {stats}")

# Start background upload daemon
logger.start_upload_daemon()

# Stop daemon when done
logger.stop_upload_daemon()
```

## WandB Integration

### Artifact Creation

Each experiment result becomes a WandB artifact with:

- **Name**: Content hash (16 characters)
- **Type**: "experiment-result"
- **Version**: Automatic versioning by WandB
- **Metadata**: Experiment ID, type, schema version, upload timestamp

### Deduplication

The system uses SHA-256 content hashing to ensure:
- Identical experiments are never uploaded twice
- Efficient storage and bandwidth usage
- Consistent artifact naming

### Retrieval

```python
import wandb

# Access logged experiments
api = wandb.Api()
artifacts = api.artifacts("my_project/experiment-result")

for artifact in artifacts:
    # Download and analyze
    artifact_dir = artifact.download()
    # Process experiment data...
```

## Error Handling and Validation

### Validation Errors

The system catches and handles various validation errors:

```python
# Invalid accuracy value
try:
    metrics = {"accuracy": 1.5, "loss": 0.2, "epoch": 10}
    log_metrics("test", metrics)
except ValueError as e:
    print(f"Validation error: {e}")
```

### Quarantine System

Invalid data is automatically quarantined:

```json
{
  "error": "accuracy must be <= 1.0",
  "timestamp": "2024-01-01T12:00:00",
  "data": {
    "accuracy": 1.5,
    "loss": 0.2,
    "epoch": 10
  }
}
```

### Schema Migration

Support for evolving schemas:

```python
# Migrate old data format
old_data = {...}  # Legacy format
migrated_data = logger.migrate_schema(old_data, target_version="1.0")

# Validate migrated data
is_valid = logger.validate_schema(migrated_data)
```

## Configuration Options

### LoggingConfig

```python
@dataclass
class LoggingConfig:
    project_name: str = "structure_net"
    queue_dir: str = "experiment_queue"
    sent_dir: str = "experiment_sent"
    rejected_dir: str = "experiment_rejected"
    enable_wandb: bool = True
    enable_local_backup: bool = True
    auto_upload: bool = True
    upload_interval: int = 30  # seconds
    max_retries: int = 3
```

## Best Practices

### 1. Experiment Naming
- Use descriptive, unique experiment IDs
- Include version numbers or timestamps
- Follow consistent naming conventions

### 2. Metrics Organization
- Log core metrics consistently
- Use specialized metrics schemas when appropriate
- Include sufficient context for reproducibility

### 3. Growth Event Tracking
- Log all significant network changes
- Include before/after performance metrics
- Document the reasoning behind growth decisions

### 4. Error Handling
- Always handle validation errors gracefully
- Check queue status regularly
- Monitor rejected experiments for issues

### 5. Schema Evolution
- Plan for schema changes from the beginning
- Implement migration functions for breaking changes
- Test migrations thoroughly before deployment

## Integration with Existing Systems

### WandB Integration

The logging system seamlessly integrates with existing WandB workflows:

```python
# Use with existing WandB runs
import wandb

wandb.init(project="my_project")

# Log using standardized system
result_hash = log_experiment(experiment_result)

# Continue with regular WandB logging
wandb.log({"custom_metric": 0.95})
```

### Metrics System Integration

Works with all Structure Net metrics analyzers:

```python
from src.structure_net.evolution.metrics import CompleteMetricsSystem

# Use complete metrics system
metrics_system = CompleteMetricsSystem()
all_metrics = metrics_system.compute_all_metrics(network, data)

# Convert to standardized format and log
experiment_result = create_experiment_from_metrics(all_metrics)
log_experiment(experiment_result)
```

## Troubleshooting

### Common Issues

1. **Validation Errors**
   - Check field types and constraints
   - Ensure required fields are present
   - Verify value ranges (e.g., accuracy ∈ [0,1])

2. **Upload Failures**
   - Check WandB authentication
   - Verify network connectivity
   - Check disk space for queue directories

3. **Schema Mismatches**
   - Update to latest schema version
   - Use migration functions for old data
   - Check for breaking changes in updates

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all logging operations will show detailed information
```

## Future Enhancements

### Planned Features

1. **Advanced Schema Migration**
   - Automatic schema version detection
   - Complex migration pipelines
   - Backward compatibility guarantees

2. **Enhanced Metrics**
   - Real-time performance monitoring
   - Automated anomaly detection
   - Comparative analysis tools

3. **Distributed Logging**
   - Multi-node experiment coordination
   - Distributed queue management
   - Cluster-wide experiment tracking

4. **Visualization Tools**
   - Custom WandB dashboards
   - Interactive experiment browsers
   - Automated report generation

## Conclusion

The Structure Net Standardized Logging System provides a robust, scalable foundation for experiment tracking and analysis. By combining Pydantic validation, WandB integration, and offline resilience, it ensures that experimental data is always consistent, accessible, and valuable for research and development.

The system's modular design allows for easy extension and customization while maintaining strict data quality standards. Whether you're running single experiments or large-scale research campaigns, this logging system provides the reliability and features needed for serious machine learning research.
