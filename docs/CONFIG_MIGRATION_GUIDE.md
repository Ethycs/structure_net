# Configuration Migration Guide

This guide explains how to migrate from the old scattered configuration system to the new unified configuration system.

## Overview

The old system had multiple configuration classes scattered throughout the codebase:
- `LabConfig` + `LabConfigFactory` (Neural Architecture Lab)
- `LoggingConfig` (StandardizedLogger) 
- `DatasetConfig` (Data Factory)
- `ProfilerConfig`, `ChromaConfig`, `ArtifactConfig`, etc.

The new system provides a single `UnifiedConfig` class that contains all configuration in a hierarchical structure.

## Quick Start

### Old Way
```python
from neural_architecture_lab import LabConfig
from structure_net.logging import LoggingConfig

lab_config = LabConfig(
    project_name="my_project",
    results_dir="data/results",
    device_ids=[0, 1],
    enable_wandb=True
)

logging_config = LoggingConfig(
    project_name="my_project",
    enable_wandb=True,
    chromadb_path="data/chroma_db"
)
```

### New Way
```python
from structure_net.config import UnifiedConfig

config = UnifiedConfig()
config.experiment.project_name = "my_project"
config.storage.results_dir = "data/results"
config.compute.device_ids = [0, 1]
config.wandb.enabled = True
config.storage.chromadb_path = "data/chroma_db"
```

## Configuration Structure

The `UnifiedConfig` class has five main sections:

### 1. Storage Configuration
```python
config.storage.data_root         # Base directory for all data
config.storage.results_dir       # Experiment results
config.storage.datasets_dir      # Dataset storage
config.storage.models_dir        # Saved models
config.storage.chromadb_path     # ChromaDB location
```

### 2. WandB Configuration
```python
config.wandb.enabled             # Enable/disable WandB
config.wandb.project            # Project name
config.wandb.entity             # Team/organization
config.wandb.tags               # Run tags
```

### 3. Logging Configuration
```python
config.logging.level            # Log level (INFO, DEBUG, etc.)
config.logging.enable_file      # Log to file
config.logging.enable_wandb     # Log to WandB
config.logging.module_levels    # Per-module log levels
```

### 4. Compute Configuration
```python
config.compute.device_ids       # GPU IDs to use
config.compute.max_parallel_experiments  # Parallelism
config.compute.num_workers      # DataLoader workers
config.compute.pin_memory       # Pin memory for GPU
```

### 5. Experiment Configuration
```python
config.experiment.project_name  # Experiment project
config.experiment.batch_size    # Training batch size
config.experiment.learning_rate # Learning rate
config.experiment.epochs        # Number of epochs
```

## Migration Strategies

### Strategy 1: Automatic Migration (Recommended)

Use the migration helper to automatically convert old configs:

```python
from structure_net.config import ConfigMigrationHelper

# Migrate from old LabConfig
unified = ConfigMigrationHelper.from_lab_config(old_lab_config)

# Migrate from old LoggingConfig  
unified = ConfigMigrationHelper.from_logging_config(old_logging_config)

# Migrate multiple configs at once
unified = ConfigMigrationHelper.from_multiple_configs(
    lab_config=old_lab_config,
    logging_config=old_logging_config
)
```

### Strategy 2: Compatibility Shims

For gradual migration, use shim classes that maintain the old interface:

```python
from structure_net.config import LabConfigShim

# Drop-in replacement for old LabConfig
lab_config = LabConfigShim(
    project_name="my_project",
    device_ids=[0, 1]
)

# Works with old code expecting LabConfig
# But uses UnifiedConfig internally
```

### Strategy 3: Backward Compatibility Helpers

Generate old-style configs from unified config:

```python
from structure_net.config import UnifiedConfig

# Create unified config
config = UnifiedConfig()
config.experiment.project_name = "my_project"

# Get old-style configs when needed
lab_config = config.get_lab_config()
logging_config = config.get_logging_config()
```

## Environment Variables

The unified config automatically reads from environment variables:

```bash
# Set configuration via environment
export WANDB_PROJECT=my_project
export LOG_LEVEL=DEBUG
export MAX_PARALLEL_EXPERIMENTS=8
export DEVICE=cuda:0
export STRUCTURE_NET_DATA_ROOT=/path/to/data

# Python automatically picks these up
python my_script.py
```

## Configuration Files

Save and load configurations as YAML or JSON:

```python
# Save configuration
config = UnifiedConfig()
config.save("config.yaml")  # or config.json

# Load configuration
config = UnifiedConfig.from_file("config.yaml")
```

Example YAML configuration:
```yaml
storage:
  data_root: /data/structure_net
  results_dir: /data/structure_net/results
  
wandb:
  enabled: true
  project: my_research
  entity: my_team
  
logging:
  level: INFO
  enable_file: true
  module_levels:
    chromadb: WARNING
    
compute:
  device_ids: [0, 1, 2, 3]
  max_parallel_experiments: 8
  
experiment:
  project_name: architecture_search
  batch_size: 256
  learning_rate: 0.001
```

## Global Configuration

Use the global configuration singleton:

```python
from structure_net.config import get_config, set_config

# Get global config (creates default if needed)
config = get_config()

# Modify global config
config.experiment.project_name = "new_project"

# Replace global config
new_config = UnifiedConfig.from_file("production.yaml")
set_config(new_config)
```

## Common Migration Patterns

### Pattern 1: NAL Integration
```python
# Old way
lab_config = LabConfig(project_name="nal_experiment", ...)
lab = NeuralArchitectureLab(lab_config)

# New way
config = UnifiedConfig()
config.experiment.project_name = "nal_experiment"
lab_config = config.get_lab_config()  # For compatibility
lab = NeuralArchitectureLab(lab_config)

# Future way (when NAL is updated)
lab = NeuralArchitectureLab(config)
```

### Pattern 2: Logging Integration
```python
# Old way
logging_config = LoggingConfig(enable_wandb=True, ...)
logger = StandardizedLogger(logging_config)

# New way
config = UnifiedConfig()
config.wandb.enabled = True
logging_config = config.get_logging_config()  # For compatibility
logger = StandardizedLogger(logging_config)
```

### Pattern 3: Multi-Script Projects
```python
# config.py - Shared configuration
from structure_net.config import UnifiedConfig

def get_project_config():
    config = UnifiedConfig()
    config.experiment.project_name = "my_research"
    config.wandb.entity = "my_team"
    config.compute.device_ids = [0, 1, 2, 3]
    return config

# train.py
from config import get_project_config

config = get_project_config()
config.experiment.batch_size = 256  # Override for training

# evaluate.py  
from config import get_project_config

config = get_project_config()
config.experiment.batch_size = 512  # Override for evaluation
```

## Benefits of Migration

1. **Single Source of Truth**: All configuration in one place
2. **Type Safety**: Dataclasses provide type hints and validation
3. **Environment Support**: Automatic env var reading
4. **File Persistence**: Save/load as YAML or JSON
5. **Backward Compatible**: Works with existing code
6. **Auto Directory Creation**: Directories created automatically
7. **Hierarchical**: Logical grouping of related settings
8. **Extensible**: Easy to add new configuration sections

## Deprecation Timeline

- **Phase 1 (Current)**: Both old and new configs supported
- **Phase 2**: Deprecation warnings added to old configs
- **Phase 3**: Old configs removed, only shims remain
- **Phase 4**: Full migration to unified config

During the transition, all old code continues to work using compatibility layers.