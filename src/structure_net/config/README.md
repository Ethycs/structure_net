# Structure Net Configuration

This directory contains the unified configuration system for Structure Net.

## Overview

The configuration system provides a single source of truth for all settings across the project, replacing the previously scattered configuration classes throughout the codebase.

## Files

- `unified_config.py` - Main configuration classes and logic
- `migration.py` - Tools for migrating from old configuration systems
- `default.yaml` - Default configuration values
- `__init__.py` - Public API exports

## Usage

### Basic Usage

```python
from config import UnifiedConfig, get_config

# Use global config
config = get_config()
config.experiment.project_name = "my_experiment"

# Or create custom config
config = UnifiedConfig()
config.wandb.enabled = True
config.compute.device_ids = [0, 1]
```

### Environment Variables

Configuration automatically reads from environment:

```bash
export WANDB_PROJECT=my_project
export LOG_LEVEL=DEBUG
export MAX_PARALLEL_EXPERIMENTS=8
python train.py
```

### Configuration Files

Create `config.yaml` in project root:

```yaml
experiment:
  project_name: my_research
  batch_size: 256

compute:
  device_ids: [0, 1, 2, 3]
  max_parallel_experiments: 8
```

Then load it:

```python
config = UnifiedConfig.from_file("config.yaml")
```

## Configuration Structure

The unified config has 5 main sections:

1. **StorageConfig** - All file paths and directories
2. **WandBConfig** - Weights & Biases integration
3. **LoggingConfig** - Logging settings
4. **ComputeConfig** - Device and resource settings
5. **ExperimentConfig** - Experiment parameters

## Migration from Old Configs

If you have code using old configs like `LabConfig` or `LoggingConfig`:

```python
from config import ConfigMigrationHelper

# Automatic migration
unified = ConfigMigrationHelper.from_lab_config(old_lab_config)

# Or use compatibility shims
from config import LabConfigShim
lab_config = LabConfigShim(project_name="test")  # Works like old LabConfig
```

See `docs/CONFIG_MIGRATION_GUIDE.md` for detailed migration instructions.

## Best Practices

1. **Use environment variables** for deployment-specific settings (API keys, paths)
2. **Use config files** for experiment-specific settings
3. **Use code** for dynamic or computed settings
4. **Keep secrets out** - use environment variables for sensitive data
5. **Version control** your config files (except those with secrets)

## Examples

See `examples/unified_config_example.py` for comprehensive usage examples.