#!/usr/bin/env python3
"""
Example: Using the Unified Configuration System

This demonstrates how to use the new unified configuration system that
replaces all the scattered config classes throughout the codebase.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    UnifiedConfig,
    get_config,
    set_config,
    create_lab_config,
    ConfigMigrationHelper
)


def example_1_basic_usage():
    """Example 1: Basic usage of unified config."""
    print("=" * 60)
    print("Example 1: Basic Unified Configuration")
    print("=" * 60)
    
    # Create a unified config with defaults
    config = UnifiedConfig()
    
    print("Default configuration:")
    print(f"  Data root: {config.storage.data_root}")
    print(f"  WandB project: {config.wandb.project}")
    print(f"  Log level: {config.logging.level}")
    print(f"  Max parallel: {config.compute.max_parallel_experiments}")
    print(f"  Experiment name: {config.experiment.get_experiment_name()}")
    
    # Modify configuration
    config.wandb.project = "my_custom_project"
    config.compute.max_parallel_experiments = 4
    config.experiment.batch_size = 256
    
    print("\nModified configuration:")
    print(f"  WandB project: {config.wandb.project}")
    print(f"  Max parallel: {config.compute.max_parallel_experiments}")
    print(f"  Batch size: {config.experiment.batch_size}")


def example_2_environment_variables():
    """Example 2: Configuration from environment variables."""
    print("\n" + "=" * 60)
    print("Example 2: Environment Variable Configuration")
    print("=" * 60)
    
    # Set some environment variables
    os.environ['WANDB_PROJECT'] = 'env_project'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    os.environ['MAX_PARALLEL_EXPERIMENTS'] = '8'
    os.environ['DEVICE'] = 'cuda:0'
    
    # Create config - it picks up env vars automatically
    config = UnifiedConfig()
    
    print("Configuration from environment:")
    print(f"  WandB project: {config.wandb.project}")
    print(f"  Log level: {config.logging.level}")
    print(f"  Max parallel: {config.compute.max_parallel_experiments}")
    print(f"  Device: {config.compute.device}")
    
    # Clean up
    for key in ['WANDB_PROJECT', 'LOG_LEVEL', 'MAX_PARALLEL_EXPERIMENTS', 'DEVICE']:
        os.environ.pop(key, None)


def example_3_file_based_config():
    """Example 3: Loading and saving configuration files."""
    print("\n" + "=" * 60)
    print("Example 3: File-Based Configuration")
    print("=" * 60)
    
    # Create a custom config
    config = UnifiedConfig()
    config.experiment.project_name = "my_experiment"
    config.wandb.tags = ["test", "example"]
    config.compute.device_ids = [0, 1]
    config.logging.module_levels = {
        'chromadb': 'ERROR',
        'my_module': 'DEBUG'
    }
    
    # Save to YAML
    yaml_path = Path("example_config.yaml")
    config.save(yaml_path)
    print(f"Saved configuration to {yaml_path}")
    
    # Save to JSON
    json_path = Path("example_config.json")
    config.save(json_path)
    print(f"Saved configuration to {json_path}")
    
    # Load from file
    loaded_config = UnifiedConfig.from_file(yaml_path)
    print(f"\nLoaded configuration:")
    print(f"  Project: {loaded_config.experiment.project_name}")
    print(f"  Tags: {loaded_config.wandb.tags}")
    print(f"  Device IDs: {loaded_config.compute.device_ids}")
    
    # Clean up
    yaml_path.unlink()
    json_path.unlink()


def example_4_backward_compatibility():
    """Example 4: Backward compatibility with old configs."""
    print("\n" + "=" * 60)
    print("Example 4: Backward Compatibility")
    print("=" * 60)
    
    # Create unified config
    config = UnifiedConfig()
    config.experiment.project_name = "compat_test"
    config.compute.device_ids = [0, 1, 2]
    
    # Get old-style configs from unified config
    lab_config = config.get_lab_config()
    logging_config = config.get_logging_config()
    
    print("Generated backward-compatible configs:")
    print(f"  LabConfig.project_name: {lab_config.project_name}")
    print(f"  LabConfig.device_ids: {lab_config.device_ids}")
    print(f"  LoggingConfig.chromadb_path: {logging_config.chromadb_path}")
    
    # Use the compatibility helper
    lab_config_compat = create_lab_config(
        project_name="override_project",
        max_parallel_experiments=16
    )
    print(f"\nCreated LabConfig with overrides:")
    print(f"  Project: {lab_config_compat.project_name}")
    print(f"  Max parallel: {lab_config_compat.max_parallel_experiments}")


def example_5_global_config():
    """Example 5: Using the global configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Global Configuration")
    print("=" * 60)
    
    # Get the global config (creates default if needed)
    config = get_config()
    print(f"Global config project: {config.experiment.project_name}")
    
    # Modify global config
    config.experiment.project_name = "global_project"
    config.verbose = True
    
    # Get it again - same instance
    config2 = get_config()
    print(f"Modified global project: {config2.experiment.project_name}")
    print(f"Verbose mode: {config2.verbose}")
    
    # Set a completely new global config
    new_config = UnifiedConfig()
    new_config.experiment.project_name = "replaced_project"
    set_config(new_config)
    
    config3 = get_config()
    print(f"Replaced global project: {config3.experiment.project_name}")


def example_6_config_composition():
    """Example 6: Composing configurations."""
    print("\n" + "=" * 60)
    print("Example 6: Configuration Composition")
    print("=" * 60)
    
    # Create base config for all experiments
    base_config = UnifiedConfig()
    base_config.wandb.project = "structure_net_research"
    base_config.wandb.entity = "my_team"
    base_config.logging.level = "INFO"
    base_config.compute.pin_memory = True
    
    print("Base configuration:")
    print(f"  WandB: {base_config.wandb.entity}/{base_config.wandb.project}")
    print(f"  Logging: {base_config.logging.level}")
    
    # Create experiment-specific config
    exp_config = UnifiedConfig.from_dict({
        'experiment': {
            'project_name': 'specific_experiment',
            'batch_size': 512,
            'learning_rate': 0.0001
        },
        'compute': {
            'device_ids': [0, 1],
            'max_parallel_experiments': 2
        },
        'debug': True
    })
    
    print("\nExperiment-specific overrides:")
    print(f"  Batch size: {exp_config.experiment.batch_size}")
    print(f"  Learning rate: {exp_config.experiment.learning_rate}")
    print(f"  Debug mode: {exp_config.debug}")
    
    # In practice, you'd merge these configs
    # For now, showing how different configs can coexist


def example_7_migration():
    """Example 7: Migrating from old configs."""
    print("\n" + "=" * 60)
    print("Example 7: Configuration Migration")
    print("=" * 60)
    
    # Simulate old LabConfig data
    old_lab_config_data = {
        'project_name': 'old_project',
        'results_dir': '/tmp/results',
        'device_ids': [0, 1],
        'max_parallel_experiments': 4,
        'log_level': 'DEBUG',
        'enable_wandb': True,
        'verbose': True
    }
    
    # Create a mock old config (normally you'd have the actual old class)
    class MockLabConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    old_config = MockLabConfig(**old_lab_config_data)
    
    # Migrate to unified config
    unified = ConfigMigrationHelper.from_lab_config(old_config)
    
    print("Migrated configuration:")
    print(f"  Project: {unified.experiment.project_name}")
    print(f"  Results dir: {unified.storage.results_dir}")
    print(f"  Device IDs: {unified.compute.device_ids}")
    print(f"  Log level: {unified.logging.level}")
    print(f"  WandB enabled: {unified.wandb.enabled}")
    print(f"  Verbose: {unified.verbose}")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_usage()
    example_2_environment_variables()
    example_3_file_based_config()
    example_4_backward_compatibility()
    example_5_global_config()
    example_6_config_composition()
    example_7_migration()
    
    print("\n" + "=" * 60)
    print("Unified configuration examples complete!")
    print("=" * 60)
    print("\nKey benefits:")
    print("1. Single source of truth for all configuration")
    print("2. Environment variable support out of the box")
    print("3. File-based config (YAML/JSON) with validation")
    print("4. Backward compatibility with old configs")
    print("5. Composable configurations for complex setups")
    print("6. Automatic directory creation")
    print("7. Type safety with dataclasses")