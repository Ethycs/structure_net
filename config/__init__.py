"""
Structure Net Configuration System

A unified configuration system that provides a single source of truth
for all settings across the project. This replaces the scattered config
classes throughout the codebase.

Usage:
    from config import UnifiedConfig, get_config
    
    # Get global config
    config = get_config()
    
    # Or create custom config
    config = UnifiedConfig()
    config.experiment.project_name = "my_experiment"
"""

# Import unified configuration system
from .unified_config import (
    UnifiedConfig,
    StorageConfig,
    WandBConfig,
    LoggingConfig,
    ComputeConfig,
    ExperimentConfig,
    get_config,
    set_config,
    reset_config,
    create_lab_config,
    create_logging_config
)

# Import migration helpers
from .migration import (
    ConfigMigrationHelper,
    LabConfigShim,
    LoggingConfigShim,
    auto_migrate_config,
    migrate_config_files
)

__all__ = [
    # Unified configuration
    'UnifiedConfig',
    'StorageConfig', 
    'WandBConfig',
    'LoggingConfig',
    'ComputeConfig',
    'ExperimentConfig',
    'get_config',
    'set_config',
    'reset_config',
    
    # Backward compatibility
    'create_lab_config',
    'create_logging_config',
    
    # Migration helpers
    'ConfigMigrationHelper',
    'LabConfigShim',
    'LoggingConfigShim',
    'auto_migrate_config',
    'migrate_config_files'
]