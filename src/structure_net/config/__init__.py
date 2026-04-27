"""
Configuration module for Structure Net.

Holds environment setup, the unified configuration system, and migration
helpers. `setup_cuda_devices` must be importable before torch is imported
elsewhere, so keep it at the top of this module.
"""

from .environment import setup_cuda_devices

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
    create_logging_config,
)

from .migration import (
    ConfigMigrationHelper,
    LabConfigShim,
    LoggingConfigShim,
    auto_migrate_config,
    migrate_config_files,
)

__all__ = [
    'setup_cuda_devices',
    'UnifiedConfig',
    'StorageConfig',
    'WandBConfig',
    'LoggingConfig',
    'ComputeConfig',
    'ExperimentConfig',
    'get_config',
    'set_config',
    'reset_config',
    'create_lab_config',
    'create_logging_config',
    'ConfigMigrationHelper',
    'LabConfigShim',
    'LoggingConfigShim',
    'auto_migrate_config',
    'migrate_config_files',
]
