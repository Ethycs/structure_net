"""
Configuration module for Structure Net.

This module now redirects to the top-level config directory.
Import this before torch to ensure proper environment setup.
"""

# Import environment setup first (keep this local for torch setup)
from .environment import setup_cuda_devices

# Import everything from top-level config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config import (
    # Unified configuration
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
    # Migration helpers
    ConfigMigrationHelper,
    LabConfigShim,
    LoggingConfigShim,
    auto_migrate_config,
    migrate_config_files
)

__all__ = [
    # Environment setup
    'setup_cuda_devices',
    
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