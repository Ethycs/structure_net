#!/usr/bin/env python3
"""
Configuration Migration Helpers

Provides utilities to migrate from the old scattered configuration system
to the new unified configuration system.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import warnings

from .unified_config import (
    UnifiedConfig, 
    StorageConfig, 
    WandBConfig, 
    LoggingConfig as UnifiedLoggingConfig,
    ComputeConfig,
    ExperimentConfig
)


class ConfigMigrationHelper:
    """Helper class for migrating old configurations to the unified system."""
    
    @staticmethod
    def from_lab_config(lab_config: 'LabConfig') -> UnifiedConfig:
        """
        Migrate from old LabConfig to UnifiedConfig.
        
        Args:
            lab_config: Old LabConfig instance
            
        Returns:
            UnifiedConfig with equivalent settings
        """
        unified = UnifiedConfig()
        
        # Map storage paths
        if hasattr(lab_config, 'results_dir'):
            unified.storage.results_dir = Path(lab_config.results_dir)
        
        # Map compute settings
        if hasattr(lab_config, 'device_ids'):
            unified.compute.device_ids = lab_config.device_ids
        if hasattr(lab_config, 'max_parallel_experiments'):
            unified.compute.max_parallel_experiments = lab_config.max_parallel_experiments
        if hasattr(lab_config, 'num_workers') and lab_config.num_workers is not None:
            unified.compute.num_workers = lab_config.num_workers
        if hasattr(lab_config, 'max_memory_per_experiment'):
            unified.compute.max_gpu_memory_percent = lab_config.max_memory_per_experiment * 100
        
        # Map auto-balancing settings
        if hasattr(lab_config, 'auto_balance'):
            unified.compute.enable_auto_balance = lab_config.auto_balance
        if hasattr(lab_config, 'target_cpu_percent'):
            unified.compute.target_cpu_utilization = lab_config.target_cpu_percent
        if hasattr(lab_config, 'max_cpu_percent'):
            unified.compute.max_cpu_percent = lab_config.max_cpu_percent
        if hasattr(lab_config, 'target_gpu_percent'):
            unified.compute.target_gpu_utilization = lab_config.target_gpu_percent
        if hasattr(lab_config, 'max_gpu_percent'):
            unified.compute.max_gpu_memory_percent = lab_config.max_gpu_percent
        if hasattr(lab_config, 'target_memory_percent'):
            unified.compute.max_memory_percent = lab_config.target_memory_percent
        if hasattr(lab_config, 'max_memory_percent'):
            unified.compute.max_memory_percent = lab_config.max_memory_percent
        
        # Map logging settings
        if hasattr(lab_config, 'log_level'):
            unified.logging.level = lab_config.log_level
        if hasattr(lab_config, 'log_file'):
            unified.logging.log_file = Path(lab_config.log_file) if lab_config.log_file else None
        if hasattr(lab_config, 'enable_wandb'):
            unified.wandb.enabled = lab_config.enable_wandb
        if hasattr(lab_config, 'module_log_levels') and lab_config.module_log_levels:
            unified.logging.module_levels = lab_config.module_log_levels
        
        # Map experiment settings
        if hasattr(lab_config, 'project_name'):
            unified.experiment.project_name = lab_config.project_name
            unified.wandb.project = lab_config.project_name
        if hasattr(lab_config, 'experiment_timeout'):
            # Convert seconds to minutes for consistency
            unified.experiment.timeout_minutes = lab_config.experiment_timeout // 60
        if hasattr(lab_config, 'checkpoint_frequency'):
            unified.experiment.checkpoint_interval = lab_config.checkpoint_frequency
        if hasattr(lab_config, 'save_all_models'):
            unified.experiment.save_all_checkpoints = lab_config.save_all_models
        if hasattr(lab_config, 'save_best_models'):
            unified.experiment.save_checkpoints = lab_config.save_best_models
        
        # Map scientific rigor settings
        if hasattr(lab_config, 'min_experiments_per_hypothesis'):
            unified.experiment.min_experiments = lab_config.min_experiments_per_hypothesis
        if hasattr(lab_config, 'require_statistical_significance'):
            unified.experiment.require_significance = lab_config.require_statistical_significance
        if hasattr(lab_config, 'significance_level'):
            unified.experiment.significance_level = lab_config.significance_level
        
        # Map adaptive exploration settings
        if hasattr(lab_config, 'enable_adaptive_hypotheses'):
            unified.experiment.enable_adaptive = lab_config.enable_adaptive_hypotheses
        if hasattr(lab_config, 'max_hypothesis_depth'):
            unified.experiment.max_hypothesis_depth = lab_config.max_hypothesis_depth
        
        # Map global settings
        if hasattr(lab_config, 'verbose'):
            unified.verbose = lab_config.verbose
        
        return unified
    
    @staticmethod
    def from_logging_config(logging_config: 'LoggingConfig') -> UnifiedConfig:
        """
        Migrate from old LoggingConfig to UnifiedConfig.
        
        Args:
            logging_config: Old LoggingConfig instance
            
        Returns:
            UnifiedConfig with equivalent settings
        """
        unified = UnifiedConfig()
        
        # Map storage paths
        if hasattr(logging_config, 'queue_dir'):
            unified.storage.queue_dir = Path(logging_config.queue_dir)
        if hasattr(logging_config, 'sent_dir'):
            unified.storage.sent_dir = Path(logging_config.sent_dir)
        if hasattr(logging_config, 'rejected_dir'):
            unified.storage.rejected_dir = Path(logging_config.rejected_dir)
        if hasattr(logging_config, 'chromadb_path'):
            unified.storage.chromadb_path = Path(logging_config.chromadb_path)
        
        # Map logging settings
        if hasattr(logging_config, 'enable_wandb'):
            unified.wandb.enabled = logging_config.enable_wandb
        if hasattr(logging_config, 'enable_chromadb'):
            unified.logging.enable_chromadb = logging_config.enable_chromadb
        
        # Map experiment settings
        if hasattr(logging_config, 'project_name'):
            unified.experiment.project_name = logging_config.project_name
            unified.wandb.project = logging_config.project_name
        
        return unified
    
    @staticmethod
    def from_multiple_configs(**configs) -> UnifiedConfig:
        """
        Merge multiple old configs into UnifiedConfig.
        
        Example:
            unified = ConfigMigrationHelper.from_multiple_configs(
                lab_config=old_lab_config,
                logging_config=old_logging_config,
                dataset_config=old_dataset_config
            )
        """
        unified = UnifiedConfig()
        
        # Process each config type
        if 'lab_config' in configs and configs['lab_config']:
            temp = ConfigMigrationHelper.from_lab_config(configs['lab_config'])
            unified = ConfigMigrationHelper._merge_configs(unified, temp)
        
        if 'logging_config' in configs and configs['logging_config']:
            temp = ConfigMigrationHelper.from_logging_config(configs['logging_config'])
            unified = ConfigMigrationHelper._merge_configs(unified, temp)
        
        # Add more config types as needed...
        
        return unified
    
    @staticmethod
    def _merge_configs(base: UnifiedConfig, override: UnifiedConfig) -> UnifiedConfig:
        """Merge two UnifiedConfig instances, with override taking precedence."""
        # This is a simple implementation - could be enhanced
        result = UnifiedConfig()
        
        # Merge each sub-config
        for attr in ['storage', 'wandb', 'logging', 'compute', 'experiment']:
            base_sub = getattr(base, attr)
            override_sub = getattr(override, attr)
            
            # Create new instance with merged values
            merged_dict = {**base_sub.__dict__, **override_sub.__dict__}
            setattr(result, attr, type(base_sub)(**merged_dict))
        
        # Merge global settings
        result.debug = override.debug if override.debug != base.debug else base.debug
        result.verbose = override.verbose if override.verbose != base.verbose else base.verbose
        
        return result


# Compatibility shims for old imports
class LabConfigShim:
    """
    Shim class that mimics old LabConfig but uses UnifiedConfig internally.
    
    This allows old code to continue working while using the new system.
    """
    
    def __init__(self, **kwargs):
        warnings.warn(
            "LabConfig is deprecated. Use UnifiedConfig instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create unified config
        self._unified = UnifiedConfig()
        
        # Map old kwargs to new structure
        if 'project_name' in kwargs:
            self._unified.experiment.project_name = kwargs['project_name']
            self._unified.wandb.project = kwargs['project_name']
        
        if 'results_dir' in kwargs:
            self._unified.storage.results_dir = Path(kwargs['results_dir'])
        
        if 'device_ids' in kwargs:
            self._unified.compute.device_ids = kwargs['device_ids']
        
        if 'max_parallel_experiments' in kwargs:
            self._unified.compute.max_parallel_experiments = kwargs['max_parallel_experiments']
        
        if 'experiment_timeout' in kwargs:
            self._unified.experiment.timeout_minutes = kwargs['experiment_timeout'] // 60
        
        if 'log_level' in kwargs:
            self._unified.logging.level = kwargs['log_level']
        
        if 'enable_wandb' in kwargs:
            self._unified.wandb.enabled = kwargs['enable_wandb']
        
        if 'verbose' in kwargs:
            self._unified.verbose = kwargs['verbose']
        
        if 'auto_balance' in kwargs:
            self._unified.compute.enable_auto_balance = kwargs['auto_balance']
        
        # Store any extra kwargs
        mapped_keys = [
            'project_name', 'results_dir', 'device_ids', 'max_parallel_experiments',
            'experiment_timeout', 'log_level', 'enable_wandb', 'verbose', 'auto_balance'
        ]
        self._extra_kwargs = {k: v for k, v in kwargs.items() if k not in mapped_keys}
    
    def __getattr__(self, name):
        """Proxy attribute access to unified config."""
        # Check common mappings
        mappings = {
            'project_name': lambda: self._unified.experiment.project_name,
            'results_dir': lambda: str(self._unified.storage.results_dir),
            'device_ids': lambda: self._unified.compute.device_ids,
            'max_parallel_experiments': lambda: self._unified.compute.max_parallel_experiments,
            'experiment_timeout': lambda: self._unified.experiment.timeout_minutes * 60,
            'log_level': lambda: self._unified.logging.level,
            'log_file': lambda: str(self._unified.logging.get_log_file_path(self._unified.storage)) if self._unified.logging.enable_file else None,
            'enable_wandb': lambda: self._unified.wandb.enabled,
            'verbose': lambda: self._unified.verbose,
            'auto_balance': lambda: self._unified.compute.enable_auto_balance,
            'target_cpu_percent': lambda: self._unified.compute.target_cpu_utilization,
            'max_cpu_percent': lambda: self._unified.compute.max_cpu_percent,
            'target_gpu_percent': lambda: self._unified.compute.target_gpu_utilization,
            'max_gpu_percent': lambda: self._unified.compute.max_gpu_memory_percent,
            'min_experiments_per_hypothesis': lambda: self._unified.experiment.min_experiments,
            'require_statistical_significance': lambda: self._unified.experiment.require_significance,
            'significance_level': lambda: self._unified.experiment.significance_level,
            'enable_adaptive_hypotheses': lambda: self._unified.experiment.enable_adaptive,
            'max_hypothesis_depth': lambda: self._unified.experiment.max_hypothesis_depth,
            'checkpoint_frequency': lambda: self._unified.experiment.checkpoint_interval,
            'save_all_models': lambda: self._unified.experiment.save_all_checkpoints,
            'save_best_models': lambda: self._unified.experiment.save_checkpoints
        }
        
        if name in mappings:
            return mappings[name]()
        
        # Check extra kwargs
        if name in self._extra_kwargs:
            return self._extra_kwargs[name]
        
        # Generate reasonable defaults for missing fields
        defaults = {
            'module_log_levels': None,
            'max_memory_per_experiment': 0.8,
            'target_memory_percent': 80.0,
            'max_memory_percent': 90.0
        }
        
        if name in defaults:
            return defaults[name]
        
        raise AttributeError(f"LabConfig has no attribute '{name}'")
    
    def to_unified_config(self) -> UnifiedConfig:
        """Get the underlying UnifiedConfig."""
        return self._unified


class LoggingConfigShim:
    """
    Shim class that mimics old LoggingConfig but uses UnifiedConfig internally.
    """
    
    def __init__(self, **kwargs):
        warnings.warn(
            "LoggingConfig is deprecated. Use UnifiedConfig instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create unified config
        self._unified = UnifiedConfig()
        
        # Map old kwargs to new structure
        if 'project_name' in kwargs:
            self._unified.experiment.project_name = kwargs['project_name']
            self._unified.wandb.project = kwargs['project_name']
        
        if 'queue_dir' in kwargs:
            self._unified.storage.queue_dir = Path(kwargs['queue_dir'])
        
        if 'enable_wandb' in kwargs:
            self._unified.wandb.enabled = kwargs['enable_wandb']
        
        if 'chromadb_path' in kwargs:
            self._unified.storage.chromadb_path = Path(kwargs['chromadb_path'])
        
        self._extra_kwargs = kwargs
    
    def __getattr__(self, name):
        """Proxy attribute access to unified config."""
        # Try to find in extra kwargs first
        if name in self._extra_kwargs:
            return self._extra_kwargs[name]
        
        # Check common mappings
        mappings = {
            'project_name': lambda: self._unified.experiment.project_name,
            'queue_dir': lambda: str(self._unified.storage.queue_dir),
            'sent_dir': lambda: str(self._unified.storage.sent_dir),
            'rejected_dir': lambda: str(self._unified.storage.rejected_dir),
            'enable_wandb': lambda: self._unified.wandb.enabled,
            'enable_chromadb': lambda: self._unified.logging.enable_chromadb,
            'chromadb_path': lambda: str(self._unified.storage.chromadb_path)
        }
        
        if name in mappings:
            return mappings[name]()
        
        raise AttributeError(f"LoggingConfig has no attribute '{name}'")
    
    def to_unified_config(self) -> UnifiedConfig:
        """Get the underlying UnifiedConfig."""
        return self._unified


# Auto-migration for old imports
def auto_migrate_config(old_config: Any) -> UnifiedConfig:
    """
    Automatically detect and migrate old config types.
    
    Args:
        old_config: Any old configuration object
        
    Returns:
        UnifiedConfig equivalent
    """
    # Check if it's already unified
    if isinstance(old_config, UnifiedConfig):
        return old_config
    
    # Check for shim classes
    if isinstance(old_config, (LabConfigShim, LoggingConfigShim)):
        return old_config.to_unified_config()
    
    # Check for old config types by attribute inspection
    if hasattr(old_config, 'results_dir') and hasattr(old_config, 'device_ids'):
        # Likely LabConfig
        return ConfigMigrationHelper.from_lab_config(old_config)
    
    if hasattr(old_config, 'queue_dir') and hasattr(old_config, 'chromadb_path'):
        # Likely LoggingConfig
        return ConfigMigrationHelper.from_logging_config(old_config)
    
    # Unknown config type
    warnings.warn(
        f"Unknown config type: {type(old_config)}. Creating default UnifiedConfig.",
        UserWarning
    )
    return UnifiedConfig()


# Example migration script
def migrate_config_files(old_config_path: Path, output_path: Path):
    """
    Migrate old config files to unified format.
    
    Example:
        migrate_config_files(
            Path("old_configs/lab_config.json"),
            Path("config.yaml")
        )
    """
    import json
    
    # Load old config
    with open(old_config_path) as f:
        old_data = json.load(f)
    
    # Detect type and migrate
    if 'results_dir' in old_data and 'device_ids' in old_data:
        # LabConfig format
        from src.neural_architecture_lab import LabConfig
        old_config = LabConfig(**old_data)
        unified = ConfigMigrationHelper.from_lab_config(old_config)
    elif 'queue_dir' in old_data:
        # LoggingConfig format
        from src.structure_net.logging.standardized_logging import LoggingConfig
        old_config = LoggingConfig(**old_data)
        unified = ConfigMigrationHelper.from_logging_config(old_config)
    else:
        raise ValueError(f"Unknown config format in {old_config_path}")
    
    # Save as unified config
    unified.save(output_path)
    print(f"Migrated {old_config_path} -> {output_path}")