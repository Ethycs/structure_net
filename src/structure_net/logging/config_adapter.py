"""
Configuration adapter for logging system.

This module provides smooth migration from old LoggingConfig to the new unified
configuration system.
"""

from typing import Union
from pathlib import Path

from .standardized_logging import LoggingConfig, StandardizedLogger
from config import UnifiedConfig, get_config, auto_migrate_config


def ensure_logging_config(config: Union[LoggingConfig, UnifiedConfig, dict, None] = None) -> LoggingConfig:
    """
    Ensure we have a valid LoggingConfig, converting from other formats if needed.
    
    Args:
        config: Can be:
            - LoggingConfig: Used as-is
            - UnifiedConfig: Converted to LoggingConfig
            - dict: Converted to LoggingConfig
            - None: Uses global config
    
    Returns:
        LoggingConfig instance
    """
    if config is None:
        # Use global unified config
        unified = get_config()
        return unified.get_logging_config()
    
    if isinstance(config, LoggingConfig):
        # Already the right type
        return config
    
    if isinstance(config, UnifiedConfig):
        # Convert from unified
        return config.get_logging_config()
    
    if isinstance(config, dict):
        # Try to determine type from dict contents
        if 'storage' in config or 'wandb' in config:
            # Looks like unified config dict
            unified = UnifiedConfig.from_dict(config)
            return unified.get_logging_config()
        else:
            # Assume it's LoggingConfig kwargs
            return LoggingConfig(**config)
    
    # Try auto-migration for unknown types
    unified = auto_migrate_config(config)
    return unified.get_logging_config()


def update_logger_for_unified_config():
    """
    Monkey-patch the StandardizedLogger to accept UnifiedConfig.
    
    This allows existing code to work with both old and new config systems.
    """
    from . import standardized_logging
    
    # Store original init
    original_init = standardized_logging.StandardizedLogger.__init__
    
    def new_init(self, config: Union[LoggingConfig, UnifiedConfig, dict, None] = None):
        """Enhanced init that accepts multiple config types."""
        logging_config = ensure_logging_config(config)
        original_init(self, logging_config)
    
    # Replace init
    standardized_logging.StandardizedLogger.__init__ = new_init
    
    # Also update the type hint in the class
    standardized_logging.StandardizedLogger.__init__.__annotations__['config'] = Union[LoggingConfig, UnifiedConfig, dict, None]


# Auto-apply the patch when this module is imported
update_logger_for_unified_config()