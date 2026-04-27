"""
Configuration adapter for Data Factory components.

This module provides smooth migration from old config classes to the new unified
configuration system.
"""

from typing import Union
from pathlib import Path

from .search.chroma_client import ChromaConfig, ChromaSearchClient
from .time_series_storage import TimeSeriesConfig, TimeSeriesStorage
from config import UnifiedConfig, get_config, auto_migrate_config


def ensure_chroma_config(config: Union[ChromaConfig, UnifiedConfig, dict, None] = None) -> ChromaConfig:
    """
    Ensure we have a valid ChromaConfig, converting from other formats if needed.
    
    Args:
        config: Can be:
            - ChromaConfig: Used as-is
            - UnifiedConfig: Converted to ChromaConfig
            - dict: Converted to ChromaConfig
            - None: Uses global config
    
    Returns:
        ChromaConfig instance
    """
    if config is None:
        # Use global unified config
        unified = get_config()
        return unified.get_chroma_config()
    
    if isinstance(config, ChromaConfig):
        # Already the right type
        return config
    
    if isinstance(config, UnifiedConfig):
        # Convert from unified
        return config.get_chroma_config()
    
    if isinstance(config, dict):
        # Try to determine type from dict contents
        if 'storage' in config or 'wandb' in config:
            # Looks like unified config dict
            unified = UnifiedConfig.from_dict(config)
            return unified.get_chroma_config()
        else:
            # Assume it's ChromaConfig kwargs
            return ChromaConfig(**config)
    
    # Try auto-migration for unknown types
    unified = auto_migrate_config(config)
    return unified.get_chroma_config()


def ensure_timeseries_config(config: Union[TimeSeriesConfig, UnifiedConfig, dict, None] = None) -> TimeSeriesConfig:
    """
    Ensure we have a valid TimeSeriesConfig, converting from other formats if needed.
    
    Args:
        config: Can be:
            - TimeSeriesConfig: Used as-is
            - UnifiedConfig: Converted to TimeSeriesConfig
            - dict: Converted to TimeSeriesConfig
            - None: Uses global config
    
    Returns:
        TimeSeriesConfig instance
    """
    if config is None:
        # Use global unified config
        unified = get_config()
        return unified.get_timeseries_config()
    
    if isinstance(config, TimeSeriesConfig):
        # Already the right type
        return config
    
    if isinstance(config, UnifiedConfig):
        # Convert from unified
        return config.get_timeseries_config()
    
    if isinstance(config, dict):
        # Try to determine type from dict contents
        if 'storage' in config or 'wandb' in config:
            # Looks like unified config dict
            unified = UnifiedConfig.from_dict(config)
            return unified.get_timeseries_config()
        else:
            # Assume it's TimeSeriesConfig kwargs
            return TimeSeriesConfig(**config)
    
    # Try auto-migration for unknown types
    unified = auto_migrate_config(config)
    return unified.get_timeseries_config()


def update_chroma_client_for_unified_config():
    """
    Monkey-patch ChromaSearchClient to accept UnifiedConfig.
    
    This allows existing code to work with both old and new config systems.
    """
    from .search import chroma_client
    
    # Store original init
    original_init = chroma_client.ChromaSearchClient.__init__
    
    def new_init(self, config: Union[ChromaConfig, UnifiedConfig, dict, None] = None):
        """Enhanced init that accepts multiple config types."""
        chroma_config = ensure_chroma_config(config)
        original_init(self, chroma_config)
    
    # Replace init
    chroma_client.ChromaSearchClient.__init__ = new_init
    
    # Also update the type hint in the class
    chroma_client.ChromaSearchClient.__init__.__annotations__['config'] = Union[ChromaConfig, UnifiedConfig, dict, None]


def update_timeseries_storage_for_unified_config():
    """
    Monkey-patch TimeSeriesStorage to accept UnifiedConfig.
    
    This allows existing code to work with both old and new config systems.
    """
    from . import time_series_storage
    
    # Store original init
    original_init = time_series_storage.TimeSeriesStorage.__init__
    
    def new_init(self, config: Union[TimeSeriesConfig, UnifiedConfig, dict, None] = None):
        """Enhanced init that accepts multiple config types."""
        ts_config = ensure_timeseries_config(config)
        original_init(self, ts_config)
    
    # Replace init
    time_series_storage.TimeSeriesStorage.__init__ = new_init
    
    # Also update the type hint in the class
    time_series_storage.TimeSeriesStorage.__init__.__annotations__['config'] = Union[TimeSeriesConfig, UnifiedConfig, dict, None]


# Auto-apply the patches when this module is imported
update_chroma_client_for_unified_config()
update_timeseries_storage_for_unified_config()