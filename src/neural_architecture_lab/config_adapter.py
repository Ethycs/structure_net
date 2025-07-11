"""
Configuration adapter for Neural Architecture Lab.

This module provides smooth migration from old LabConfig to the new unified
configuration system.
"""

from typing import Union
from pathlib import Path

from .core import LabConfig
from config import UnifiedConfig, get_config, auto_migrate_config


def ensure_lab_config(config: Union[LabConfig, UnifiedConfig, dict, None] = None) -> LabConfig:
    """
    Ensure we have a valid LabConfig, converting from other formats if needed.
    
    Args:
        config: Can be:
            - LabConfig: Used as-is
            - UnifiedConfig: Converted to LabConfig
            - dict: Converted to LabConfig
            - None: Uses global config
    
    Returns:
        LabConfig instance
    """
    if config is None:
        # Use global unified config
        unified = get_config()
        return unified.get_lab_config()
    
    if isinstance(config, LabConfig):
        # Already the right type
        return config
    
    if isinstance(config, UnifiedConfig):
        # Convert from unified
        return config.get_lab_config()
    
    if isinstance(config, dict):
        # Try to determine type from dict contents
        if 'storage' in config or 'wandb' in config:
            # Looks like unified config dict
            unified = UnifiedConfig.from_dict(config)
            return unified.get_lab_config()
        else:
            # Assume it's LabConfig kwargs
            return LabConfig(**config)
    
    # Try auto-migration for unknown types
    unified = auto_migrate_config(config)
    return unified.get_lab_config()


def update_lab_for_unified_config():
    """
    Monkey-patch the NeuralArchitectureLab to accept UnifiedConfig.
    
    This allows existing code to work with both old and new config systems.
    """
    from . import lab
    
    # Store original init
    original_init = lab.NeuralArchitectureLab.__init__
    
    def new_init(self, config: Union[LabConfig, UnifiedConfig, dict, None] = None):
        """Enhanced init that accepts multiple config types."""
        lab_config = ensure_lab_config(config)
        original_init(self, lab_config)
    
    # Replace init
    lab.NeuralArchitectureLab.__init__ = new_init
    
    # Also update the type hint in the class
    lab.NeuralArchitectureLab.__init__.__annotations__['config'] = Union[LabConfig, UnifiedConfig, dict, None]


# Auto-apply the patch when this module is imported
update_lab_for_unified_config()