#!/usr/bin/env python3
"""
Core Layer Definitions - Compatibility Module

This module provides backward compatibility imports for the refactored layer components.
All layer implementations have been moved to structure_net.components.layers.

DEPRECATED: Please import directly from structure_net.components.layers instead.
"""

import warnings

# Import all layer components for backward compatibility
from ..components.layers import (
    StandardSparseLayer,
    ExtremaAwareSparseLayer,
    TemporaryPatchLayer,
    SparseLinear,
    StructuredLinear
)

# Issue deprecation warning
warnings.warn(
    "Importing layers from structure_net.core.layers is deprecated. "
    "Please import from structure_net.components.layers instead.",
    DeprecationWarning,
    stacklevel=2
)

# Export layer definitions for compatibility
__all__ = [
    'StandardSparseLayer',
    'ExtremaAwareSparseLayer', 
    'TemporaryPatchLayer',
    'SparseLinear',
    'StructuredLinear'
]