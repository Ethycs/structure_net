"""
Models module for Structure Net

Contains various neural network architectures and model implementations.
"""

from .fiber_bundle_network import (
    FiberBundle, 
    FiberBundleConfig, 
    FiberBundleBuilder,
    StructuredConnection
)
from .minimal_network import MinimalNetwork
from .modern_multi_scale_network import ModernMultiScaleNetwork

__all__ = [
    'FiberBundle',
    'FiberBundleConfig', 
    'FiberBundleBuilder',
    'StructuredConnection',
    'MinimalNetwork',
    'ModernMultiScaleNetwork'
]
