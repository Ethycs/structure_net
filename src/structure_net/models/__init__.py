"""
Models module for Structure Net

Contains various neural network architectures and model implementations.

DEPRECATED: This entire module is deprecated. Please use the new component-based models:
    from structure_net.components.models import (
        MinimalModel,           # Replaces MinimalNetwork
        FiberBundleModel,       # Replaces FiberBundle
        FiberBundleConfig,      # Replaces FiberBundleConfig
        MultiScaleModel         # Replaces ModernMultiScaleNetwork
    )

The new models provide:
- Full component architecture integration
- Self-aware design with contracts and versioning
- Better integration with metrics, analyzers, and evolvers
- Enhanced functionality and flexibility

See individual module docstrings for specific migration guides.
"""

import warnings

# Issue deprecation warning when importing from this module
warnings.warn(
    "The structure_net.models module is deprecated. "
    "Please use structure_net.components.models instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

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
