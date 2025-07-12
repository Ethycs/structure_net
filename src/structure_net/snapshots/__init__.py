"""
Snapshot management for multi-scale networks.

DEPRECATED: This module is deprecated. Please use the new component-based architecture:
    from structure_net.components.orchestrators import SnapshotOrchestrator
    from structure_net.components.strategies import SnapshotStrategy
    from structure_net.components.metrics import SnapshotMetric
    
The new components provide:
- Better integration with the evolution system
- Intelligent snapshot strategies
- Performance analysis and metrics
- Self-aware component design

See individual component docstrings for migration examples.
"""

import warnings

# Issue deprecation warning when importing from this module
warnings.warn(
    "The structure_net.snapshots module is deprecated. "
    "Please use structure_net.components.orchestrators.SnapshotOrchestrator and related components instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

from .snapshot_manager import SnapshotManager

__all__ = [
    "SnapshotManager"
]
