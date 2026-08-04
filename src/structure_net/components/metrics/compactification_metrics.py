"""Compatibility exports for compactification metrics.

The implementations live in focused modules and conform to :class:`BaseMetric`.
This module remains as an import-compatible facade for existing callers.
"""

from .compression_ratio_metric import CompressionRatioMetric
from .memory_efficiency_metric import MemoryEfficiencyMetric
from .patch_effectiveness_metric import PatchEffectivenessMetric
from .reconstruction_quality_metric import ReconstructionQualityMetric

__all__ = [
    "CompressionRatioMetric",
    "PatchEffectivenessMetric",
    "MemoryEfficiencyMetric",
    "ReconstructionQualityMetric",
]
