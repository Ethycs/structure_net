"""
Gauge Theory for Neural Network Evolution

This package contains modules related to the application of gauge theory
principles to neural network optimization, compression, and analysis.
"""

from .gauge import (
    GaugeInvariantOptimizer,
    compress_network_gauge_aware,
    GaugeAwareNAS,
    GaugeAugmentedTraining,
    fuse_models_gauge_aware,
    CatastropheMinimizingGauge,
    GaugeInvariantMetrics
)

__all__ = [
    "GaugeInvariantOptimizer",
    "compress_network_gauge_aware",
    "GaugeAwareNAS",
    "GaugeAugmentedTraining",
    "fuse_models_gauge_aware",
    "CatastropheMinimizingGauge",
    "GaugeInvariantMetrics"
]
