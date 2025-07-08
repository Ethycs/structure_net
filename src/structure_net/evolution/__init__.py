"""
Evolution Module

This module provides network evolution capabilities using the canonical
model standard for perfect compatibility across the project.
"""

from .network_evolver import OptimalGrowthEvolver
from .extrema_analyzer import analyze_layer_extrema, detect_network_extrema
from .information_theory import estimate_mi_proxy, analyze_information_flow
from .integrated_growth_system import IntegratedGrowthSystem
from .advanced_layers import (
    ThresholdConfig,
    MetricsConfig,
    ExtremaAwareSparseLayer, 
    TemporaryPatchModule, 
    lsuv_init_layer, 
    apply_neuron_sorting, 
    sort_all_network_layers
)
from .adaptive_learning_rates import (
    ExponentialBackoffScheduler,
    LayerwiseAdaptiveRates,
    SoftClampingScheduler,
    ScaleDependentRates,
    GrowthPhaseScheduler,
    AdaptiveLearningRateManager,
    create_adaptive_training_loop,
    # Advanced combination systems
    ExtremaPhaseScheduler,
    LayerAgeAwareLR,
    MultiScaleLearning,
    UnifiedAdaptiveLearning
)

__version__ = "1.2.0"  # Updated version with advanced components
__author__ = "Structure Net Team"

__all__ = [
    'OptimalGrowthEvolver',
    'analyze_layer_extrema',
    'detect_network_extrema', 
    'estimate_mi_proxy',
    'analyze_information_flow',
    'IntegratedGrowthSystem',
    'ThresholdConfig',
    'MetricsConfig',
    'ExtremaAwareSparseLayer',
    'TemporaryPatchModule',
    'lsuv_init_layer',
    'apply_neuron_sorting',
    'sort_all_network_layers',
    # Adaptive Learning Rate Strategies
    'ExponentialBackoffScheduler',
    'LayerwiseAdaptiveRates',
    'SoftClampingScheduler',
    'ScaleDependentRates',
    'GrowthPhaseScheduler',
    'AdaptiveLearningRateManager',
    'create_adaptive_training_loop',
    # Advanced combination systems
    'ExtremaPhaseScheduler',
    'LayerAgeAwareLR',
    'MultiScaleLearning',
    'UnifiedAdaptiveLearning'
]
