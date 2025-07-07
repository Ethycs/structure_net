"""
Evolution Module

This module provides network evolution capabilities using the canonical
model standard for perfect compatibility across the project.
"""

from .network_evolver import OptimalGrowthEvolver
from .extrema_analyzer import analyze_layer_extrema, detect_network_extrema
from .information_theory import estimate_mi_proxy, analyze_information_flow

__version__ = "1.0.0"
__author__ = "Structure Net Team"

__all__ = [
    'OptimalGrowthEvolver',
    'analyze_layer_extrema',
    'detect_network_extrema', 
    'estimate_mi_proxy',
    'analyze_information_flow'
]
