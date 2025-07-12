"""
Trainer Components

This module contains training-related components including initialization methods,
optimization strategies, and training orchestration.
"""

from .lsuv_trainer import LSUVTrainer
from .gauge_theory_trainer import GaugeTheoryTrainer

__all__ = [
    'LSUVTrainer',
    'GaugeTheoryTrainer'
]