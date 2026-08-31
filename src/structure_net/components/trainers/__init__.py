"""
Trainer Components

This module contains training-related components including initialization methods,
optimization strategies, and training orchestration.
"""

from .lsuv_trainer import LSUVTrainer
from .gauge_theory_trainer import GaugeTheoryTrainer
from .causal_language_model_trainer import CausalLanguageModelTrainer
from .continuous_autoregressive_trainer import ContinuousAutoregressiveTrainer

__all__ = [
    'LSUVTrainer',
    'GaugeTheoryTrainer',
    'CausalLanguageModelTrainer',
    'ContinuousAutoregressiveTrainer',
]
