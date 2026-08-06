"""
Evolver components for network evolution.

Evolvers execute evolution plans to modify network structure,
parameters, and behavior.
"""

from .compactification_evolver import CompactificationEvolver
from .input_highway_evolver import InputHighwayEvolver
from .feedback_growth_evolver import FeedbackGrowthEvolver
from .tournament_evolver import TournamentEvolver

__all__ = [
    'CompactificationEvolver',
    'InputHighwayEvolver',
    'FeedbackGrowthEvolver',
    'TournamentEvolver'
]
