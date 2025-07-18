"""
Evolver components for network evolution.

Evolvers execute evolution plans to modify network structure,
parameters, and behavior.
"""

from .compactification_evolver import CompactificationEvolver
from .input_highway_evolver import InputHighwayEvolver
from .tournament_evolver import TournamentEvolver

__all__ = [
    'CompactificationEvolver',
    'InputHighwayEvolver',
    'TournamentEvolver'
]