"""
Core components for multi-scale neural networks.
"""

from .minimal_network import MinimalNetwork, create_minimal_network
from .growth_scheduler import GrowthScheduler, StructuralLimits
from .connection_router import ConnectionRouter, ParsimonousRouter

__all__ = [
    "MinimalNetwork",
    "create_minimal_network", 
    "GrowthScheduler",
    "StructuralLimits",
    "ConnectionRouter",
    "ParsimonousRouter"
]
