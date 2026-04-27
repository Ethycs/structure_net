"""
Seed Search Module

This module provides architecture generation and, formerly, GPU-accelerated
seed hunting capabilities. The seed hunting is now managed by the
Neural Architecture Lab (NAL).
"""

from .architecture_generator import ArchitectureGenerator

__version__ = "2.0.0"
__author__ = "Structure Net Team"

__all__ = [
    'ArchitectureGenerator'
]
