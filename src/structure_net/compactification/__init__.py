#!/usr/bin/env python3
"""
Homologically-Guided Network Compactification

This module implements the revolutionary sparse network architecture with:
- 2-5% total sparsity with 20% dense patches at extrema
- Chain complex analysis for principled compression
- Input highway preservation system
- Layer-wise compactification for constant memory training
- Topological data analysis for structure guidance

Key Components:
- HomologicalCompactNetwork: Main architecture
- InputHighwaySystem: Information preservation
- ChainMapAnalyzer: Topological guidance
- PatchCompactifier: Efficient storage
- ExtremaDetector: Critical point identification
"""

from .homological_network import (
    HomologicalCompactNetwork,
    InputHighwaySystem,
    ChainMapAnalyzer,
    create_homological_network
)

from .patch_compactification import (
    PatchCompactifier,
    CompactLayer,
    ExtremaDetector
)

__all__ = [
    # Core architecture
    'HomologicalCompactNetwork',
    'InputHighwaySystem', 
    'ChainMapAnalyzer',
    'create_homological_network',
    
    # Compactification
    'PatchCompactifier',
    'CompactLayer',
    'ExtremaDetector'
]

__version__ = "1.0.0"
