#!/usr/bin/env python3
"""
Homologically-Guided Network Compactification

This module implements the revolutionary sparse network architecture with:
- 2-5% total sparsity with 20% dense patches at extrema
- Chain complex analysis for principled compression
- Input highway preservation system
- Layer-wise compactification for constant memory training
- Topological data analysis for structure guidance

DEPRECATED: This module is deprecated. Please use the new component-based
implementations in:
- structure_net.components.models - For network architectures
- structure_net.components.analyzers - For analysis components
- structure_net.components.strategies - For compactification strategies
- structure_net.components.evolvers - For evolution components

Key Components:
- HomologicalCompactNetwork: Main architecture -> Use components.models
- InputHighwaySystem: Information preservation -> Use components.models
- ChainMapAnalyzer: Topological guidance -> Use components.analyzers
- PatchCompactifier: Efficient storage -> Use components.evolvers
- ExtremaDetector: Critical point identification -> Use components.analyzers
"""

import warnings

warnings.warn(
    "The compactification module is deprecated. "
    "Please use the new component-based implementations in structure_net.components.",
    DeprecationWarning,
    stacklevel=2
)

# Import new components for compatibility
from ..components.analyzers import CompactificationAnalyzer
from ..components.evolvers import CompactificationEvolver  
from ..components.strategies import CompactificationStrategy
from ..components.metrics import CompactificationMetrics

# Import old implementations (deprecated)
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

def _compactification_migration_guide():
    """Print migration guide for compactification components."""
    print("""
    === Compactification Migration Guide ===
    
    The compactification module has been refactored into components.
    
    Migration examples:
    
    1. ExtremaDetector -> CompactificationAnalyzer
       OLD: from structure_net.compactification import ExtremaDetector
       NEW: from structure_net.components.analyzers import CompactificationAnalyzer
    
    2. PatchCompactifier -> CompactificationEvolver
       OLD: from structure_net.compactification import PatchCompactifier
       NEW: from structure_net.components.evolvers import CompactificationEvolver
    
    3. CompactLayer -> Use layers in components.layers
       OLD: from structure_net.compactification import CompactLayer
       NEW: from structure_net.components.layers import [appropriate layer type]
    
    4. For strategic decisions -> CompactificationStrategy
       NEW: from structure_net.components.strategies import CompactificationStrategy
    
    The new components offer:
    - Better modularity and composition
    - Formal contracts defining inputs/outputs
    - Resource requirement declarations
    - Improved state management
    - Better integration with the component system
    """)

__all__ = [
    # Core architecture (deprecated)
    'HomologicalCompactNetwork',
    'InputHighwaySystem', 
    'ChainMapAnalyzer',
    'create_homological_network',
    
    # Compactification (deprecated)
    'PatchCompactifier',
    'CompactLayer',
    'ExtremaDetector',
    
    # New components (recommended)
    'CompactificationAnalyzer',
    'CompactificationEvolver',
    'CompactificationStrategy',
    'CompactificationMetrics',
    
    # Migration helper
    '_compactification_migration_guide'
]

__version__ = "1.0.0"
