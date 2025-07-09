#!/usr/bin/env python3
"""
Composable Network Evolution Components

This module provides concrete implementations of the composable evolution
interfaces, enabling modular and configurable network evolution systems.

Key components:
- Analyzers: Network analysis components
- Strategies: Growth strategy implementations  
- Evolution Systems: Complete coordinated systems
- Factories: Component creation utilities
"""

# Import core interfaces
from ..interfaces import (
    NetworkComponent, NetworkAnalyzer, GrowthStrategy, 
    LearningRateStrategy, NetworkTrainer, NetworkEvolutionSystem,
    NetworkContext, AnalysisResult, GrowthAction, ActionType
)

# Import concrete analyzers
from .analyzers import (
    StandardExtremaAnalyzer,
    NetworkStatsAnalyzer,
    SimpleInformationFlowAnalyzer
)

# Import concrete strategies
from .strategies import (
    ExtremaGrowthStrategy,
    InformationFlowGrowthStrategy,
    ResidualBlockGrowthStrategy,
    HybridGrowthStrategy
)

# Import evolution systems
from .evolution_system import (
    StandardNetworkTrainer,
    ComposableEvolutionSystem,
    create_standard_evolution_system,
    create_extrema_focused_system,
    create_hybrid_system
)

# Export all public components
__all__ = [
    # Core interfaces
    'NetworkComponent', 'NetworkAnalyzer', 'GrowthStrategy',
    'LearningRateStrategy', 'NetworkTrainer', 'NetworkEvolutionSystem',
    'NetworkContext', 'AnalysisResult', 'GrowthAction', 'ActionType',
    
    # Analyzers
    'StandardExtremaAnalyzer', 'NetworkStatsAnalyzer', 'SimpleInformationFlowAnalyzer',
    
    # Strategies
    'ExtremaGrowthStrategy', 'InformationFlowGrowthStrategy', 
    'ResidualBlockGrowthStrategy', 'HybridGrowthStrategy',
    
    # Evolution systems
    'StandardNetworkTrainer', 'ComposableEvolutionSystem',
    'create_standard_evolution_system', 'create_extrema_focused_system', 
    'create_hybrid_system'
]

# Version info
__version__ = "1.0.0"
