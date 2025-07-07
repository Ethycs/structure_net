"""
Metrics package for comprehensive network analysis.

This package provides modular components for analyzing neural networks:
- Mutual Information Analysis
- Activity Pattern Analysis  
- Sensitivity Analysis (SensLI)
- Graph-theoretic Analysis
- Integrated Metrics System
"""

from .base import (
    ThresholdConfig,
    MetricsConfig,
    MetricResult,
    BaseMetricAnalyzer,
    NetworkAnalyzerMixin,
    StatisticalUtilsMixin
)

# Import analyzers (will be available after we create them)
try:
    from .mutual_information import MutualInformationAnalyzer
except ImportError:
    pass

try:
    from .activity_analysis import ActivityAnalyzer
except ImportError:
    pass

try:
    from .sensitivity_analysis import SensitivityAnalyzer
except ImportError:
    pass

try:
    from .graph_analysis import GraphAnalyzer
except ImportError:
    pass

try:
    from .integrated_system import CompleteMetricsSystem
except ImportError:
    pass

__all__ = [
    # Base classes
    'ThresholdConfig',
    'MetricsConfig',
    'MetricResult',
    'BaseMetricAnalyzer',
    'NetworkAnalyzerMixin',
    'StatisticalUtilsMixin',
    
    # Analyzers (when available)
    'MutualInformationAnalyzer',
    'ActivityAnalyzer',
    'SensitivityAnalyzer', 
    'GraphAnalyzer',
    'CompleteMetricsSystem'
]
