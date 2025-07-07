"""
Autocorrelation Framework for Meta-Learning Growth

This package implements the revolutionary autocorrelation framework that discovers
which metrics actually predict neural network learning success through statistical
analysis and correlation discovery.

Components:
- PerformanceAnalyzer: Core metric-performance correlation analysis
- CorrelationAnalyzer: Statistical correlation and pattern discovery
- PatternDiscovery: Advanced pattern detection and interaction analysis
- StrategyLearner: Strategy effectiveness tracking and recommendations
- TemporalAnalyzer: Time-series analysis and trend detection
- ThresholdDetector: Critical threshold identification
- VisualizationEngine: Insights dashboard and plotting tools
"""

# Import core components (will be available after we create them)
try:
    from .performance_analyzer import PerformanceAnalyzer
except ImportError:
    pass

try:
    from .correlation_analyzer import CorrelationAnalyzer
except ImportError:
    pass

try:
    from .pattern_discovery import PatternDiscovery
except ImportError:
    pass

try:
    from .strategy_learner import StrategyLearner
except ImportError:
    pass

try:
    from .temporal_analyzer import TemporalAnalyzer
except ImportError:
    pass

try:
    from .threshold_detector import ThresholdDetector
except ImportError:
    pass

try:
    from .visualization import VisualizationEngine
except ImportError:
    pass

__all__ = [
    'PerformanceAnalyzer',
    'CorrelationAnalyzer',
    'PatternDiscovery',
    'StrategyLearner',
    'TemporalAnalyzer',
    'ThresholdDetector',
    'VisualizationEngine'
]
