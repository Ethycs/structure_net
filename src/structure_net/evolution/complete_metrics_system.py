#!/usr/bin/env python3
"""
Complete Metrics System - Modular Network Analysis Framework

This module provides backward compatibility for the complete metrics system
while delegating to the new modular architecture for improved maintainability.

The system has been refactored into specialized components:
- Mutual Information Analysis (metrics.mutual_information)
- Activity Pattern Analysis (metrics.activity_analysis)  
- Sensitivity Analysis (metrics.sensitivity_analysis)
- Graph-theoretic Analysis (metrics.graph_analysis)
- Autocorrelation Framework (autocorrelation.*)

For new code, prefer importing from the specific modules:
    from ..evolution.metrics import CompleteMetricsSystem
    from ..evolution.autocorrelation import PerformanceAnalyzer
"""

import logging
import warnings
import traceback
from typing import Dict, Any

# Import the new modular components
from .metrics.base import ThresholdConfig, MetricsConfig
from .metrics.integrated_system import CompleteMetricsSystem as NewCompleteMetricsSystem
from .metrics.mutual_information import MutualInformationAnalyzer
from .metrics.activity_analysis import ActivityAnalyzer
from .metrics.sensitivity_analysis import SensitivityAnalyzer
from .metrics.graph_analysis import GraphAnalyzer

# Import autocorrelation framework
try:
    from .autocorrelation.performance_analyzer import PerformanceAnalyzer as MetricPerformanceAnalyzer
except ImportError:
    # Fallback if autocorrelation framework is not available
    MetricPerformanceAnalyzer = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# ============================================================================

class CompleteMIAnalyzer:
    """Backward compatibility wrapper for MutualInformationAnalyzer."""
    
    def __init__(self, threshold=0.01):
        from .metrics.base import ThresholdConfig
        threshold_config = ThresholdConfig(activation_threshold=threshold)
        self._analyzer = MutualInformationAnalyzer(threshold_config)
        
        warnings.warn(
            "CompleteMIAnalyzer is deprecated. Use MutualInformationAnalyzer from "
            "structure_net.evolution.metrics.mutual_information instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
    def compute_complete_mi_metrics(self, X, Y):
        """Backward compatibility method."""
        return self._analyzer.compute_metrics(X, Y)


class CompleteActivityAnalyzer:
    """Backward compatibility wrapper for ActivityAnalyzer."""
    
    def __init__(self, threshold_config):
        self._analyzer = ActivityAnalyzer(threshold_config)
        
        warnings.warn(
            "CompleteActivityAnalyzer is deprecated. Use ActivityAnalyzer from "
            "structure_net.evolution.metrics.activity_analysis instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
    def compute_complete_activity_metrics(self, activations, layer_idx):
        """Backward compatibility method."""
        return self._analyzer.compute_metrics(activations, layer_idx)


class CompleteSensLIAnalyzer:
    """Backward compatibility wrapper for SensitivityAnalyzer."""
    
    def __init__(self, network, threshold_config):
        self._analyzer = SensitivityAnalyzer(network, threshold_config)
        
        warnings.warn(
            "CompleteSensLIAnalyzer is deprecated. Use SensitivityAnalyzer from "
            "structure_net.evolution.metrics.sensitivity_analysis instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
    def compute_complete_sensli_metrics(self, layer_i, layer_j, data_loader, num_batches=10):
        """Backward compatibility method."""
        return self._analyzer.compute_metrics(layer_i, layer_j, data_loader, num_batches)
    
    def compute_sensli_from_precomputed_data(self, acts_i, acts_j, grads_i, grads_j, layer_i, layer_j):
        """Backward compatibility method."""
        return self._analyzer.compute_metrics_from_precomputed_data(
            acts_i, acts_j, grads_i, grads_j, layer_i, layer_j
        )


class CompleteGraphAnalyzer:
    """Backward compatibility wrapper for GraphAnalyzer."""
    
    def __init__(self, network, threshold_config):
        self._analyzer = GraphAnalyzer(network, threshold_config)
        
        warnings.warn(
            "CompleteGraphAnalyzer is deprecated. Use GraphAnalyzer from "
            "structure_net.evolution.metrics.graph_analysis instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
    def compute_complete_graph_metrics(self, activation_data):
        """Backward compatibility method."""
        return self._analyzer.compute_metrics(activation_data)


class CompleteMetricsSystem:
    """Backward compatibility wrapper for the new integrated metrics system."""
    
    def __init__(self, network, threshold_config, metrics_config):
        self._system = NewCompleteMetricsSystem(network, threshold_config, metrics_config)
        
        # Also initialize old-style analyzers for backward compatibility
        self.mi_analyzer = CompleteMIAnalyzer(threshold_config.activation_threshold)
        self.activity_analyzer = CompleteActivityAnalyzer(threshold_config)
        self.sensli_analyzer = CompleteSensLIAnalyzer(network, threshold_config)
        self.graph_analyzer = CompleteGraphAnalyzer(network, threshold_config)
        
        logger.info("🔄 Using modular metrics system with backward compatibility")
        
    def compute_all_metrics(self, data_loader, num_batches: int = 10):
        """Compute all metrics using the new modular system."""
        return self._system.compute_all_metrics(data_loader, num_batches)
    
    # Delegate all other methods to the new system
    def __getattr__(self, name):
        return getattr(self._system, name)


# ============================================================================
# EXPORTS AND COMPATIBILITY
# ============================================================================

# Export all classes for backward compatibility
__all__ = [
    # New modular components (preferred)
    'ThresholdConfig',
    'MetricsConfig',
    'MutualInformationAnalyzer',
    'ActivityAnalyzer', 
    'SensitivityAnalyzer',
    'GraphAnalyzer',
    'MetricPerformanceAnalyzer',
    
    # Backward compatibility wrappers (deprecated)
    'CompleteMIAnalyzer',
    'CompleteActivityAnalyzer', 
    'CompleteSensLIAnalyzer',
    'CompleteGraphAnalyzer',
    'CompleteMetricsSystem'
]

# Show deprecation warning when importing old classes
def _show_migration_info():
    """Show information about migrating to the new modular system."""
    # Get the calling file information
    stack = traceback.extract_stack()
    calling_file = None
    for frame in reversed(stack):
        if frame.filename != __file__ and not frame.filename.endswith('importlib/_bootstrap.py'):
            calling_file = frame.filename
            break
    
    caller_info = f"\n📍 Called from: {calling_file}" if calling_file else ""
    
    logger.info(f"""
🔄 METRICS SYSTEM MIGRATION NOTICE:{caller_info}

The complete metrics system has been refactored into modular components for better maintainability.

NEW RECOMMENDED IMPORTS:
  from ..evolution.metrics import CompleteMetricsSystem
  from ..evolution.metrics import MutualInformationAnalyzer, ActivityAnalyzer
  from ..evolution.autocorrelation import PerformanceAnalyzer

OLD IMPORTS (still work but deprecated):
  from ..evolution.complete_metrics_system import CompleteMetricsSystem

The new modular system provides:
✅ Better performance through optimized data collection
✅ Enhanced autocorrelation framework for meta-learning
✅ Improved caching and computation statistics
✅ Cleaner separation of concerns
✅ Better testing and maintainability

Your existing code will continue to work without changes.
""")

# Show migration info when module is imported
_show_migration_info()
