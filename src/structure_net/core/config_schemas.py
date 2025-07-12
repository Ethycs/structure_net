"""
Configuration schemas for Structure Net components.
"""
from dataclasses import dataclass, field

@dataclass
class ThresholdConfig:
    """Configuration for thresholds used in growth and analysis."""
    type: str = "config"
    activation_threshold: float = 0.01
    weight_threshold: float = 0.01
    gradient_threshold: float = 1e-4
    persistence_ratio: float = 0.8
    adaptive: bool = True
    min_active_ratio: float = 0.05
    max_active_ratio: float = 0.5

@dataclass
class MetricsConfig:
    """Configuration for which metrics to compute."""
    type: str = "config"
    compute_mi: bool = True
    compute_activity: bool = True
    compute_sensli: bool = True
    compute_graph: bool = True
    compute_betweenness: bool = False
    compute_spectral: bool = False
