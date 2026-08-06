"""
Analyzer components for the Structure Net framework.

Analyzers are high-level components that combine multiple metrics
to provide comprehensive insights about neural network behavior.
"""

from .information_flow_analyzer import InformationFlowAnalyzer
from .homological_analyzer import HomologicalAnalyzer
from .sensitivity_analyzer import SensitivityAnalyzer
from .topological_analyzer import TopologicalAnalyzer
from .activity_analyzer import ActivityAnalyzer
from .graph_analyzer import GraphAnalyzer
from .catastrophe_analyzer import CatastropheAnalyzer
from .compactification_analyzer import CompactificationAnalyzer
from .extrema_analyzer import ExtremaAnalyzer
from .performance_correlation_analyzer import PerformanceCorrelationAnalyzer
from .semantic_quotient_analyzer import (
    PersistenceSummary,
    bootstrap_max_h1,
    circular_phase_alignment,
    circular_winding_degree,
    complex_defect_charge,
    fisher_rao_distance_matrix,
    knn_geodesic_distance_matrix,
    nuisance_collapse_ratio,
    paired_geometry_alignment,
    persistence_diagrams,
    persistent_cohomology_circle_coordinate,
    representation_distance_matrix,
    summarize_persistence,
)

__all__ = [
    'InformationFlowAnalyzer',
    'HomologicalAnalyzer',
    'SensitivityAnalyzer',
    'TopologicalAnalyzer',
    'ActivityAnalyzer',
    'GraphAnalyzer',
    'CatastropheAnalyzer',
    'CompactificationAnalyzer',
    'ExtremaAnalyzer',
    'PerformanceCorrelationAnalyzer',
    'PersistenceSummary',
    'bootstrap_max_h1',
    'circular_phase_alignment',
    'circular_winding_degree',
    'complex_defect_charge',
    'fisher_rao_distance_matrix',
    'knn_geodesic_distance_matrix',
    'nuisance_collapse_ratio',
    'paired_geometry_alignment',
    'persistence_diagrams',
    'persistent_cohomology_circle_coordinate',
    'representation_distance_matrix',
    'summarize_persistence',
]
