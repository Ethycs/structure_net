"""
Component test pipelines.

Provides standardized testing pipelines for metrics and analyzers
that leverage the component architecture for thorough validation.
"""

from .base_test_pipeline import (
    ComponentTestPipeline,
    MetricTestPipeline,
    AnalyzerTestPipeline,
    TestPipelineRegistry,
    create_test_suite
)

from .metric_pipelines import (
    ActivationBasedMetricPipeline,
    GradientBasedMetricPipeline,
    LayerAnalysisMetricPipeline,
    ModelAnalysisMetricPipeline,
    EntropyMetricPipeline,
    LayerMIMetricPipeline,
    GradientSensitivityPipeline,
    CompactificationMetricPipeline,
    DynamicsMetricPipeline,
    CompressionRatioMetricPipeline
)

from .analyzer_pipelines import (
    InformationFlowAnalyzerPipeline,
    HomologicalAnalyzerPipeline,
    ActivityAnalyzerPipeline,
    GraphAnalyzerPipeline,
    CatastropheAnalyzerPipeline,
    CompactificationAnalyzerPipeline,
    AnalyzerIntegrationTests
)

__all__ = [
    # Base classes
    'ComponentTestPipeline',
    'MetricTestPipeline',
    'AnalyzerTestPipeline',
    'TestPipelineRegistry',
    'create_test_suite',
    
    # Metric pipelines
    'ActivationBasedMetricPipeline',
    'GradientBasedMetricPipeline',
    'LayerAnalysisMetricPipeline',
    'ModelAnalysisMetricPipeline',
    'EntropyMetricPipeline',
    'LayerMIMetricPipeline',
    'GradientSensitivityPipeline',
    'CompactificationMetricPipeline',
    'DynamicsMetricPipeline',
    'CompressionRatioMetricPipeline',
    
    # Analyzer pipelines
    'InformationFlowAnalyzerPipeline',
    'HomologicalAnalyzerPipeline',
    'ActivityAnalyzerPipeline',
    'GraphAnalyzerPipeline',
    'CatastropheAnalyzerPipeline',
    'CompactificationAnalyzerPipeline',
    'AnalyzerIntegrationTests'
]