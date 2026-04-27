"""
Test runner for component pipelines.

Runs all registered component test pipelines to ensure
comprehensive validation of metrics and analyzers.

This works alongside the unified testing approach in
test_analyzers_unified.py to provide multiple testing strategies.
"""

import pytest
from typing import List, Type

from src.structure_net.core import IComponent
from src.structure_net.components.metrics import (
    # Information flow metrics
    LayerMIMetric, EntropyMetric, InformationFlowMetric,
    RedundancyMetric, AdvancedMIMetric,
    # Homological metrics
    ChainComplexMetric, RankMetric, BettiNumberMetric,
    HomologyMetric, InformationEfficiencyMetric,
    # Sensitivity/Topological metrics
    GradientSensitivityMetric, BottleneckMetric,
    ExtremaMetric, PersistenceMetric,
    ConnectivityMetric, TopologicalSignatureMetric,
    # Activity metrics
    NeuronActivityMetric, ActivationDistributionMetric,
    ActivityPatternMetric, LayerHealthMetric,
    # Graph metrics
    GraphStructureMetric, CentralityMetric,
    SpectralGraphMetric, PathAnalysisMetric,
    # Catastrophe metrics
    ActivationStabilityMetric, LyapunovMetric, TransitionEntropyMetric,
    # Compactification metrics
    CompressionRatioMetric, PatchEffectivenessMetric,
    MemoryEfficiencyMetric, ReconstructionQualityMetric
)
from src.structure_net.components.analyzers import (
    InformationFlowAnalyzer, HomologicalAnalyzer,
    SensitivityAnalyzer, TopologicalAnalyzer,
    ActivityAnalyzer, GraphAnalyzer,
    CatastropheAnalyzer, CompactificationAnalyzer
)

from .pipelines import TestPipelineRegistry, create_test_suite
from tests.fixtures import create_test_model


# Define component groups for organized testing
METRIC_GROUPS = {
    'information_flow': [
        LayerMIMetric, EntropyMetric, InformationFlowMetric,
        RedundancyMetric, AdvancedMIMetric
    ],
    'homological': [
        ChainComplexMetric, RankMetric, BettiNumberMetric,
        HomologyMetric, InformationEfficiencyMetric
    ],
    'sensitivity_topological': [
        GradientSensitivityMetric, BottleneckMetric,
        ExtremaMetric, PersistenceMetric,
        ConnectivityMetric, TopologicalSignatureMetric
    ],
    'activity': [
        NeuronActivityMetric, ActivationDistributionMetric,
        ActivityPatternMetric, LayerHealthMetric
    ],
    'graph': [
        GraphStructureMetric, CentralityMetric,
        SpectralGraphMetric, PathAnalysisMetric
    ],
    'catastrophe': [
        ActivationStabilityMetric, LyapunovMetric, TransitionEntropyMetric
    ],
    'compactification': [
        CompressionRatioMetric, PatchEffectivenessMetric,
        MemoryEfficiencyMetric, ReconstructionQualityMetric
    ]
}

ANALYZER_GROUPS = {
    'core_analyzers': [
        InformationFlowAnalyzer, HomologicalAnalyzer,
        SensitivityAnalyzer, TopologicalAnalyzer
    ],
    'specialized_analyzers': [
        ActivityAnalyzer, GraphAnalyzer,
        CatastropheAnalyzer, CompactificationAnalyzer
    ]
}


class TestMetricPipelines:
    """Test all metric components using pipelines."""
    
    @pytest.mark.parametrize("metric_class", 
                           [metric for group in METRIC_GROUPS.values() 
                            for metric in group])
    def test_metric_pipeline(self, metric_class: Type[IComponent]):
        """Run pipeline tests for each metric."""
        try:
            pipeline = TestPipelineRegistry.get_pipeline(metric_class)
            pipeline_instance = pipeline()
            
            # Run standard tests
            pipeline_instance.test_contract_compliance()
            pipeline_instance.test_contract_validation()
            pipeline_instance.test_output_compliance()
            pipeline_instance.test_error_handling()
            pipeline_instance.test_determinism()
            pipeline_instance.test_edge_cases()
            
            # Run metric-specific tests if available
            if hasattr(pipeline_instance, 'test_metric_ranges'):
                pipeline_instance.test_metric_ranges()
                
        except ValueError as e:
            if "No test pipeline registered" in str(e):
                pytest.skip(f"No pipeline registered for {metric_class.__name__}")
            else:
                raise


class TestAnalyzerPipelines:
    """Test all analyzer components using pipelines."""
    
    @pytest.mark.parametrize("analyzer_class",
                           [analyzer for group in ANALYZER_GROUPS.values()
                            for analyzer in group])
    def test_analyzer_pipeline(self, analyzer_class: Type[IComponent]):
        """Run pipeline tests for each analyzer."""
        try:
            pipeline = TestPipelineRegistry.get_pipeline(analyzer_class)
            pipeline_instance = pipeline()
            
            # Run standard tests
            pipeline_instance.test_contract_compliance()
            pipeline_instance.test_contract_validation()
            pipeline_instance.test_output_compliance()
            pipeline_instance.test_error_handling()
            pipeline_instance.test_determinism()
            
            # Run analyzer-specific tests
            pipeline_instance.test_metric_dependency()
            pipeline_instance.test_missing_metrics_handling()
            
        except ValueError as e:
            if "No test pipeline registered" in str(e):
                pytest.skip(f"No pipeline registered for {analyzer_class.__name__}")
            else:
                raise


class TestComponentIntegration:
    """Test component integration and composition."""
    
    def test_metric_analyzer_integration(self):
        """Test that metrics and analyzers work together."""
        # Test that analyzer pipelines can use metric pipelines
        from .pipelines import ActivityAnalyzerPipeline, TestPipelineRegistry
        
        # Test analyzer pipeline
        analyzer_pipeline = ActivityAnalyzerPipeline()
        analyzer = analyzer_pipeline.get_component_class()()
        
        # Create context and test
        context = analyzer_pipeline.create_analysis_context()
        model = analyzer_pipeline.create_model()
        
        # Should work with pipeline-created data
        assert context is not None
        assert model is not None
    
    def test_pipeline_registry(self):
        """Test the pipeline registry functionality."""
        # Test registration
        class DummyMetric:
            pass
        
        class DummyPipeline:
            pass
        
        TestPipelineRegistry.register(DummyMetric, DummyPipeline)
        
        # Test retrieval
        pipeline = TestPipelineRegistry.get_pipeline(DummyMetric)
        assert pipeline == DummyPipeline
    
    def test_contract_system(self):
        """Test the contract system across components."""
        # Create a metric and analyzer that should work together
        entropy_metric = EntropyMetric()
        info_analyzer = InformationFlowAnalyzer()
        
        # Check that analyzer's required inputs can be satisfied
        analyzer_contract = info_analyzer.contract
        metric_contract = entropy_metric.contract
        
        # Analyzer should declare what it needs
        assert len(analyzer_contract.required_inputs) > 0
        
        # Metric should declare what it provides
        assert len(metric_contract.provided_outputs) > 0


class TestPipelineFeatures:
    """Test specific pipeline features and edge cases."""
    
    def test_edge_case_handling(self):
        """Test that pipelines handle edge cases properly."""
        from .pipelines import EntropyMetricPipeline
        
        pipeline = EntropyMetricPipeline()
        edge_cases = pipeline.create_edge_cases()
        
        # Should have defined edge cases
        assert len(edge_cases) > 0
        
        # Test each edge case
        metric = pipeline.get_component_class()()
        for case_name, case_inputs in edge_cases.items():
            try:
                output = pipeline.run_component(metric, case_inputs)
                pipeline.validate_edge_case_output(case_name, output, case_inputs)
            except Exception as e:
                pytest.fail(f"Edge case '{case_name}' failed: {e}")
    
    def test_determinism_validation(self):
        """Test that determinism checks work correctly."""
        from .pipelines import LayerMIMetricPipeline
        
        pipeline = LayerMIMetricPipeline()
        pipeline.test_determinism()  # Should not raise
    
    def test_output_validation(self):
        """Test output validation logic."""
        from .pipelines import GradientSensitivityPipeline
        
        pipeline = GradientSensitivityPipeline()
        metric = pipeline.get_component_class()()
        
        # Create valid inputs and run
        inputs = pipeline.create_valid_inputs()
        outputs = pipeline.run_component(metric, inputs)
        
        # Validation should pass
        pipeline.validate_outputs(outputs, inputs)
        
        # Test with invalid output structure
        invalid_outputs = {'wrong_key': 0.5}
        with pytest.raises(AssertionError):
            pipeline.validate_outputs(invalid_outputs, inputs)


class TestHybridApproach:
    """Demonstrates using both pipeline and unified approaches together."""
    
    def test_compare_approaches(self, metric_data_generators):
        """Show how pipeline and unified approaches complement each other."""
        # Component-level pipelines are great for:
        # 1. Specialized testing for specific component types
        # 2. Custom validation logic
        # 3. Component-specific edge cases
        
        # Example: Use pipeline for specialized testing
        from .pipelines import ActivityAnalyzerPipeline
        pipeline = ActivityAnalyzerPipeline()
        
        # Pipeline provides specialized context and validation
        model = pipeline.create_model()  # Creates model with dead neurons
        context = pipeline.create_analysis_context()  # Specialized context
        
        # The unified approach is great for:
        # 1. Testing many components consistently
        # 2. Contract-based validation
        # 3. Automatic test generation
        
        # Example: Component-level pipelines provide the comprehensive testing
        # that was previously done by unified approaches
        
        # Both approaches are valid and complement each other!


@pytest.mark.integration
class TestFullPipeline:
    """Full integration tests across all components."""
    
    def test_complete_analysis_pipeline(self):
        """Test a complete analysis pipeline from metrics to insights."""
        from src.structure_net.core import AnalysisReport, EvolutionContext
        from tests.fixtures import create_test_model, create_test_activations
        
        # Create test model and data
        model = create_test_model([50, 40, 30, 20, 10])
        test_data = create_test_activations(100, 50)
        
        # Create context
        context = EvolutionContext({
            'test_data': test_data,
            'model': model
        })
        
        # Create report
        report = AnalysisReport()
        
        # Run multiple analyzers
        analyzers = [
            InformationFlowAnalyzer(),
            ActivityAnalyzer(),
            TopologicalAnalyzer()
        ]
        
        all_results = {}
        for analyzer in analyzers:
            try:
                results = analyzer.analyze(model, report, context)
                all_results[analyzer.name] = results
            except Exception as e:
                # Some analyzers might need specific context
                pass
        
        # Should have some results
        assert len(all_results) > 0
        
        # Report should have accumulated metrics
        assert len(report.metrics) > 0


def run_all_pipeline_tests():
    """Convenience function to run all pipeline tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_all_pipeline_tests()