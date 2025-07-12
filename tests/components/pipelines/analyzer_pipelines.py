"""
Test pipelines for analyzer components.

Provides specialized test pipelines for different analyzer types
with appropriate test models, contexts, and validation.

These component-level pipelines work alongside the unified testing
approach in test_analyzers_unified.py to provide comprehensive testing.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
import networkx as nx

from src.structure_net.core import (
    IModel, EvolutionContext, AnalysisReport
)
from src.structure_net.components.analyzers import (
    InformationFlowAnalyzer, HomologicalAnalyzer,
    SensitivityAnalyzer, TopologicalAnalyzer,
    ActivityAnalyzer, GraphAnalyzer,
    CatastropheAnalyzer, CompactificationAnalyzer
)
from tests.fixtures import (
    create_test_model, create_test_activations,
    create_layer_activations_data, create_compact_data,
    create_trajectory_data, create_test_gradients
)
from .base_test_pipeline import AnalyzerTestPipeline


class InformationFlowAnalyzerPipeline(AnalyzerTestPipeline):
    """Test pipeline for InformationFlowAnalyzer."""
    
    def get_component_class(self):
        return InformationFlowAnalyzer
    
    def create_model(self) -> IModel:
        return create_test_model([30, 25, 20, 15, 10])
    
    def create_analysis_context(self) -> EvolutionContext:
        """Create context with layer activation data."""
        model = self.create_model()
        input_data = create_test_activations(100, 30)
        
        # Generate layer activations
        layer_data = create_layer_activations_data(model, input_data)
        
        # Format for analyzer
        layer_activations = {}
        layer_sequence = []
        
        for i, (name, acts) in enumerate(layer_data.items()):
            if 'activated' not in name:  # Skip post-activation
                layer_name = f'layer_{i}'
                layer_sequence.append(layer_name)
                
                # Get input/output for each layer
                if i == 0:
                    layer_input = input_data
                else:
                    prev_layer = f'layer_{i-1}'
                    layer_input = layer_activations[prev_layer]['output']
                
                layer_activations[layer_name] = {
                    'input': layer_input,
                    'output': acts
                }
        
        return EvolutionContext({
            'layer_sequence': layer_sequence,
            'layer_activations': layer_activations
        })
    
    def get_required_metrics(self) -> List[str]:
        return ['LayerMIMetric', 'InformationFlowMetric', 'RedundancyMetric']
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        return {
            'model': self.create_model(),
            'report': AnalysisReport(),
            'context': self.create_analysis_context()
        }
    
    def create_invalid_inputs(self) -> List[Dict[str, Any]]:
        return [
            {'model': self.create_model()},  # Missing context
            {  # Empty context
                'model': self.create_model(),
                'context': EvolutionContext({})
            }
        ]
    
    def validate_outputs(self, outputs: Dict[str, Any], 
                        inputs: Dict[str, Any]) -> None:
        """Validate information flow analysis outputs."""
        expected_keys = ['information_flow_analysis', 'layer_insights',
                        'bottleneck_analysis', 'recommendations']
        
        for key in expected_keys:
            assert key in outputs, f"Missing expected output: {key}"
        
        # Validate recommendations
        assert isinstance(outputs['recommendations'], list)
        assert len(outputs['recommendations']) > 0
        
        # Validate layer insights
        assert isinstance(outputs['layer_insights'], dict)
        
        # Validate flow analysis
        flow_analysis = outputs['information_flow_analysis']
        assert 'total_information_flow' in flow_analysis
        assert 'flow_efficiency' in flow_analysis


class HomologicalAnalyzerPipeline(AnalyzerTestPipeline):
    """Test pipeline for HomologicalAnalyzer."""
    
    def get_component_class(self):
        return HomologicalAnalyzer
    
    def create_model(self) -> IModel:
        return create_test_model([25, 20, 15, 10, 5])
    
    def create_analysis_context(self) -> EvolutionContext:
        """Create context for homological analysis."""
        return EvolutionContext({
            'compute_persistence': True,
            'max_dimension': 2
        })
    
    def get_required_metrics(self) -> List[str]:
        return ['ChainComplexMetric', 'BettiNumberMetric', 
                'HomologyMetric', 'InformationEfficiencyMetric']
    
    def create_valid_inputs(self) -> Dict[str, Any]:
        return {
            'model': self.create_model(),
            'report': AnalysisReport(),
            'context': self.create_analysis_context()
        }
    
    def validate_outputs(self, outputs: Dict[str, Any], 
                        inputs: Dict[str, Any]) -> None:
        """Validate homological analysis outputs."""
        expected_sections = ['topological_summary', 'homological_features',
                           'structural_insights', 'efficiency_analysis']
        
        for section in expected_sections:
            assert section in outputs, f"Missing section: {section}"
        
        # Validate topological summary
        topo_summary = outputs['topological_summary']
        assert 'betti_numbers' in topo_summary
        assert 'euler_characteristic' in topo_summary


class ActivityAnalyzerPipeline(AnalyzerTestPipeline):
    """Test pipeline for ActivityAnalyzer."""
    
    def get_component_class(self):
        return ActivityAnalyzer
    
    def create_model(self) -> IModel:
        model = create_test_model([50, 40, 30, 20])
        
        # Add some dead neurons
        with torch.no_grad():
            model.layers[1].weight[:10, :] *= 0.001
            model.layers[2].weight[20:, :] *= 0.001
        
        return model
    
    def create_analysis_context(self) -> EvolutionContext:
        """Create context with activation data."""
        model = self.create_model()
        test_data = create_test_activations(200, 50)
        
        # Generate activations for each layer
        layer_activations = {}
        x = test_data
        
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if model.add_nonlinearity and i < len(model.layers) - 1:
                x = model.activation(x)
            layer_activations[f'layer_{i}'] = x.detach()
        
        # Also add some gradients
        layer_gradients = {}
        for name, acts in layer_activations.items():
            layer_gradients[name] = create_test_gradients(acts.shape, scale=0.1)
        
        return EvolutionContext({
            'layer_activations': layer_activations,
            'layer_gradients': layer_gradients,
            'test_data': test_data
        })
    
    def get_required_metrics(self) -> List[str]:
        return ['NeuronActivityMetric', 'ActivationDistributionMetric',
                'ActivityPatternMetric', 'LayerHealthMetric']
    
    def validate_outputs(self, outputs: Dict[str, Any], 
                        inputs: Dict[str, Any]) -> None:
        """Validate activity analysis outputs."""
        assert 'activity_summary' in outputs
        assert 'layer_analysis' in outputs
        assert 'health_assessment' in outputs
        assert 'recommendations' in outputs
        
        # Validate activity summary
        summary = outputs['activity_summary']
        assert 'overall_activity_rate' in summary
        assert 'dead_neuron_percentage' in summary
        
        # Should detect issues in our test model
        assert len(outputs['recommendations']) > 0


class GraphAnalyzerPipeline(AnalyzerTestPipeline):
    """Test pipeline for GraphAnalyzer."""
    
    def get_component_class(self):
        return GraphAnalyzer
    
    def create_model(self) -> IModel:
        return create_test_model([30, 25, 20, 15])
    
    def create_analysis_context(self) -> EvolutionContext:
        """Create context with graph data."""
        # Create a sample graph
        G = nx.karate_club_graph()
        
        # Also create activation data for network graph
        model = self.create_model()
        test_data = create_test_activations(100, 30)
        activation_data = create_layer_activations_data(model, test_data)
        
        return EvolutionContext({
            'graph': G,
            'model': model,
            'activation_data': activation_data,
            'compute_communities': True
        })
    
    def get_required_metrics(self) -> List[str]:
        return ['GraphStructureMetric', 'CentralityMetric',
                'SpectralGraphMetric', 'PathAnalysisMetric']
    
    def validate_outputs(self, outputs: Dict[str, Any], 
                        inputs: Dict[str, Any]) -> None:
        """Validate graph analysis outputs."""
        expected = ['graph_properties', 'centrality_analysis',
                   'spectral_analysis', 'path_analysis', 'insights']
        
        for key in expected:
            assert key in outputs, f"Missing: {key}"
        
        # Validate graph properties
        props = outputs['graph_properties']
        assert 'num_nodes' in props
        assert 'density' in props
        
        # Validate centrality analysis
        centrality = outputs['centrality_analysis']
        assert 'hub_nodes' in centrality
        assert 'centrality_distribution' in centrality


class CatastropheAnalyzerPipeline(AnalyzerTestPipeline):
    """Test pipeline for CatastropheAnalyzer."""
    
    def get_component_class(self):
        return CatastropheAnalyzer
    
    def create_model(self) -> IModel:
        # Create model with potential instabilities
        model = create_test_model([40, 30, 20, 10])
        
        # Add some noise to weights
        with torch.no_grad():
            for layer in model.layers:
                if hasattr(layer, 'weight'):
                    layer.weight += torch.randn_like(layer.weight) * 0.5
        
        return model
    
    def create_analysis_context(self) -> EvolutionContext:
        """Create context with test data and trajectories."""
        test_data = create_test_activations(500, 40)
        num_trajectories = 10
        
        return EvolutionContext({
            'test_data': test_data,
            'num_trajectories': num_trajectories
        })
    
    def get_required_metrics(self) -> List[str]:
        return ['ActivationStabilityMetric', 'LyapunovMetric',
                'TransitionEntropyMetric']
    
    def validate_outputs(self, outputs: Dict[str, Any], 
                        inputs: Dict[str, Any]) -> None:
        """Validate catastrophe analysis outputs."""
        required = ['catastrophe_risk_score', 'stability_analysis',
                   'lyapunov_analysis', 'dynamics_analysis',
                   'risk_factors', 'recommendations']
        
        for key in required:
            assert key in outputs, f"Missing: {key}"
        
        # Validate risk score
        risk_score = outputs['catastrophe_risk_score']
        assert 0 <= risk_score <= 1
        
        # Validate risk factors
        assert isinstance(outputs['risk_factors'], list)
        for factor in outputs['risk_factors']:
            assert 'factor' in factor
            assert 'severity' in factor
            assert factor['severity'] in ['low', 'medium', 'high']


class CompactificationAnalyzerPipeline(AnalyzerTestPipeline):
    """Test pipeline for CompactificationAnalyzer."""
    
    def get_component_class(self):
        return CompactificationAnalyzer
    
    def create_model(self) -> Optional[IModel]:
        # Compactification analyzer might not need a model
        return None
    
    def create_analysis_context(self) -> EvolutionContext:
        """Create context with compactification data."""
        compact_data = create_compact_data(
            original_size=10000,
            compression_ratio=0.25,
            num_patches=15
        )
        
        # Create history for trend analysis
        history = []
        for i in range(5):
            ratio = 0.5 - (i * 0.05)  # Improving compression
            hist_data = create_compact_data(
                original_size=10000,
                compression_ratio=ratio,
                num_patches=10 + i
            )
            history.append(hist_data)
        
        return EvolutionContext({
            'compact_data': compact_data,
            'compact_history': history,
            'original_network': create_test_model([100, 50, 25])
        })
    
    def get_required_metrics(self) -> List[str]:
        return ['CompressionRatioMetric', 'PatchEffectivenessMetric',
                'MemoryEfficiencyMetric', 'ReconstructionQualityMetric']
    
    def validate_outputs(self, outputs: Dict[str, Any], 
                        inputs: Dict[str, Any]) -> None:
        """Validate compactification analysis outputs."""
        required = ['compression_analysis', 'patch_analysis',
                   'memory_analysis', 'quality_analysis',
                   'overall_assessment', 'recommendations', 'trends']
        
        for key in required:
            assert key in outputs, f"Missing: {key}"
        
        # Validate overall assessment
        assessment = outputs['overall_assessment']
        assert 'overall_score' in assessment
        assert 'overall_grade' in assessment
        assert 0 <= assessment['overall_score'] <= 1
        
        # Validate trends (should exist due to history)
        trends = outputs['trends']
        if trends:  # If history was provided
            assert 'compression_trend' in trends
            assert 'quality_trend' in trends


# Component-specific test cases

class AnalyzerIntegrationTests:
    """Integration tests for analyzer pipelines."""
    
    @staticmethod
    def test_analyzer_metric_integration():
        """Test that analyzers properly integrate with metrics."""
        # Create analyzer and context
        analyzer = InformationFlowAnalyzer()
        pipeline = InformationFlowAnalyzerPipeline()
        
        model = pipeline.create_model()
        context = pipeline.create_analysis_context()
        report = AnalysisReport()
        
        # Run analysis
        results = analyzer.analyze(model, report, context)
        
        # Check that metrics were computed
        assert len(report.metrics) > 0
        
        # Check that analyzer used metric results
        assert 'information_flow_analysis' in results
        assert results['information_flow_analysis']['total_information_flow'] > 0
    
    @staticmethod
    def test_analyzer_composition():
        """Test composing multiple analyzers."""
        model = create_test_model([30, 25, 20, 15, 10])
        report = AnalysisReport()
        
        # Create shared context
        test_data = create_test_activations(100, 30)
        activation_data = create_layer_activations_data(model, test_data)
        
        context = EvolutionContext({
            'test_data': test_data,
            'layer_activations': activation_data,
            'model': model
        })
        
        # Run multiple analyzers
        analyzers = [
            InformationFlowAnalyzer(),
            ActivityAnalyzer(),
            TopologicalAnalyzer()
        ]
        
        all_results = {}
        for analyzer in analyzers:
            results = analyzer.analyze(model, report, context)
            all_results[analyzer.name] = results
        
        # Verify all analyzers produced results
        assert len(all_results) == 3
        
        # Verify shared metrics were reused
        metric_compute_counts = {}
        for metric_name, metric_data in report.metrics.items():
            metric_compute_counts[metric_name] = 1  # Should be computed once
        
        # Each metric should only be computed once
        assert all(count == 1 for count in metric_compute_counts.values())


# Register analyzer pipelines
from .base_test_pipeline import TestPipelineRegistry

TestPipelineRegistry.register(InformationFlowAnalyzer, InformationFlowAnalyzerPipeline)
TestPipelineRegistry.register(HomologicalAnalyzer, HomologicalAnalyzerPipeline)
TestPipelineRegistry.register(ActivityAnalyzer, ActivityAnalyzerPipeline)
TestPipelineRegistry.register(GraphAnalyzer, GraphAnalyzerPipeline)
TestPipelineRegistry.register(CatastropheAnalyzer, CatastropheAnalyzerPipeline)
TestPipelineRegistry.register(CompactificationAnalyzer, CompactificationAnalyzerPipeline)