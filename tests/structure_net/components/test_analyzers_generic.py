"""
Generic component-based testing for all analyzers.

This provides comprehensive testing for IAnalyzer components using
the same contract-based approach as metrics.
"""

import pytest
import torch
from typing import Dict, Any, List, Type, Optional, Set
from dataclasses import dataclass

from src.structure_net.core import (
    IAnalyzer, IModel, AnalysisReport, EvolutionContext,
    ComponentContract
)
from src.structure_net.components.analyzers import (
    HomologicalAnalyzer, SensitivityAnalyzer, TopologicalAnalyzer,
    ActivityAnalyzer, GraphAnalyzer, CatastropheAnalyzer,
    CompactificationAnalyzer
)
from tests.fixtures import create_test_model, create_test_activations


# ===== Analyzer-Specific Test Data =====

@dataclass
class AnalyzerTestCase:
    """Test case specifically for analyzers."""
    name: str
    analyzer_class: Type[IAnalyzer]
    model: IModel
    report: AnalysisReport
    context_data: Dict[str, Any]
    should_succeed: bool = True
    expected_outputs: Optional[Set[str]] = None


class AnalyzerTestDataGenerator:
    """Generate test data specific to analyzer needs."""
    
    @staticmethod
    def create_base_report() -> AnalysisReport:
        """Create a basic analysis report with common metrics."""
        report = AnalysisReport()
        
        # Add basic metric data that analyzers might need
        report.add_metric_data('SparsityMetric', {
            'sparsity': 0.3,
            'density': 0.7,
            'zero_count': 300,
            'total_params': 1000
        })
        
        report.add_metric_data('EntropyMetric', {
            'entropy': 2.5,
            'normalized_entropy': 0.8,
            'effective_bits': 2.5
        })
        
        report.add_metric_data('LayerMIMetric', {
            'mutual_information': 1.2,
            'normalized_mi': 0.6,
            'layer_pairs': {'layer_0->layer_1': 1.2}
        })
        
        return report
    
    @staticmethod
    def create_homological_report() -> AnalysisReport:
        """Create report with homological metric data."""
        report = AnalyzerTestDataGenerator.create_base_report()
        
        report.add_metric_data('ChainComplexMetric', {
            'chain_length': 3,
            'boundary_ranks': [5, 4],
            'chain_connectivity': 0.8,
            'complex_dimension': 3
        })
        
        report.add_metric_data('BettiNumberMetric', {
            'betti_0': 1,
            'betti_1': 2,
            'betti_2': 0,
            'euler_characteristic': -1
        })
        
        report.add_metric_data('HomologyMetric', {
            'homology_groups': [1, 2, 0],
            'persistent_features': 3,
            'birth_death_pairs': [(0, float('inf')), (0.1, 0.5), (0.2, 0.8)]
        })
        
        return report
    
    @staticmethod
    def create_activity_report() -> AnalysisReport:
        """Create report with activity metric data."""
        report = AnalyzerTestDataGenerator.create_base_report()
        
        report.add_metric_data('NeuronActivityMetric', {
            'active_ratio': 0.85,
            'dead_neurons': 15,
            'activity_variance': 0.2
        })
        
        report.add_metric_data('ActivationDistributionMetric', {
            'mean_activation': 0.1,
            'std_activation': 1.2,
            'skewness': 0.3,
            'kurtosis': 2.8
        })
        
        return report
    
    @staticmethod
    def create_graph_report() -> AnalysisReport:
        """Create report with graph metric data."""
        report = AnalyzerTestDataGenerator.create_base_report()
        
        report.add_metric_data('GraphStructureMetric', {
            'num_nodes': 100,
            'num_edges': 450,
            'density': 0.045,
            'average_degree': 9.0
        })
        
        report.add_metric_data('CentralityMetric', {
            'mean_betweenness': 0.02,
            'max_betweenness': 0.15,
            'hub_nodes': [5, 12, 23]
        })
        
        return report
    
    @staticmethod
    def create_sensitivity_report() -> AnalysisReport:
        """Create report with sensitivity metric data."""
        report = AnalyzerTestDataGenerator.create_base_report()
        
        report.add_metric_data('GradientSensitivityMetric', {
            'mean_sensitivity': 0.5,
            'max_sensitivity': 2.3,
            'gradient_variance': 0.8
        })
        
        report.add_metric_data('BottleneckMetric', {
            'bottleneck_score': 0.7,
            'critical_layers': ['layer_2', 'layer_5']
        })
        
        return report
    
    @staticmethod
    def create_comprehensive_report() -> AnalysisReport:
        """Create a comprehensive report with all types of metrics."""
        report = AnalysisReport()
        
        # Merge all specialized reports
        for creator in [
            AnalyzerTestDataGenerator.create_base_report,
            AnalyzerTestDataGenerator.create_homological_report,
            AnalyzerTestDataGenerator.create_activity_report,
            AnalyzerTestDataGenerator.create_graph_report,
            AnalyzerTestDataGenerator.create_sensitivity_report
        ]:
            temp_report = creator()
            report.update(temp_report)
            report.sources.update(temp_report.sources)
        
        return report


# ===== Analyzer Test Fixtures =====

@pytest.fixture
def analyzer_test_model():
    """Provide a test model for analyzer testing."""
    return create_test_model([20, 15, 10, 5])


@pytest.fixture
def analyzer_reports():
    """Provide various analysis reports for testing."""
    return {
        'base': AnalyzerTestDataGenerator.create_base_report(),
        'homological': AnalyzerTestDataGenerator.create_homological_report(),
        'activity': AnalyzerTestDataGenerator.create_activity_report(),
        'graph': AnalyzerTestDataGenerator.create_graph_report(),
        'sensitivity': AnalyzerTestDataGenerator.create_sensitivity_report(),
        'comprehensive': AnalyzerTestDataGenerator.create_comprehensive_report()
    }


@pytest.fixture
def analyzer_context():
    """Provide a context with analyzer-specific data."""
    return EvolutionContext({
        'epoch': 10,
        'step': 1000,
        'learning_rate': 0.001,
        'loss': 0.5,
        'accuracy': 0.85,
        'layer_activations': {
            f'layer_{i}': create_test_activations()
            for i in range(4)
        },
        'gradients': {
            f'layer_{i}': torch.randn(10, 10)
            for i in range(4)
        },
        'optimization_history': {
            'losses': [1.0, 0.8, 0.6, 0.5],
            'accuracies': [0.5, 0.7, 0.8, 0.85]
        }
    })


# ===== Generic Analyzer Testing =====

class AnalyzerTester:
    """Generic tester for analyzer components."""
    
    def __init__(self, analyzer_class: Type[IAnalyzer]):
        self.analyzer_class = analyzer_class
        self.analyzer = analyzer_class()
        self.contract = self.analyzer.contract
    
    def determine_required_metrics(self) -> Set[str]:
        """Determine which metrics this analyzer needs."""
        if hasattr(self.analyzer, 'get_required_metrics'):
            return self.analyzer.get_required_metrics()
        
        # Infer from analyzer type
        analyzer_name = self.analyzer_class.__name__
        if 'Homological' in analyzer_name:
            return {'ChainComplexMetric', 'BettiNumberMetric', 'HomologyMetric'}
        elif 'Activity' in analyzer_name:
            return {'NeuronActivityMetric', 'ActivationDistributionMetric'}
        elif 'Graph' in analyzer_name:
            return {'GraphStructureMetric', 'CentralityMetric'}
        elif 'Sensitivity' in analyzer_name:
            return {'GradientSensitivityMetric', 'BottleneckMetric'}
        else:
            return {'SparsityMetric', 'EntropyMetric'}
    
    def create_appropriate_report(self) -> AnalysisReport:
        """Create an analysis report with required metrics."""
        analyzer_name = self.analyzer_class.__name__
        
        if 'Homological' in analyzer_name:
            return AnalyzerTestDataGenerator.create_homological_report()
        elif 'Activity' in analyzer_name:
            return AnalyzerTestDataGenerator.create_activity_report()
        elif 'Graph' in analyzer_name:
            return AnalyzerTestDataGenerator.create_graph_report()
        elif 'Sensitivity' in analyzer_name or 'Topological' in analyzer_name:
            return AnalyzerTestDataGenerator.create_sensitivity_report()
        else:
            return AnalyzerTestDataGenerator.create_comprehensive_report()
    
    def test_basic_functionality(self, model: IModel, context: EvolutionContext):
        """Test basic analyzer functionality."""
        report = self.create_appropriate_report()
        
        # Run analyzer
        result = self.analyzer.analyze(model, report, context)
        
        # Validate result
        assert isinstance(result, dict), f"Result must be dict, got {type(result)}"
        assert len(result) > 0, "Result should not be empty"
        
        # Check for expected sections
        expected_sections = ['summary', 'metrics', 'recommendations']
        for section in expected_sections:
            if section in ['summary', 'recommendations']:  # Common sections
                assert section in result, f"Missing expected section: {section}"
        
        return result
    
    def test_with_minimal_report(self, model: IModel, context: EvolutionContext):
        """Test analyzer with minimal report data."""
        minimal_report = AnalysisReport()
        minimal_report.add_metric_data('BasicMetric', {'value': 1.0})
        
        try:
            result = self.analyzer.analyze(model, minimal_report, context)
            return True, result
        except ValueError as e:
            # Expected if analyzer needs specific metrics
            return False, str(e)
    
    def test_output_structure(self, result: Dict[str, Any]):
        """Validate analyzer output structure."""
        issues = []
        
        # Check summary
        if 'summary' in result:
            summary = result['summary']
            if not isinstance(summary, (str, dict)):
                issues.append("Summary must be string or dict")
        
        # Check metrics
        if 'metrics' in result:
            metrics = result['metrics']
            if not isinstance(metrics, dict):
                issues.append("Metrics must be dict")
            else:
                for key, value in metrics.items():
                    if not isinstance(value, (int, float, str, list, dict)):
                        issues.append(f"Invalid metric type for {key}: {type(value)}")
        
        # Check recommendations
        if 'recommendations' in result:
            recs = result['recommendations']
            if not isinstance(recs, list):
                issues.append("Recommendations must be list")
            else:
                for rec in recs:
                    if not isinstance(rec, str):
                        issues.append("Each recommendation must be string")
        
        return issues


# ===== Pytest Test Classes =====

ALL_ANALYZERS = [
    HomologicalAnalyzer,
    SensitivityAnalyzer,
    TopologicalAnalyzer,
    ActivityAnalyzer,
    GraphAnalyzer,
    CatastropheAnalyzer,
    CompactificationAnalyzer
]


@pytest.mark.component
class TestAnalyzersGeneric:
    """Generic tests for all analyzers."""
    
    @pytest.mark.parametrize("analyzer_class", ALL_ANALYZERS,
                           ids=lambda c: c.__name__)
    def test_analyzer_contract(self, analyzer_class):
        """Test that analyzer has valid contract."""
        analyzer = analyzer_class()
        contract = analyzer.contract
        
        assert contract.component_name
        assert contract.version
        assert contract.maturity
        assert 'model' in contract.required_inputs
        assert 'report' in contract.required_inputs
    
    @pytest.mark.parametrize("analyzer_class", ALL_ANALYZERS,
                           ids=lambda c: c.__name__)
    def test_analyzer_basic_functionality(self, analyzer_class, 
                                        analyzer_test_model,
                                        analyzer_context):
        """Test basic functionality of each analyzer."""
        tester = AnalyzerTester(analyzer_class)
        result = tester.test_basic_functionality(analyzer_test_model, analyzer_context)
        
        # Validate output structure
        issues = tester.test_output_structure(result)
        assert len(issues) == 0, f"Output issues for {analyzer_class.__name__}: {issues}"
    
    @pytest.mark.parametrize("analyzer_class", ALL_ANALYZERS,
                           ids=lambda c: c.__name__)
    def test_analyzer_with_minimal_data(self, analyzer_class,
                                      analyzer_test_model,
                                      analyzer_context):
        """Test analyzers with minimal report data."""
        tester = AnalyzerTester(analyzer_class)
        success, result = tester.test_with_minimal_report(
            analyzer_test_model, analyzer_context
        )
        
        # Some analyzers should fail with minimal data
        required_metrics = tester.determine_required_metrics()
        if len(required_metrics) > 1:
            assert not success, f"{analyzer_class.__name__} should require specific metrics"
        else:
            # Others might work with minimal data
            pass


@pytest.mark.integration
class TestAnalyzerIntegration:
    """Integration tests for analyzers."""
    
    def test_analyzer_report_compatibility(self, analyzer_reports,
                                         analyzer_test_model,
                                         analyzer_context):
        """Test which analyzers work with which report types."""
        compatibility = {}
        
        for analyzer_class in ALL_ANALYZERS:
            analyzer = analyzer_class()
            compatibility[analyzer.name] = []
            
            for report_type, report in analyzer_reports.items():
                try:
                    result = analyzer.analyze(analyzer_test_model, report, analyzer_context)
                    compatibility[analyzer.name].append(report_type)
                except:
                    pass
        
        # Print compatibility matrix
        print("\nAnalyzer-Report Compatibility:")
        for analyzer_name, compatible_reports in compatibility.items():
            print(f"  {analyzer_name}: {', '.join(compatible_reports)}")
        
        # Verify expected compatibility
        assert 'homological' in compatibility.get('HomologicalAnalyzer', [])
        assert 'activity' in compatibility.get('ActivityAnalyzer', [])
    
    def test_analyzer_chaining(self, analyzer_test_model, analyzer_context):
        """Test running multiple analyzers on same report."""
        report = AnalyzerTestDataGenerator.create_comprehensive_report()
        
        results = {}
        for analyzer_class in [ActivityAnalyzer, GraphAnalyzer]:
            analyzer = analyzer_class()
            result = analyzer.analyze(analyzer_test_model, report, analyzer_context)
            results[analyzer.name] = result
            
            # Each analyzer can add to the report
            report.add_analyzer_data(analyzer.name, result)
        
        # Report should accumulate data
        assert len(report.sources) >= 2
        assert 'ActivityAnalyzer' in str(report)
        assert 'GraphAnalyzer' in str(report)


@pytest.mark.contract
class TestAnalyzerContracts:
    """Test analyzer contract compliance."""
    
    @pytest.mark.parametrize("analyzer_class", ALL_ANALYZERS)
    def test_required_metrics_declaration(self, analyzer_class):
        """Test that analyzers declare their required metrics."""
        analyzer = analyzer_class()
        
        if hasattr(analyzer, 'get_required_metrics'):
            required = analyzer.get_required_metrics()
            assert isinstance(required, set), "Required metrics must be a set"
            assert len(required) > 0, "Analyzer should require at least one metric"
    
    @pytest.mark.parametrize("analyzer_class", ALL_ANALYZERS)
    def test_output_contract_compliance(self, analyzer_class,
                                      analyzer_test_model,
                                      analyzer_context):
        """Test that analyzer outputs match their contracts."""
        tester = AnalyzerTester(analyzer_class)
        report = tester.create_appropriate_report()
        
        result = analyzer_class().analyze(analyzer_test_model, report, analyzer_context)
        
        # Check promised outputs exist
        contract = analyzer_class().contract
        for output_key in contract.provided_outputs:
            if '.' in output_key:
                parts = output_key.split('.')
                current = result
                for part in parts:
                    assert part in current, f"Missing promised output: {output_key}"
                    current = current[part]


# ===== Analyzer-Specific Test Helpers =====

class TestAnalyzerSpecificBehaviors:
    """Test specific behaviors of individual analyzers."""
    
    def test_homological_analyzer_computations(self, analyzer_test_model):
        """Test HomologicalAnalyzer specific computations."""
        analyzer = HomologicalAnalyzer()
        report = AnalyzerTestDataGenerator.create_homological_report()
        context = EvolutionContext()
        
        result = analyzer.analyze(analyzer_test_model, report, context)
        
        # Should compute topological summary
        assert 'topological_summary' in result
        assert 'recommendations' in result
        
        # Check for homological insights
        if 'metrics' in result:
            metrics = result['metrics']
            # Should reference homological concepts
            assert any(key for key in metrics if 'homolog' in key.lower() or 
                      'betti' in key.lower() or 'hole' in key.lower())
    
    def test_activity_analyzer_patterns(self, analyzer_test_model):
        """Test ActivityAnalyzer pattern detection."""
        analyzer = ActivityAnalyzer()
        report = AnalyzerTestDataGenerator.create_activity_report()
        context = EvolutionContext({
            'layer_activations': {
                f'layer_{i}': create_test_activations(sparsity=0.8)
                for i in range(4)
            }
        })
        
        result = analyzer.analyze(analyzer_test_model, report, context)
        
        # Should identify activity patterns
        assert 'activity_summary' in result or 'summary' in result
        
        # Should provide specific recommendations for dead neurons
        if report['metrics.NeuronActivityMetric']['dead_neurons'] > 10:
            recs = result.get('recommendations', [])
            assert any('dead' in rec.lower() or 'inactive' in rec.lower() 
                      for rec in recs)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])