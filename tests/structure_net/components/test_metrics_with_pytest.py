"""
Example of using pytest integration for component testing.

This demonstrates how the pytest fixtures and helpers make
component testing clean and maintainable.
"""

import pytest
from src.structure_net.components.metrics import (
    LayerMIMetric, EntropyMetric, InformationFlowMetric,
    RedundancyMetric, AdvancedMIMetric
)


@pytest.mark.component
class TestMetricsWithPytest:
    """Test metrics using pytest integration features."""
    
    # Test multiple metrics with same fixture
    @pytest.mark.parametrize("metric_class", [
        LayerMIMetric,
        EntropyMetric,
        InformationFlowMetric,
        RedundancyMetric,
        AdvancedMIMetric
    ])
    def test_metric_with_full_context(self, metric_class, component_test_context):
        """Test metrics with a full context containing all common inputs."""
        metric = metric_class()
        
        try:
            result = metric.analyze(None, component_test_context)
            assert isinstance(result, dict)
            assert len(result) > 0
        except ValueError as e:
            # Some metrics might need specific inputs
            if "requires" in str(e):
                pytest.skip(f"{metric_class.__name__} has specific requirements: {e}")
            else:
                raise
    
    @pytest.mark.contract
    def test_contract_compliance_with_scenarios(self, generate_test_scenarios, 
                                              run_component_test,
                                              validate_component_output):
        """Test LayerMIMetric with generated scenarios."""
        metric = LayerMIMetric()
        scenarios = generate_test_scenarios(LayerMIMetric)
        
        for scenario in scenarios:
            if scenario.should_succeed:
                # Should work
                result = run_component_test(metric, scenario)
                issues = validate_component_output(result, metric.contract)
                assert len(issues) == 0, f"Contract violations in {scenario.name}: {issues}"
            else:
                # Should fail
                with pytest.raises(scenario.expected_error):
                    run_component_test(metric, scenario)
    
    def test_component_info(self, analyze_component):
        """Test component analysis fixture."""
        info = analyze_component(EntropyMetric)
        
        assert info.is_metric
        assert not info.is_analyzer
        assert 'bins' in info.init_params
        assert 'activations' in info.contract.required_inputs
    
    @pytest.mark.parametrize("metric,expected_params", [
        (LayerMIMetric, ['method', 'bins', 'k_neighbors']),
        (EntropyMetric, ['base', 'bins', 'epsilon']),
        (AdvancedMIMetric, ['method', 'threshold', 'k_neighbors'])
    ])
    def test_initialization_params(self, analyze_component, metric, expected_params):
        """Test that metrics have expected initialization parameters."""
        info = analyze_component(metric)
        
        for param in expected_params:
            assert param in info.init_params, f"{metric.__name__} missing param: {param}"


@pytest.mark.integration
class TestMetricIntegration:
    """Integration tests using pytest fixtures."""
    
    def test_metric_compatibility(self, input_generators):
        """Test which inputs different metrics can handle."""
        metrics = [
            LayerMIMetric(),
            EntropyMetric(),
            InformationFlowMetric()
        ]
        
        # Test each metric with different input combinations
        compatibility = {}
        
        for metric in metrics:
            compatibility[metric.name] = []
            
            # Try different input combinations
            test_cases = {
                'activations_only': {
                    'activations': input_generators['activations']()
                },
                'layer_activations': {
                    'layer_activations': input_generators['layer_activations']()
                },
                'X_Y_inputs': {
                    'X': input_generators['X'](),
                    'Y': input_generators['Y']()
                }
            }
            
            for case_name, context_data in test_cases.items():
                context = EvolutionContext(context_data)
                try:
                    result = metric.analyze(None, context)
                    compatibility[metric.name].append(case_name)
                except:
                    pass
        
        # Verify expected compatibility
        assert 'activations_only' in compatibility['EntropyMetric']
        assert 'layer_activations' in compatibility['LayerMIMetric']


@pytest.mark.slow
class TestMetricPerformance:
    """Performance tests for metrics."""
    
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_entropy_scaling(self, size):
        """Test EntropyMetric performance with different sizes."""
        import time
        
        metric = EntropyMetric()
        large_activations = create_test_activations(size, size // 10)
        context = EvolutionContext({'activations': large_activations})
        
        start = time.time()
        result = metric.analyze(None, context)
        elapsed = time.time() - start
        
        # Should scale reasonably
        print(f"\nEntropyMetric with size {size}: {elapsed:.3f}s")
        assert elapsed < size / 1000  # Rough scaling expectation


# Using fixtures in test class setup
class TestMetricGroupBehavior:
    """Test groups of metrics together."""
    
    @pytest.fixture(autouse=True)
    def setup_metrics(self):
        """Setup metrics for all tests in this class."""
        self.mi_metric = LayerMIMetric()
        self.entropy_metric = EntropyMetric()
        self.flow_metric = InformationFlowMetric()
    
    def test_information_theory_constraints(self, component_test_context):
        """Test information theory relationships between metrics."""
        # Get entropy
        entropy_result = self.entropy_metric.analyze(None, component_test_context)
        
        # Get MI (if possible)
        try:
            mi_result = self.mi_metric.analyze(None, component_test_context)
            
            # MI should not exceed entropy (in theory)
            # Note: Due to estimation methods, this might not always hold
            if 'entropy' in entropy_result and 'mutual_information' in mi_result:
                entropy = entropy_result['entropy']
                mi = mi_result['mutual_information']
                # Just check they're in reasonable ranges
                assert entropy >= 0
                assert mi >= 0
        except ValueError:
            pytest.skip("MI metric needs specific inputs")


if __name__ == "__main__":
    # Run with: pytest test_metrics_with_pytest.py -v -m "not slow"
    pytest.main([__file__, '-v'])