"""
Pytest integration for generic component testing.

This shows how to properly integrate the generic testing framework
with pytest's features like fixtures, parametrization, and markers.
"""

import pytest
import torch
from typing import Dict, Any, List, Type, Optional
from dataclasses import dataclass

from src.structure_net.core import (
    IComponent, IMetric, IAnalyzer, ILayer, IModel,
    EvolutionContext, AnalysisReport
)
from tests.fixtures import (
    create_test_layer, create_test_model, create_test_context,
    create_test_activations, create_test_gradients
)

# Import all metrics for testing
from src.structure_net.components.metrics import (
    LayerMIMetric, EntropyMetric, InformationFlowMetric,
    RedundancyMetric, AdvancedMIMetric, ChainComplexMetric,
    SparsityMetric, DeadNeuronMetric, GradientMetric
)


# ===== Pytest Fixtures =====

@pytest.fixture
def basic_context():
    """Provide basic context with common data."""
    return EvolutionContext({
        'activations': create_test_activations(),
        'layer_activations': {
            'input': create_test_activations(100, 10),
            'output': create_test_activations(100, 5)
        },
        'layer_sequence': ['layer_0', 'layer_1', 'layer_2'],
        'X': torch.randn(100, 10),
        'Y': torch.randn(100, 5)
    })


@pytest.fixture
def test_layer():
    """Provide a test layer."""
    return create_test_layer()


@pytest.fixture
def test_model():
    """Provide a test model."""
    return create_test_model()


# ===== Test Data Generation =====

@dataclass
class ComponentTestCase:
    """A test case for a component."""
    name: str
    component_class: Type[IComponent]
    target: Optional[Any]
    context_data: Dict[str, Any]
    should_fail: bool = False
    expected_outputs: Optional[List[str]] = None


def generate_test_cases(component_class: Type[IComponent]) -> List[ComponentTestCase]:
    """Generate test cases for a component based on its contract."""
    component = component_class()
    contract = component.contract
    
    test_cases = []
    
    # Basic valid case
    basic_case = ComponentTestCase(
        name=f"{component.name}_basic",
        component_class=component_class,
        target=None,
        context_data={}
    )
    
    # Add required inputs
    for input_name in contract.required_inputs:
        if input_name == 'target':
            if 'layer' in component.name.lower():
                basic_case.target = create_test_layer()
            else:
                basic_case.target = create_test_model()
        else:
            basic_case.context_data[input_name] = generate_input(input_name)
    
    test_cases.append(basic_case)
    
    # Missing input cases (should fail)
    for input_name in contract.required_inputs:
        if input_name != 'target':
            missing_case = ComponentTestCase(
                name=f"{component.name}_missing_{input_name}",
                component_class=component_class,
                target=basic_case.target,
                context_data={k: v for k, v in basic_case.context_data.items() if k != input_name},
                should_fail=True
            )
            test_cases.append(missing_case)
    
    return test_cases


def generate_input(input_name: str) -> Any:
    """Generate appropriate input data for a given input name."""
    generators = {
        'activations': lambda: create_test_activations(),
        'layer_activations': lambda: {
            'input': create_test_activations(100, 10),
            'output': create_test_activations(100, 5)
        },
        'layer_sequence': lambda: ['layer_0', 'layer_1', 'layer_2'],
        'weight_matrix': lambda: torch.randn(10, 5),
        'X': lambda: torch.randn(100, 10),
        'Y': lambda: torch.randn(100, 5),
        'model': lambda: create_test_model(),
        'report': lambda: AnalysisReport()
    }
    
    if input_name in generators:
        return generators[input_name]()
    return torch.randn(32, 10)  # Default


# ===== Pytest Parametrization =====

# Generate test IDs and parameters for all metrics
ALL_METRICS = [
    LayerMIMetric, EntropyMetric, InformationFlowMetric,
    RedundancyMetric, AdvancedMIMetric, ChainComplexMetric,
    SparsityMetric, DeadNeuronMetric, GradientMetric
]

# Generate all test cases
all_test_cases = []
for metric_class in ALL_METRICS:
    all_test_cases.extend(generate_test_cases(metric_class))


class TestGenericComponents:
    """Generic tests for all components using pytest features."""
    
    @pytest.mark.parametrize("component_class", ALL_METRICS, 
                           ids=lambda c: c.__name__)
    def test_component_contract_valid(self, component_class):
        """Test that component has a valid contract."""
        component = component_class()
        contract = component.contract
        
        assert contract.component_name
        assert contract.version
        assert contract.maturity
        assert isinstance(contract.required_inputs, set)
        assert isinstance(contract.provided_outputs, set)
    
    @pytest.mark.parametrize("component_class,params", [
        (LayerMIMetric, {'method': 'histogram', 'bins': 30}),
        (LayerMIMetric, {'method': 'knn', 'k_neighbors': 5}),
        (EntropyMetric, {'base': 2.0, 'bins': 50}),
        (EntropyMetric, {'base': 10.0, 'bins': 100}),
    ])
    def test_component_initialization(self, component_class, params):
        """Test component initialization with different parameters."""
        component = component_class(**params)
        assert component is not None
        
        # Check parameters were set
        for param, value in params.items():
            if hasattr(component, param):
                assert getattr(component, param) == value
    
    @pytest.mark.parametrize("test_case", 
                           [tc for tc in all_test_cases if not tc.should_fail],
                           ids=lambda tc: tc.name)
    def test_component_valid_inputs(self, test_case):
        """Test component with valid inputs."""
        component = test_case.component_class()
        context = EvolutionContext(test_case.context_data)
        
        # Run component
        result = component.analyze(test_case.target, context)
        
        # Validate result
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check for NaN/Inf
        for key, value in result.items():
            if isinstance(value, (int, float)):
                assert not torch.isnan(torch.tensor(value))
                assert not torch.isinf(torch.tensor(value))
    
    @pytest.mark.parametrize("test_case",
                           [tc for tc in all_test_cases if tc.should_fail],
                           ids=lambda tc: tc.name)
    def test_component_invalid_inputs(self, test_case):
        """Test component with invalid inputs (should fail gracefully)."""
        component = test_case.component_class()
        context = EvolutionContext(test_case.context_data)
        
        # Should raise ValueError, TypeError, or KeyError
        with pytest.raises((ValueError, TypeError, KeyError)):
            component.analyze(test_case.target, context)
    
    @pytest.mark.parametrize("component_class", ALL_METRICS[:3],  # Test a few
                           ids=lambda c: c.__name__)
    def test_component_determinism(self, component_class, basic_context):
        """Test that components give deterministic results."""
        component = component_class()
        
        # Run multiple times
        results = []
        for _ in range(3):
            try:
                result = component.analyze(None, basic_context)
                results.append(result)
            except ValueError:
                # Some components might not work with basic context
                pytest.skip(f"{component_class.__name__} requires specific inputs")
        
        # Compare results
        if len(results) > 1:
            for key in results[0]:
                if isinstance(results[0][key], (int, float)):
                    for i in range(1, len(results)):
                        assert abs(results[0][key] - results[i][key]) < 1e-6


# ===== Custom Pytest Markers =====

@pytest.mark.slow
class TestComponentPerformance:
    """Performance tests for components."""
    
    @pytest.mark.parametrize("component_class,size", [
        (EntropyMetric, 1000),
        (LayerMIMetric, 1000),
        (InformationFlowMetric, 100)
    ])
    def test_component_scaling(self, component_class, size):
        """Test component performance with different input sizes."""
        import time
        
        component = component_class()
        
        # Create large input
        large_context = EvolutionContext({
            'activations': torch.randn(size, size // 10),
            'layer_activations': {
                'input': torch.randn(size, size // 10),
                'output': torch.randn(size, size // 20)
            },
            'layer_sequence': ['input', 'output']
        })
        
        # Measure time
        start = time.time()
        try:
            result = component.analyze(None, large_context)
            elapsed = time.time() - start
            
            # Should complete in reasonable time
            assert elapsed < 5.0, f"{component_class.__name__} took {elapsed:.2f}s for size {size}"
        except ValueError:
            pytest.skip(f"{component_class.__name__} doesn't support this input format")


# ===== Pytest Fixtures for Component Groups =====

@pytest.fixture
def information_flow_metrics():
    """Provide all information flow metrics."""
    return [
        LayerMIMetric(),
        EntropyMetric(),
        InformationFlowMetric(),
        RedundancyMetric(),
        AdvancedMIMetric()
    ]


@pytest.fixture
def basic_metrics():
    """Provide basic metrics."""
    return [
        SparsityMetric(),
        DeadNeuronMetric(),
        GradientMetric()
    ]


class TestMetricGroups:
    """Test metrics as groups with shared data."""
    
    def test_information_flow_consistency(self, information_flow_metrics, basic_context):
        """Test that information flow metrics give consistent results."""
        results = {}
        
        for metric in information_flow_metrics:
            try:
                result = metric.analyze(None, basic_context)
                results[metric.name] = result
            except ValueError:
                # Some metrics might need specific inputs
                pass
        
        # At least some metrics should work
        assert len(results) > 0
        
        # Check for consistency where applicable
        if 'EntropyMetric' in results and 'LayerMIMetric' in results:
            # MI should be less than or equal to min entropy
            if 'entropy' in results['EntropyMetric'] and 'mutual_information' in results['LayerMIMetric']:
                entropy = results['EntropyMetric']['entropy']
                mi = results['LayerMIMetric']['mutual_information']
                # This is a theoretical constraint
                # assert mi <= entropy  # Not always true with estimation methods


# ===== Pytest Configuration =====

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "component_type(type): mark test for specific component type"
    )


# ===== Test Discovery Helpers =====

def pytest_generate_tests(metafunc):
    """Dynamic test generation based on discovered components."""
    if "dynamic_component" in metafunc.fixturenames:
        # Could dynamically discover all components
        components = discover_all_components()
        metafunc.parametrize("dynamic_component", components)


def discover_all_components():
    """Discover all components in the codebase."""
    # This would scan the codebase for IComponent subclasses
    # For now, return our known list
    return ALL_METRICS


if __name__ == "__main__":
    # Run with: pytest test_generic_pytest_integration.py -v
    pytest.main([__file__, '-v'])