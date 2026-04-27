"""
Pytest configuration and fixtures for component testing.

This conftest.py file provides shared fixtures and configuration
for all component tests, making the generic testing framework
available throughout the test suite.
"""

import pytest
import torch
from typing import Dict, Any, List, Type, Optional, Callable
from dataclasses import dataclass
import inspect

from src.structure_net.core import (
    IComponent, IMetric, IAnalyzer, ILayer, IModel,
    EvolutionContext, AnalysisReport, ComponentContract
)
from tests.fixtures import (
    create_test_layer, create_test_model, create_test_context,
    create_test_activations, create_test_gradients
)


# ===== Custom Markers =====

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "component: mark test as a component test"
    )
    config.addinivalue_line(
        "markers", "contract: mark test as a contract validation test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )


# ===== Analyzer-Specific Fixtures =====

@pytest.fixture
def metric_data_generators():
    """Generate metric data for different metric types."""
    return {
        'SparsityMetric': lambda: {
            'sparsity': 0.3,
            'density': 0.7,
            'zero_count': 300,
            'total_params': 1000
        },
        'EntropyMetric': lambda: {
            'entropy': 2.5,
            'normalized_entropy': 0.8,
            'effective_bits': 2.5,
            'entropy_ratio': 0.7
        },
        'LayerMIMetric': lambda: {
            'mutual_information': 1.2,
            'normalized_mi': 0.6,
            'information_ratio': 0.5,
            'layer_pairs': {'layer_0->layer_1': 1.2}
        },
        'ChainComplexMetric': lambda: {
            'chain_length': 3,
            'boundary_ranks': [5, 4],
            'chain_connectivity': 0.8,
            'complex_dimension': 3
        },
        'NeuronActivityMetric': lambda: {
            'active_ratio': 0.85,
            'dead_neurons': 15,
            'activity_variance': 0.2,
            'layer_activity': {'layer_0': 0.9, 'layer_1': 0.8}
        },
        'GraphStructureMetric': lambda: {
            'num_nodes': 100,
            'num_edges': 450,
            'density': 0.045,
            'average_degree': 9.0,
            'clustering_coefficient': 0.3
        }
    }


@pytest.fixture
def create_metric_report(metric_data_generators):
    """Create an AnalysisReport with specified metrics."""
    def _create(metric_names: List[str]) -> AnalysisReport:
        report = AnalysisReport()
        for metric_name in metric_names:
            if metric_name in metric_data_generators:
                data = metric_data_generators[metric_name]()
                report.add_metric_data(metric_name, data)
        return report
    return _create


# ===== Shared Fixtures =====

@pytest.fixture(scope="session")
def input_generators():
    """Provide input generators for different data types."""
    return {
        'activations': lambda **kwargs: create_test_activations(**kwargs),
        'layer_activations': lambda: {
            'input': create_test_activations(100, 10),
            'output': create_test_activations(100, 5)
        },
        'layer_sequence': lambda: ['layer_0', 'layer_1', 'layer_2'],
        'weight_matrix': lambda: torch.randn(10, 5),
        'gradients': lambda: create_test_gradients((10, 10)),
        'X': lambda: torch.randn(100, 10),
        'Y': lambda: torch.randn(100, 5),
        'Z': lambda: torch.randn(100, 5),
        'model': create_test_model,
        'report': AnalysisReport,
        'target': create_test_layer
    }


@pytest.fixture
def component_test_context(input_generators):
    """Create a context with all common inputs."""
    context_data = {}
    for key, generator in input_generators.items():
        if key not in ['model', 'target', 'report']:
            context_data[key] = generator()
    return EvolutionContext(context_data)


@pytest.fixture
def minimal_context():
    """Create minimal context for basic testing."""
    return EvolutionContext({
        'activations': create_test_activations()
    })


# ===== Component Analysis Fixtures =====

@dataclass
class ComponentInfo:
    """Information about a component for testing."""
    component_class: Type[IComponent]
    instance: IComponent
    contract: ComponentContract
    init_params: Dict[str, Any]
    is_metric: bool
    is_analyzer: bool
    required_target_type: Optional[Type]


@pytest.fixture
def analyze_component():
    """Fixture that analyzes a component class."""
    def _analyze(component_class: Type[IComponent]) -> ComponentInfo:
        instance = component_class()
        contract = instance.contract
        
        # Extract init parameters
        sig = inspect.signature(component_class.__init__)
        init_params = {}
        for name, param in sig.parameters.items():
            if name not in ['self', 'name']:
                init_params[name] = {
                    'default': param.default if param.default != param.empty else None,
                    'type': param.annotation if param.annotation != param.empty else Any
                }
        
        # Determine component type
        is_metric = issubclass(component_class, IMetric)
        is_analyzer = issubclass(component_class, IAnalyzer)
        
        # Determine required target type
        required_target_type = None
        if 'target' in contract.required_inputs:
            if 'layer' in component_class.__name__.lower():
                required_target_type = ILayer
            elif 'model' in component_class.__name__.lower():
                required_target_type = IModel
        
        return ComponentInfo(
            component_class=component_class,
            instance=instance,
            contract=contract,
            init_params=init_params,
            is_metric=is_metric,
            is_analyzer=is_analyzer,
            required_target_type=required_target_type
        )
    
    return _analyze


# ===== Test Case Generation =====

@dataclass
class TestScenario:
    """A test scenario for a component."""
    name: str
    description: str
    target: Optional[Any]
    context_data: Dict[str, Any]
    should_succeed: bool = True
    expected_error: Optional[Type[Exception]] = None


@pytest.fixture
def generate_test_scenarios(analyze_component, input_generators):
    """Generate test scenarios for a component."""
    def _generate(component_class: Type[IComponent]) -> List[TestScenario]:
        info = analyze_component(component_class)
        scenarios = []
        
        # 1. Minimal valid scenario
        minimal_data = {}
        for input_name in info.contract.required_inputs:
            if input_name != 'target' and input_name in input_generators:
                minimal_data[input_name] = input_generators[input_name]()
        
        target = None
        if info.required_target_type:
            if info.required_target_type == ILayer:
                target = create_test_layer()
            else:
                target = create_test_model()
        
        scenarios.append(TestScenario(
            name="minimal_valid",
            description="Minimal valid inputs",
            target=target,
            context_data=minimal_data,
            should_succeed=True
        ))
        
        # 2. Missing required inputs
        for input_name in info.contract.required_inputs:
            if input_name != 'target':
                missing_data = {k: v for k, v in minimal_data.items() if k != input_name}
                scenarios.append(TestScenario(
                    name=f"missing_{input_name}",
                    description=f"Missing required input: {input_name}",
                    target=target,
                    context_data=missing_data,
                    should_succeed=False,
                    expected_error=ValueError
                ))
        
        # 3. With optional inputs
        for optional_input in info.contract.optional_inputs:
            if optional_input in input_generators:
                with_optional = minimal_data.copy()
                with_optional[optional_input] = input_generators[optional_input]()
                scenarios.append(TestScenario(
                    name=f"with_{optional_input}",
                    description=f"With optional input: {optional_input}",
                    target=target,
                    context_data=with_optional,
                    should_succeed=True
                ))
        
        return scenarios
    
    return _generate


# ===== Component Test Helper =====

@pytest.fixture
def run_component_test():
    """Helper to run a component with proper error handling."""
    def _run(component: IComponent, scenario: TestScenario) -> Dict[str, Any]:
        context = EvolutionContext(scenario.context_data)
        
        if isinstance(component, IAnalyzer):
            # Analyzers need special handling
            model = scenario.target or create_test_model()
            report = scenario.context_data.get('report', AnalysisReport())
            return component.analyze(model, report, context)
        else:
            return component.analyze(scenario.target, context)
    
    return _run


# ===== Validation Helpers =====

@pytest.fixture
def validate_component_output():
    """Validate component output against contract."""
    def _validate(output: Dict[str, Any], contract: ComponentContract) -> List[str]:
        issues = []
        
        # Check promised outputs
        for output_key in contract.provided_outputs:
            if '.' in output_key:
                # Handle nested keys
                parts = output_key.split('.')
                if parts[0] == 'metrics':
                    # For metrics, outputs are usually direct
                    if parts[1] not in output:
                        issues.append(f"Missing output: {parts[1]}")
            else:
                if output_key not in output:
                    issues.append(f"Missing output: {output_key}")
        
        # Check for invalid values
        for key, value in output.items():
            if isinstance(value, (int, float)):
                if torch.isnan(torch.tensor(float(value))):
                    issues.append(f"NaN value for {key}")
                if torch.isinf(torch.tensor(float(value))):
                    issues.append(f"Inf value for {key}")
        
        return issues
    
    return _validate


# ===== Parametrization Helpers =====

def pytest_generate_tests(metafunc):
    """Dynamic test generation for components."""
    if "component_class" in metafunc.fixturenames:
        # Could be customized based on markers
        if metafunc.config.getoption("--all-components"):
            components = discover_all_components()
            metafunc.parametrize("component_class", components)


def discover_all_components():
    """Discover all component classes."""
    # This would scan for all IComponent subclasses
    # For now, return empty list (tests will provide their own)
    return []


# ===== CLI Options =====

def pytest_addoption(parser):
    """Add custom CLI options."""
    parser.addoption(
        "--all-components",
        action="store_true",
        default=False,
        help="Run tests on all discovered components"
    )
    parser.addoption(
        "--component-group",
        action="store",
        default=None,
        help="Run tests only on specified component group (metrics, analyzers, etc.)"
    )


# ===== Test Collection Hooks =====

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options."""
    if config.getoption("--component-group"):
        group = config.getoption("--component-group")
        # Filter tests based on component group
        # This is where you'd implement group-based filtering