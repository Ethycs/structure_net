"""
Generic component testing framework.

This provides a universal testing approach for all components in the 
Structure Net architecture, automatically handling common patterns and
reducing boilerplate in individual tests.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, List, Type, Optional, Union, Set
from abc import ABC, abstractmethod
import inspect

from src.structure_net.core import (
    IComponent, IMetric, IAnalyzer, ILayer, IModel,
    EvolutionContext, AnalysisReport, ComponentContract
)
from tests.fixtures import (
    create_test_layer, create_test_model, create_test_context,
    create_test_activations, create_test_gradients
)


class ComponentTestSpec:
    """Specification for testing a component."""
    
    def __init__(self, component_class: Type[IComponent]):
        self.component_class = component_class
        self.component = component_class()
        self.contract = self.component.contract
        
        # Analyze component to determine its type
        self.is_metric = issubclass(component_class, IMetric)
        self.is_analyzer = issubclass(component_class, IAnalyzer)
        
        # Extract parameter info from __init__
        self.init_params = self._extract_init_params()
        
    def _extract_init_params(self) -> Dict[str, Any]:
        """Extract parameter names and defaults from __init__."""
        sig = inspect.signature(self.component_class.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name not in ['self', 'name']:
                params[name] = param.default if param.default != param.empty else None
        return params
    
    def get_test_scenarios(self) -> List[Dict[str, Any]]:
        """Generate test scenarios based on contract."""
        scenarios = []
        
        # Basic scenario with minimal inputs
        basic_scenario = self._create_basic_scenario()
        scenarios.append(basic_scenario)
        
        # Scenario for each optional input
        for optional_input in self.contract.optional_inputs:
            scenario = self._create_scenario_with_optional(basic_scenario, optional_input)
            scenarios.append(scenario)
        
        # Edge cases
        scenarios.extend(self._create_edge_case_scenarios())
        
        return scenarios
    
    def _create_basic_scenario(self) -> Dict[str, Any]:
        """Create basic test scenario with required inputs."""
        scenario = {
            'name': 'basic',
            'target': None,
            'context': {}
        }
        
        # Determine target based on required inputs
        if 'target' in self.contract.required_inputs:
            if 'layer' in self.component.name.lower():
                scenario['target'] = create_test_layer()
            else:
                scenario['target'] = create_test_model()
        
        # Add required context data
        for input_key in self.contract.required_inputs:
            if input_key == 'target':
                continue
            scenario['context'][input_key] = self._generate_input_data(input_key)
        
        return scenario
    
    def _generate_input_data(self, input_key: str) -> Any:
        """Generate appropriate test data for an input key."""
        # Common patterns
        if 'activations' in input_key:
            return create_test_activations()
        elif 'gradients' in input_key:
            return create_test_gradients((10, 10))
        elif 'layer_activations' in input_key:
            # Special case for layer activation data
            if hasattr(self.component, '_compute_direct_mi'):
                # LayerMIMetric style
                return {
                    'input': create_test_activations(100, 10),
                    'output': create_test_activations(100, 5)
                }
            else:
                # General layer activation dict
                return {
                    f'layer_{i}': create_test_activations()
                    for i in range(3)
                }
        elif 'layer_sequence' in input_key:
            return ['layer_0', 'layer_1', 'layer_2']
        elif 'weight_matrix' in input_key:
            return torch.randn(10, 5)
        elif 'X' in input_key:
            return torch.randn(100, 10)
        elif 'Y' in input_key:
            return torch.randn(100, 5)
        elif 'model' in input_key:
            return create_test_model()
        elif 'report' in input_key:
            return AnalysisReport()
        else:
            # Default: return a tensor
            return torch.randn(32, 10)
    
    def _create_scenario_with_optional(self, base_scenario: Dict[str, Any], 
                                     optional_key: str) -> Dict[str, Any]:
        """Create scenario with an optional input added."""
        scenario = base_scenario.copy()
        scenario['name'] = f'with_{optional_key}'
        scenario['context'] = base_scenario['context'].copy()
        scenario['context'][optional_key] = self._generate_input_data(optional_key)
        return scenario
    
    def _create_edge_case_scenarios(self) -> List[Dict[str, Any]]:
        """Create edge case scenarios."""
        scenarios = []
        
        # Empty/zero inputs
        zero_scenario = {
            'name': 'zero_input',
            'target': create_test_layer() if 'target' in self.contract.required_inputs else None,
            'context': {}
        }
        for input_key in self.contract.required_inputs:
            if input_key == 'target':
                continue
            if 'activations' in input_key:
                zero_scenario['context'][input_key] = torch.zeros(10, 5)
            else:
                zero_scenario['context'][input_key] = self._generate_input_data(input_key)
        scenarios.append(zero_scenario)
        
        # Large inputs
        large_scenario = {
            'name': 'large_input',
            'target': create_test_layer() if 'target' in self.contract.required_inputs else None,
            'context': {}
        }
        for input_key in self.contract.required_inputs:
            if input_key == 'target':
                continue
            if 'activations' in input_key:
                large_scenario['context'][input_key] = torch.randn(1000, 100)
            else:
                large_scenario['context'][input_key] = self._generate_input_data(input_key)
        scenarios.append(large_scenario)
        
        return scenarios


class GenericComponentTester:
    """Generic tester that can test any component."""
    
    def __init__(self, component_class: Type[IComponent]):
        self.spec = ComponentTestSpec(component_class)
    
    def test_contract_validity(self):
        """Test that the contract is well-formed."""
        contract = self.spec.contract
        
        assert isinstance(contract.component_name, str)
        assert len(contract.component_name) > 0
        assert contract.version is not None
        assert contract.maturity is not None
        assert isinstance(contract.required_inputs, set)
        assert isinstance(contract.provided_outputs, set)
        assert contract.resources is not None
    
    def test_initialization_variants(self):
        """Test component can be initialized with various parameters."""
        # Default initialization
        component = self.spec.component_class()
        assert component is not None
        
        # Try with each parameter
        for param_name, default_value in self.spec.init_params.items():
            if param_name in ['method', 'mi_method']:
                # Try different methods
                for method in ['histogram', 'knn', 'binning']:
                    try:
                        component = self.spec.component_class(**{param_name: method})
                        assert component is not None
                    except:
                        pass  # Some methods might not be supported
            elif param_name in ['bins', 'n_bins']:
                # Try different bin counts
                component = self.spec.component_class(**{param_name: 20})
                assert component is not None
    
    def test_all_scenarios(self):
        """Test component with all generated scenarios."""
        scenarios = self.spec.get_test_scenarios()
        
        for scenario in scenarios:
            self._test_scenario(scenario)
    
    def _test_scenario(self, scenario: Dict[str, Any]):
        """Test a single scenario."""
        component = self.spec.component_class()
        
        # Create context
        context = EvolutionContext(scenario['context'])
        
        # Run component
        try:
            if self.spec.is_metric:
                result = component.analyze(scenario['target'], context)
            elif self.spec.is_analyzer:
                # Analyzers need a model and report
                model = scenario['target'] or create_test_model()
                report = scenario['context'].get('report', AnalysisReport())
                result = component.analyze(model, report, context)
            else:
                # Other component types
                result = component.analyze(scenario['target'], context)
            
            # Validate result
            self._validate_result(result, scenario)
            
        except ValueError as e:
            # Some scenarios might legitimately fail
            if scenario['name'] in ['zero_input', 'invalid_input']:
                pass  # Expected failure
            else:
                pytest.fail(f"Scenario '{scenario['name']}' failed: {e}")
    
    def _validate_result(self, result: Dict[str, Any], scenario: Dict[str, Any]):
        """Validate component output."""
        # Check result is a dictionary
        assert isinstance(result, dict), f"Result must be dict, got {type(result)}"
        
        # Check promised outputs exist
        for output_key in self.spec.contract.provided_outputs:
            if '.' in output_key:
                # Handle nested keys like 'metrics.accuracy'
                parts = output_key.split('.')
                if parts[0] == 'metrics' and self.spec.is_metric:
                    # For metrics, the outputs are directly in result
                    assert parts[1] in result, f"Missing output: {parts[1]}"
            else:
                assert output_key in result, f"Missing output: {output_key}"
        
        # Validate numeric outputs
        for key, value in result.items():
            if isinstance(value, (int, float, np.number)):
                assert not np.isnan(value), f"NaN value for {key}"
                assert not np.isinf(value), f"Inf value for {key}"
    
    def test_determinism(self):
        """Test that component gives consistent results."""
        scenario = self.spec.get_test_scenarios()[0]  # Use basic scenario
        component = self.spec.component_class()
        context = EvolutionContext(scenario['context'])
        
        # Run multiple times
        results = []
        for _ in range(3):
            if self.spec.is_metric:
                result = component.analyze(scenario['target'], context)
            else:
                result = component.analyze(scenario['target'], context)
            results.append(result)
        
        # Compare results (with tolerance for floating point)
        for key in results[0]:
            if isinstance(results[0][key], (int, float)):
                for i in range(1, len(results)):
                    assert abs(results[0][key] - results[i][key]) < 1e-6, \
                        f"Non-deterministic result for {key}"


def create_component_test_class(component_class: Type[IComponent]):
    """Dynamically create a test class for a component."""
    
    class DynamicComponentTest:
        component_class = component_class
        
        def test_contract(self):
            tester = GenericComponentTester(self.component_class)
            tester.test_contract_validity()
        
        def test_initialization(self):
            tester = GenericComponentTester(self.component_class)
            tester.test_initialization_variants()
        
        def test_scenarios(self):
            tester = GenericComponentTester(self.component_class)
            tester.test_all_scenarios()
        
        def test_determinism(self):
            tester = GenericComponentTester(self.component_class)
            tester.test_determinism()
    
    # Set class name
    DynamicComponentTest.__name__ = f"Test{component_class.__name__}"
    
    return DynamicComponentTest


# Example usage: Auto-generate tests for all metrics
if __name__ == "__main__":
    from src.structure_net.components.metrics import (
        SparsityMetric, DeadNeuronMetric, EntropyMetric,
        LayerMIMetric, GradientMetric
    )
    
    # Create test classes
    TestSparsityMetric = create_component_test_class(SparsityMetric)
    TestDeadNeuronMetric = create_component_test_class(DeadNeuronMetric)
    TestEntropyMetric = create_component_test_class(EntropyMetric)
    TestLayerMIMetric = create_component_test_class(LayerMIMetric)
    TestGradientMetric = create_component_test_class(GradientMetric)