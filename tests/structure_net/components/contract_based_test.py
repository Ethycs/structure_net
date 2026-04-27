"""
Contract-based testing framework.

This framework uses the ComponentContract to automatically generate
appropriate test cases, making testing more maintainable and ensuring
contracts accurately reflect component behavior.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, Set, Type, List, Tuple
from dataclasses import dataclass

from src.structure_net.core import (
    IComponent, IMetric, IAnalyzer, ILayer, IModel,
    EvolutionContext, AnalysisReport, ComponentContract
)
from tests.fixtures import (
    create_test_layer, create_test_model, create_test_context,
    create_test_activations, create_test_gradients
)


@dataclass
class InputSpec:
    """Specification for generating test inputs."""
    name: str
    generator: callable
    variants: List[Dict[str, Any]]  # Different versions to test


class ContractBasedTestGenerator:
    """Generate tests based on component contracts."""
    
    # Input generators for common patterns
    INPUT_GENERATORS = {
        'activations': InputSpec(
            'activations',
            lambda: create_test_activations(),
            [
                {'batch_size': 32, 'features': 10},
                {'batch_size': 1, 'features': 100},  # Edge case
                {'batch_size': 1000, 'features': 5}  # Large batch
            ]
        ),
        'layer_activations': InputSpec(
            'layer_activations',
            lambda: {
                'layer_0': create_test_activations(),
                'layer_1': create_test_activations(),
                'layer_2': create_test_activations()
            },
            [
                {  # Direct input/output format
                    'input': create_test_activations(100, 10),
                    'output': create_test_activations(100, 5)
                },
                {  # Dictionary format with nested data
                    'layer_0': {
                        'input': create_test_activations(),
                        'output': create_test_activations()
                    }
                }
            ]
        ),
        'layer_sequence': InputSpec(
            'layer_sequence',
            lambda: ['layer_0', 'layer_1', 'layer_2'],
            [
                ['a', 'b', 'c'],  # Different names
                ['layer_0'],  # Single layer
                ['l1', 'l2', 'l3', 'l4', 'l5']  # More layers
            ]
        ),
        'gradients': InputSpec(
            'gradients',
            lambda: create_test_gradients((10, 10)),
            [
                {'shape': (5, 5), 'scale': 0.1},
                {'shape': (100, 100), 'scale': 10.0}
            ]
        ),
        'weight_matrix': InputSpec(
            'weight_matrix',
            lambda: torch.randn(10, 5),
            [
                torch.zeros(5, 5),  # Zero matrix
                torch.eye(10),  # Identity matrix
                torch.randn(100, 50)  # Large matrix
            ]
        ),
        'X': InputSpec(
            'X',
            lambda: torch.randn(100, 10),
            [
                torch.randn(50, 5),
                torch.zeros(100, 10),
                torch.ones(200, 20)
            ]
        ),
        'Y': InputSpec(
            'Y', 
            lambda: torch.randn(100, 5),
            [
                torch.randn(50, 3),
                torch.zeros(100, 5),
                torch.ones(200, 10)
            ]
        ),
        'model': InputSpec(
            'model',
            create_test_model,
            [
                create_test_model([5, 5, 5]),
                create_test_model([100, 50, 25, 10])
            ]
        ),
        'report': InputSpec(
            'report',
            AnalysisReport,
            [
                AnalysisReport({'existing': 'data'})
            ]
        ),
        'target': InputSpec(
            'target',
            create_test_layer,
            [
                create_test_model(),
                create_test_layer(20, 10),
                None
            ]
        )
    }
    
    def __init__(self, component_class: Type[IComponent]):
        self.component_class = component_class
        self.contract = component_class().contract
    
    def generate_test_cases(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate test cases based on contract."""
        test_cases = []
        
        # 1. Minimal valid case (only required inputs)
        minimal_case = self._create_minimal_case()
        test_cases.append(("minimal", minimal_case))
        
        # 2. Full case (all required + optional inputs)
        full_case = self._create_full_case()
        test_cases.append(("full", full_case))
        
        # 3. Variants for each required input
        for input_name in self.contract.required_inputs:
            if input_name in self.INPUT_GENERATORS:
                spec = self.INPUT_GENERATORS[input_name]
                for i, variant in enumerate(spec.variants):
                    variant_case = self._create_variant_case(input_name, variant)
                    test_cases.append((f"{input_name}_variant_{i}", variant_case))
        
        # 4. Missing input cases (should fail)
        for input_name in self.contract.required_inputs:
            if input_name != 'target':  # Can't remove target from analyze() call
                missing_case = self._create_missing_input_case(input_name)
                test_cases.append((f"missing_{input_name}", missing_case))
        
        return test_cases
    
    def _create_minimal_case(self) -> Dict[str, Any]:
        """Create minimal valid test case."""
        case = {'target': None, 'context': {}}
        
        for input_name in self.contract.required_inputs:
            if input_name == 'target':
                # Determine appropriate target
                if 'layer' in self.component_class.__name__.lower():
                    case['target'] = create_test_layer()
                elif 'model' in self.component_class.__name__.lower():
                    case['target'] = create_test_model()
            elif input_name in self.INPUT_GENERATORS:
                case['context'][input_name] = self.INPUT_GENERATORS[input_name].generator()
        
        return case
    
    def _create_full_case(self) -> Dict[str, Any]:
        """Create test case with all inputs."""
        case = self._create_minimal_case()
        
        # Add optional inputs
        for input_name in self.contract.optional_inputs:
            if input_name in self.INPUT_GENERATORS:
                case['context'][input_name] = self.INPUT_GENERATORS[input_name].generator()
        
        return case
    
    def _create_variant_case(self, input_name: str, variant: Any) -> Dict[str, Any]:
        """Create test case with input variant."""
        case = self._create_minimal_case()
        
        if input_name == 'target':
            case['target'] = variant
        else:
            # For variants that are dicts with parameters, generate new data
            if isinstance(variant, dict) and not all(isinstance(v, torch.Tensor) for v in variant.values()):
                if input_name == 'activations':
                    case['context'][input_name] = create_test_activations(**variant)
                elif input_name == 'gradients':
                    case['context'][input_name] = create_test_gradients(**variant)
            else:
                case['context'][input_name] = variant
        
        return case
    
    def _create_missing_input_case(self, missing_input: str) -> Dict[str, Any]:
        """Create test case missing a required input."""
        case = self._create_minimal_case()
        if missing_input in case['context']:
            del case['context'][missing_input]
        return case
    
    def validate_output_contract(self, output: Dict[str, Any]) -> List[str]:
        """Validate output matches contract. Returns list of issues."""
        issues = []
        
        # Check promised outputs
        for output_key in self.contract.provided_outputs:
            if '.' in output_key:
                # Handle nested keys
                parts = output_key.split('.')
                if parts[0] == 'metrics':
                    # For metrics, outputs are usually direct
                    if parts[1] not in output:
                        issues.append(f"Missing promised output: {parts[1]}")
            else:
                if output_key not in output:
                    issues.append(f"Missing promised output: {output_key}")
        
        return issues


class ContractTestRunner:
    """Run contract-based tests for a component."""
    
    def __init__(self, component_class: Type[IComponent]):
        self.component_class = component_class
        self.generator = ContractBasedTestGenerator(component_class)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all generated test cases."""
        results = {
            'passed': [],
            'failed': [],
            'errors': []
        }
        
        component = self.component_class()
        test_cases = self.generator.generate_test_cases()
        
        for case_name, case_data in test_cases:
            try:
                # Run the component
                context = EvolutionContext(case_data['context'])
                
                if isinstance(component, IAnalyzer):
                    # Analyzers need special handling
                    model = case_data['target'] or create_test_model()
                    report = case_data['context'].get('report', AnalysisReport())
                    output = component.analyze(model, report, context)
                else:
                    output = component.analyze(case_data['target'], context)
                
                # Validate output
                issues = self.generator.validate_output_contract(output)
                
                if issues:
                    results['failed'].append({
                        'case': case_name,
                        'issues': issues
                    })
                else:
                    results['passed'].append(case_name)
                    
            except (ValueError, TypeError, KeyError) as e:
                # Expected errors for invalid inputs
                if 'missing_' in case_name:
                    results['passed'].append(f"{case_name} (correctly failed)")
                else:
                    results['errors'].append({
                        'case': case_name,
                        'error': str(e)
                    })
            except Exception as e:
                results['errors'].append({
                    'case': case_name,
                    'error': f"{type(e).__name__}: {str(e)}"
                })
        
        return results
    
    def print_report(self, results: Dict[str, Any]):
        """Print test results."""
        total = len(results['passed']) + len(results['failed']) + len(results['errors'])
        
        print(f"\n{self.component_class.__name__} Test Results:")
        print(f"  Total: {total}")
        print(f"  Passed: {len(results['passed'])}")
        print(f"  Failed: {len(results['failed'])}")
        print(f"  Errors: {len(results['errors'])}")
        
        if results['failed']:
            print("\n  Failed Cases:")
            for failure in results['failed']:
                print(f"    - {failure['case']}: {', '.join(failure['issues'])}")
        
        if results['errors']:
            print("\n  Error Cases:")
            for error in results['errors']:
                print(f"    - {error['case']}: {error['error']}")


def test_component_with_contract(component_class: Type[IComponent]):
    """Test a component using contract-based testing."""
    runner = ContractTestRunner(component_class)
    results = runner.run_all_tests()
    runner.print_report(results)
    
    # Assert no unexpected failures
    assert len(results['errors']) == 0, f"Unexpected errors in {component_class.__name__}"
    assert len(results['failed']) == 0, f"Contract violations in {component_class.__name__}"


# Example test
if __name__ == "__main__":
    from src.structure_net.components.metrics import (
        LayerMIMetric, EntropyMetric, InformationFlowMetric
    )
    
    # Test a few components
    for component_class in [LayerMIMetric, EntropyMetric, InformationFlowMetric]:
        test_component_with_contract(component_class)