# Generic Component Testing Framework

This directory contains a generic testing framework for the Structure Net component architecture that leverages the contract system to automatically generate comprehensive tests.

## Overview

The generic testing approach provides several benefits:

1. **Consistency**: All components are tested using the same framework
2. **Contract Validation**: Tests are generated from component contracts
3. **Maintainability**: Adding new components automatically gets basic test coverage
4. **Coverage**: Multiple test scenarios are generated for each component

## Components

### 1. Generic Component Tester (`generic_component_test.py`)

Provides a universal testing framework that:
- Automatically generates test scenarios based on component contracts
- Tests initialization variants
- Validates output compliance with contracts
- Tests determinism and edge cases

```python
# Example usage
from generic_component_test import GenericComponentTester

tester = GenericComponentTester(LayerMIMetric)
tester.test_all_scenarios()
```

### 2. Contract-Based Testing (`contract_based_test.py`)

Uses component contracts to generate test cases:
- Creates minimal valid cases (only required inputs)
- Creates full cases (all inputs)
- Tests input variants
- Tests missing input handling

```python
# Example usage
from contract_based_test import ContractTestRunner

runner = ContractTestRunner(EntropyMetric)
results = runner.run_all_tests()
runner.print_report(results)
```

### 3. Component Test Pipelines (`pipelines/`)

Provides specialized test pipelines for different component types:
- `MetricTestPipeline`: For IMetric components
- `AnalyzerTestPipeline`: For IAnalyzer components
- Validates contracts, inputs/outputs, error handling

## Test Generation Strategy

### Input Generation

The framework automatically generates appropriate test inputs based on common patterns:

```python
INPUT_GENERATORS = {
    'activations': lambda: create_test_activations(),
    'layer_activations': lambda: {
        'layer_0': create_test_activations(),
        'layer_1': create_test_activations()
    },
    'weight_matrix': lambda: torch.randn(10, 5),
    # ... more generators
}
```

### Scenario Generation

For each component, multiple scenarios are generated:

1. **Basic Scenario**: Minimal required inputs
2. **Optional Scenarios**: Each optional input added
3. **Edge Cases**: Zero inputs, large inputs, etc.
4. **Invalid Cases**: Missing required inputs

### Output Validation

Outputs are validated against the component contract:
- All promised outputs must be present
- Numeric values are checked for NaN/Inf
- Nested output keys are handled (e.g., "metrics.accuracy")

## Benefits Over Manual Testing

1. **Reduced Boilerplate**: No need to write repetitive test setup
2. **Contract Enforcement**: Contracts must accurately reflect behavior
3. **Automatic Coverage**: New components get basic tests for free
4. **Consistency**: All components tested the same way

## Usage Examples

### Testing All Metrics

```python
from test_all_metrics_generic import TestAllMetricsGeneric

# Run contract validation for all metrics
pytest -k "test_metric_contract"

# Run scenario tests for working metrics
pytest -k "test_metric_scenarios"
```

### Testing Specific Component

```python
# Create dynamic test class
TestMyMetric = create_component_test_class(MyMetric)

# Run all tests
test_instance = TestMyMetric()
test_instance.test_contract()
test_instance.test_scenarios()
```

### Debugging Contract Issues

When tests fail, it often reveals contract inaccuracies:

```
AssertionError: Missing promised output: mutual_information
```

This indicates the contract promises "metrics.mutual_information" but the component returns "mutual_information" directly.

## Best Practices

1. **Accurate Contracts**: Ensure contracts accurately reflect inputs/outputs
2. **Meaningful Defaults**: Component parameters should have sensible defaults
3. **Graceful Errors**: Components should fail gracefully with clear messages
4. **Consistent Naming**: Use consistent keys across similar components

## Future Enhancements

1. **Property-Based Testing**: Generate random valid inputs
2. **Performance Testing**: Measure and track component performance
3. **Compatibility Matrix**: Auto-generate which components work together
4. **Contract Linting**: Validate contracts follow conventions