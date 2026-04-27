# Componentized Testing Framework

This directory contains a fully componentized testing framework that provides comprehensive, maintainable testing for all components in the Structure Net architecture.

## Overview

The componentized testing approach treats both metrics and analyzers as components with contracts, allowing for unified testing strategies that:

1. **Automatically generate test cases** from component contracts
2. **Provide consistent testing** across all component types
3. **Reduce boilerplate** through shared fixtures and helpers
4. **Ensure contract accuracy** by validating promised behavior

## Key Components

### 1. Universal Component Tester (`test_all_components_unified.py`)

A single tester that works for any component type:

```python
class UniversalComponentTester:
    def run_all_tests(self):
        self._test_contract()
        self._test_initialization()
        self._test_basic_functionality()
        self._test_error_handling()
        self._test_performance()
```

### 2. Analyzer-Specific Testing (`test_analyzers_generic.py`)

Specialized testing for analyzers that:
- Generates appropriate metric reports
- Tests analyzer-specific behaviors
- Validates recommendations and insights

```python
class AnalyzerTester:
    def create_appropriate_report(self) -> AnalysisReport:
        # Creates report with required metrics based on analyzer type
    
    def test_basic_functionality(self):
        # Tests analyzer with appropriate data
```

### 3. Pytest Integration (`test_metrics_with_pytest.py`)

Full pytest integration with:
- Parametrized tests across all components
- Shared fixtures for common data
- Markers for test organization
- Performance and integration tests

```python
@pytest.mark.parametrize("component_class", ALL_COMPONENTS)
def test_component_contract(self, component_class):
    # Test any component's contract
```

### 4. Shared Fixtures (`conftest.py`)

Centralized fixtures providing:
- Input data generators
- Metric data generators
- Component analysis helpers
- Test scenario generation

## Testing Strategies

### Contract-Based Testing

Tests are generated from component contracts:
1. **Required inputs** → Test with/without each input
2. **Provided outputs** → Validate all outputs exist
3. **Resource requirements** → Test performance expectations

### Scenario Generation

Multiple scenarios per component:
- **Minimal**: Only required inputs
- **Full**: All inputs including optional
- **Variants**: Different input combinations
- **Edge cases**: Zero values, large inputs
- **Error cases**: Missing required inputs

### Type-Specific Behaviors

#### Metrics
- Test with different target types (Layer, Model, None)
- Validate numerical outputs
- Check for NaN/Inf values
- Test determinism

#### Analyzers
- Test with different report contents
- Validate recommendations
- Check insight generation
- Test metric dependencies

## Usage Examples

### Test All Components

```python
# Run all component tests
pytest tests/components/test_all_components_unified.py -v

# Test specific component type
pytest -k "test_all_metrics"
pytest -k "test_all_analyzers"
```

### Test Specific Component

```python
# Test single component with all scenarios
tester = UniversalComponentTester(EntropyMetric)
result = tester.run_all_tests(input_generators, metric_data_generators)
```

### Generate Test Report

```python
runner = ComponentTestRunner(input_generators, metric_data_generators)
runner.test_components([SparsityMetric, ActivityAnalyzer, ...])
report = runner.generate_report()
```

## Benefits

### 1. **Maintainability**
- Single place to update test logic
- New components automatically get tests
- Consistent testing patterns

### 2. **Coverage**
- Every component tested the same way
- Multiple scenarios per component
- Edge cases handled systematically

### 3. **Contract Enforcement**
- Contracts must be accurate
- Missing outputs detected
- Input validation verified

### 4. **Discoverability**
- Easy to see what each component needs
- Clear compatibility matrix
- Performance characteristics visible

### 5. **Integration**
- Components tested in isolation and together
- Analyzer-metric dependencies validated
- Full pipeline testing supported

## Adding New Components

To add a new component:

1. **Implement the component** with proper contract
2. **Add to component list** in test files
3. **Run tests** - they're automatically generated!

```python
# Component automatically gets:
# - Contract validation
# - Initialization tests
# - Input/output tests
# - Error handling tests
# - Performance tests
```

## Test Organization

```
tests/components/
├── conftest.py                     # Shared fixtures
├── test_all_components_unified.py  # Universal testing
├── test_metrics_with_pytest.py     # Pytest integration
├── test_analyzers_generic.py       # Analyzer-specific
├── generic_component_test.py       # Base framework
└── contract_based_test.py          # Contract validation
```

## Running Tests

```bash
# All component tests
pytest tests/components/ -v

# By marker
pytest -m "component"
pytest -m "contract"
pytest -m "integration"

# By speed
pytest -m "not slow"

# Specific component type
pytest -k "Metric" 
pytest -k "Analyzer"

# With coverage
pytest tests/components/ --cov=src.structure_net.components
```

## Future Enhancements

1. **Property-based testing** - Generate random valid inputs
2. **Mutation testing** - Verify test quality
3. **Benchmark suite** - Track performance over time
4. **Visual test reports** - Component compatibility matrices
5. **Auto-documentation** - Generate docs from contracts