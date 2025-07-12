# Component Testing Strategy

The Structure Net testing framework provides two complementary approaches for testing components:

## 1. Component-Level Test Pipelines

Located in `pipelines/` directory, these provide:
- **Specialized testing** for specific component types
- **Custom validation logic** tailored to each component
- **Component-specific edge cases** and scenarios
- **Fine-grained control** over test data and validation

Example:
```python
class ActivityAnalyzerPipeline(AnalyzerTestPipeline):
    def create_model(self) -> IModel:
        # Creates a model with specific properties (dead neurons)
        model = create_test_model([50, 40, 30, 20])
        with torch.no_grad():
            model.layers[1].weight[:10, :] *= 0.001  # Create dead neurons
        return model
```

## 2. Unified Testing Approach

Located in `test_analyzers_unified.py` and similar files, providing:
- **Consistent testing** across all components
- **Automatic test generation** from contracts
- **Parametrized testing** for comprehensive coverage
- **Context and report generation** based on component type

Example:
```python
@pytest.mark.parametrize("analyzer_class", ALL_ANALYZERS)
def test_analyzer_basic_functionality(self, analyzer_class, test_models, 
                                    metric_data_generators):
    # Automatically generates appropriate context and reports
```

## When to Use Each Approach

### Use Component-Level Pipelines When:
- Testing **specific behaviors** unique to a component
- Need **custom test data** (e.g., models with dead neurons)
- Validating **specialized outputs** with domain knowledge
- Testing **edge cases** specific to the component type

### Use Unified Approach When:
- Running **comprehensive tests** across all components
- Ensuring **contract compliance** and consistency
- Need **automated test generation** from contracts
- Testing **common behaviors** across component types

## Both Approaches Work Together

The two approaches are complementary, not competitive:

```python
# Use pipeline for specialized setup
pipeline = ActivityAnalyzerPipeline()
model = pipeline.create_model()  # Model with dead neurons

# Use unified for broad validation
test_instance = TestAnalyzersUnified()
test_instance.test_analyzer_determinism(ActivityAnalyzer, ...)
```

## File Organization

```
tests/components/
├── pipelines/                    # Component-level pipelines
│   ├── base_test_pipeline.py    # Base classes
│   ├── metric_pipelines.py      # Metric-specific pipelines
│   └── analyzer_pipelines.py    # Analyzer-specific pipelines
├── test_analyzers_unified.py    # Unified analyzer testing
├── test_all_components_unified.py # Universal component testing
└── test_component_pipelines.py  # Pipeline test runner
```

## Key Benefits

1. **Flexibility**: Choose the right approach for your testing needs
2. **Coverage**: Comprehensive testing from multiple angles
3. **Maintainability**: Shared code and consistent patterns
4. **Scalability**: Easy to add new components and tests
5. **Clarity**: Clear separation of concerns

Both approaches ensure our components are thoroughly tested and behave correctly in all scenarios.