# Structure Net Test Suite

## Overview

This directory contains the test suite for Structure Net, including tests for core functionality, evolution systems, NAL integration, and the new data factory components.

## Test Files

### Core Tests
- `test_core_functionality.py` - Tests for basic network operations, layers, and validation
- `test_evolution.py` - Tests for evolution system components
- `test_nal.py` - Tests for Neural Architecture Lab integration
- `test_performance.py` - Performance benchmarks

### New Integration Tests
- `test_data_factory_integration.py` - Tests for ChromaDB and HDF5 storage integration
- `test_stress_test_memory.py` - Tests for memory optimizations in Ultimate Stress Test v2

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_data_factory_integration.py -v
```

### Run tests without GPU
```bash
pytest -m "not gpu"
```

### Run only fast tests (skip integration)
```bash
pytest -m "not integration"
```

### Run with coverage
```bash
pytest --cov=src/structure_net --cov=src/data_factory
```

## Test Categories

### Unit Tests
Fast, isolated tests for individual components:
- ChromaDB client operations
- HDF5 storage functions
- Memory cleanup utilities
- Profiling tools

### Integration Tests
Tests that verify component interactions:
- ChromaDB + HDF5 hybrid storage
- NAL integration with data factory
- Stress test memory management

### GPU Tests
Tests requiring CUDA:
- GPU memory tracking
- CUDA profiling
- GPU-accelerated metrics

## Key Test Fixtures

### `temp_dir`
Creates a temporary directory for test data that's automatically cleaned up.

### `chroma_config` / `timeseries_config`
Pre-configured storage configurations for testing.

### `mock_dataset`
Simulated dataset loaders for testing without real data.

### `device` / `device_cpu`
PyTorch devices for GPU/CPU testing.

## Test Data

Tests use synthetic data and mocked components where possible to ensure:
- Fast execution
- Reproducibility
- No external dependencies

## Memory Testing

The `test_stress_test_memory.py` suite specifically tests:
1. **Dataset Sharing** - Ensures datasets are loaded once and reused
2. **Memory Cleanup** - Verifies NAL memory is properly released
3. **Result Caching** - Confirms results are saved to disk
4. **ChromaDB Integration** - Tests HDF5 storage for large data

## Storage Testing

The `test_data_factory_integration.py` suite tests:
1. **ChromaDB Operations** - Vector search and metadata storage
2. **HDF5 Storage** - Compressed time series data
3. **Hybrid Storage** - Combined ChromaDB + HDF5
4. **Error Handling** - Graceful degradation and fallbacks

## Writing New Tests

When adding tests:
1. Use appropriate markers (`@pytest.mark.gpu`, `@pytest.mark.integration`)
2. Create fixtures for reusable test data
3. Mock external dependencies
4. Test both success and failure cases
5. Include docstrings explaining what's being tested

Example:
```python
@pytest.mark.integration
def test_new_feature(temp_dir, mock_config):
    """Test that new feature handles edge cases properly."""
    # Setup
    component = NewComponent(mock_config)
    
    # Execute
    result = component.process(test_data)
    
    # Assert
    assert result.status == "success"
    assert len(result.data) > 0
```

## Troubleshooting

### Import Errors
Ensure you're running from the project root:
```bash
cd /home/rabbit/structure_net
pytest
```

### GPU Tests Failing
If GPU tests fail on CPU-only systems:
```bash
pytest -m "not gpu"
```

### Slow Tests
Skip integration tests for faster runs:
```bash
pytest -m "not integration"
```

### Memory Issues
Some tests simulate memory pressure. If they fail:
- Check available system memory
- Run tests individually
- Reduce test data sizes in fixtures