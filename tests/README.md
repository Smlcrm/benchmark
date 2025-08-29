# Testing Framework Documentation

This document describes the testing framework for the benchmarking pipeline project.

## Overview

The testing framework is organized into multiple categories to ensure comprehensive coverage of the codebase:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **End-to-End Tests**: Test complete workflows
- **Smoke Tests**: Quick health checks for critical functionality
- **Fixtures**: Reusable test data and configurations

## Test Structure

```
tests/
├── __init__.py                           # Package initialization
├── conftest.py                           # Shared pytest fixtures
├── unit/                                 # Unit tests
│   ├── __init__.py
│   ├── test_basic_functionality.py      # Basic functionality tests
│   ├── test_config_validator.py         # Configuration validation tests
│   └── ... (other unit tests)
├── integration/                          # Integration tests
│   ├── __init__.py
│   ├── test_forecast_horizon_integration.py  # Forecast horizon tests
│   ├── test_pipeline_integration.py          # Pipeline integration tests
│   └── ... (other integration tests)
├── e2e/                                 # End-to-end tests
│   ├── __init__.py
│   ├── test_complete_benchmark_workflow.py   # Complete workflow tests
│   ├── test_forecast_horizon_e2e.py          # E2E forecast horizon tests
│   └── ... (other e2e tests)
├── smoke/                               # Smoke tests
│   ├── __init__.py
│   ├── test_forecast_horizon_smoke.py        # Quick forecast tests
│   ├── test_system_health.py                 # System health checks
│   └── ... (other smoke tests)
└── fixtures/                            # Test fixtures
    ├── __init__.py
    ├── test_data.py                     # Test data generation
    ├── test_simple_forecast.yaml         # Simple test configuration
    └── ... (other fixtures)
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Unit tests focus on testing individual components in isolation. They should:
- Test a single function or method
- Use mocks for external dependencies
- Be fast and deterministic
- Have clear, descriptive names

**Example Unit Test:**
```python
def test_arima_model_initialization():
    """Test ARIMA model initialization with valid parameters."""
    config = {
        'p': 1,
        'd': 1,
        'q': 1,
        's': 12,
        'forecast_horizon': 10
    }
    
    model = ArimaModel(config)
    assert model.p == 1
    assert model.d == 1
    assert model.q == 1
    assert model.s == 12
    assert model.forecast_horizon == 10
    assert not model.is_fitted
```

**Coverage Areas:**
- Model initialization and parameter validation
- Configuration parsing and validation
- Utility functions
- Data structure operations
- Metric calculations

### 2. Integration Tests (`tests/integration/`)

Integration tests verify that components work together correctly. They should:
- Test interactions between multiple components
- Use real data structures (not mocks)
- Test data flow through the pipeline
- Verify component interfaces

**Example Integration Test:**
```python
def test_data_loader_with_preprocessor():
    """Test that DataLoader and Preprocessor work together correctly."""
    config = load_test_config()
    
    # Load data
    data_loader = DataLoader(config)
    dataset = data_loader.load_single_chunk(1)
    
    # Preprocess data
    preprocessor = Preprocessor(config)
    preprocessed = preprocessor.preprocess(dataset)
    
    # Verify data flow
    assert preprocessed.data.train.targets is not None
    assert preprocessed.data.validation.targets is not None
    assert preprocessed.data.test.targets is not None
    assert len(preprocessed.preprocessing_info) > 0
```

**Coverage Areas:**
- Data loading and preprocessing pipeline
- Model training and evaluation workflow
- Configuration system integration
- Metric computation and logging
- File I/O operations

### 3. End-to-End Tests (`tests/e2e/`)

End-to-end tests verify complete workflows from start to finish. They should:
- Test the entire pipeline
- Use realistic data and configurations
- Verify end-to-end results
- Test error handling and edge cases

**Example E2E Test:**
```python
def test_complete_benchmark_workflow():
    """Test the complete benchmarking workflow with a simple configuration."""
    config = load_simple_test_config()
    
    # Run complete benchmark
    results = run_benchmark_pipeline(config)
    
    # Verify results structure
    assert 'models' in results
    assert 'datasets' in results
    assert 'metrics' in results
    
    # Verify specific results
    assert len(results['models']) > 0
    assert all('performance' in model for model in results['models'])
    assert all('hyperparameters' in model for model in results['models'])
```

**Coverage Areas:**
- Complete benchmark execution
- Result generation and storage
- Error handling and recovery
- Performance under realistic conditions
- Configuration validation end-to-end

### 4. Smoke Tests (`tests/smoke/`)

Smoke tests provide quick health checks for critical functionality. They should:
- Be very fast (seconds, not minutes)
- Test core functionality only
- Catch obvious failures quickly
- Be run frequently (e.g., on every commit)

**Example Smoke Test:**
```python
def test_system_health():
    """Quick health check for critical system components."""
    # Test imports
    from benchmarking_pipeline import model_router
    from benchmarking_pipeline.pipeline import DataLoader
    
    # Test basic functionality
    available_models = model_router.get_available_models()
    assert isinstance(available_models, dict)
    assert 'anyvariate' in available_models
    assert 'multivariate' in available_models
    assert 'univariate_only' in available_models
    
    # Test configuration loading
    config = load_minimal_test_config()
    assert config is not None
    assert 'dataset' in config
    assert 'model' in config
```

**Coverage Areas:**
- Critical imports and dependencies
- Basic configuration loading
- Core data structures
- Essential model routing
- System availability

### 5. Test Fixtures (`tests/fixtures/`)

Test fixtures provide reusable test data and configurations. They should:
- Be lightweight and fast to generate
- Cover common test scenarios
- Be deterministic and reproducible
- Support multiple test types

**Example Fixture:**
```python
@pytest.fixture
def simple_forecast_config():
    """Provide a simple configuration for forecasting tests."""
    return {
        'test_type': 'deterministic',
        'dataset': {
            'name': 'test_dataset',
            'path': 'tests/fixtures/test_data',
            'frequency': 'D',
            'forecast_horizon': 5,
            'split_ratio': [0.8, 0.1, 0.1],
            'chunks': 1
        },
        'model': {
            'arima': {
                'p': [1],
                'd': [1],
                'q': [1]
            }
        },
        'evaluation': {
            'type': 'deterministic',
            'metrics': ['mae', 'rmse']
        }
    }
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=benchmarking_pipeline

# Run tests and generate HTML coverage report
pytest --cov=benchmarking_pipeline --cov-report=html
```

### Running Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only end-to-end tests
pytest tests/e2e/

# Run only smoke tests
pytest tests/smoke/
```

### Running Specific Tests

```bash
# Run tests matching a pattern
pytest -k "test_arima"

# Run tests in a specific file
pytest tests/unit/test_basic_functionality.py

# Run a specific test function
pytest tests/unit/test_basic_functionality.py::test_arima_model_initialization
```

### Test Configuration

The test configuration is controlled by `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    smoke: Smoke tests
    slow: Slow running tests
```

## Test Data Management

### Test Data Sources

1. **Synthetic Data**: Generated programmatically for controlled testing
2. **Small Real Datasets**: Minimal versions of real datasets for realistic testing
3. **Mock Data**: Simulated data structures for unit testing

### Test Data Generation

```python
def generate_test_time_series(length=100, frequency='D'):
    """Generate synthetic time series data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=length, freq=frequency)
    values = np.random.randn(length).cumsum() + 100  # Random walk
    return pd.Series(values, index=dates)

def generate_test_dataset(num_chunks=3, chunk_length=100):
    """Generate a complete test dataset with multiple chunks."""
    datasets = []
    for i in range(num_chunks):
        data = generate_test_time_series(chunk_length)
        # Create chunk file structure
        # ... implementation details
        datasets.append(data)
    return datasets
```

### Test Data Cleanup

```python
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically clean up test files after each test."""
    yield
    # Cleanup code here
    cleanup_test_outputs()
    cleanup_test_logs()
```

## Best Practices

### 1. Test Organization

- **Group related tests** in the same test class or file
- **Use descriptive test names** that explain what is being tested
- **Follow the Arrange-Act-Assert pattern** for test structure
- **Keep tests independent** - no test should depend on another

### 2. Test Data

- **Use minimal test data** - only what's needed for the test
- **Make test data deterministic** - avoid random data when possible
- **Clean up test data** after tests complete
- **Use fixtures** for commonly used test data

### 3. Test Coverage

- **Aim for high coverage** but focus on critical paths
- **Test edge cases** and error conditions
- **Test both success and failure scenarios**
- **Cover configuration validation** thoroughly

### 4. Performance

- **Keep tests fast** - aim for seconds, not minutes
- **Use appropriate test categories** for different performance requirements
- **Mock expensive operations** in unit tests
- **Use parallel execution** when possible

### 5. Maintenance

- **Update tests** when code changes
- **Refactor tests** to reduce duplication
- **Document complex test scenarios**
- **Review test failures** carefully

## Continuous Integration

### GitHub Actions

The project includes GitHub Actions workflows for automated testing:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=benchmarking_pipeline --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Pre-commit Hooks

Consider using pre-commit hooks for local testing:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the package is installed in development mode
2. **Test Data Issues**: Check that test data files exist and are accessible
3. **Configuration Problems**: Verify test configuration files are valid
4. **Performance Issues**: Use appropriate test categories for different scenarios

### Debugging Tests

```bash
# Run tests with debug output
pytest -s

# Run tests with maximum verbosity
pytest -vvv

# Run tests and stop on first failure
pytest -x

# Run tests and show local variables on failure
pytest -l
```

### Test Isolation

```python
@pytest.fixture(autouse=True)
def isolate_test_environment():
    """Ensure each test runs in isolation."""
    # Setup
    original_env = os.environ.copy()
    
    yield
    
    # Teardown
    os.environ.clear()
    os.environ.update(original_env)
```

## Contributing

When adding new tests:

1. **Follow the existing structure** and naming conventions
2. **Add appropriate markers** for test categorization
3. **Include docstrings** explaining what is being tested
4. **Update this documentation** if adding new test categories
5. **Ensure tests pass** before submitting changes

## References

- [pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://realpython.com/python-testing/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)
- [Continuous Integration](https://en.wikipedia.org/wiki/Continuous_integration)
