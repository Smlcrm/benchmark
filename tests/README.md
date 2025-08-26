# Benchmarking Pipeline Test Suite

This directory contains the comprehensive test suite for the benchmarking pipeline, organized following Python testing best practices.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── unit/                    # Unit tests for individual components
│   ├── test_data_loader.py
│   ├── test_evaluator.py
│   └── test_model_configs.py
├── integration/             # Integration tests for component interactions
│   └── test_pipeline_integration.py
├── e2e/                    # End-to-end tests for complete workflows
│   └── test_complete_benchmark_workflow.py
├── fixtures/                # Test data and utilities
│   └── test_data.py
└── README.md               # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual functions, methods, and classes in isolation
- **Scope**: Single component functionality
- **Speed**: Fast execution
- **Dependencies**: Minimal external dependencies, mostly mocked

### Integration Tests (`tests/integration/`)
- **Purpose**: Test how different components work together
- **Scope**: Component interactions and data flow
- **Speed**: Medium execution time
- **Dependencies**: Multiple components, some real data

### End-to-End Tests (`tests/e2e/`)
- **Purpose**: Test complete workflows from start to finish
- **Scope**: Full pipeline execution
- **Speed**: Slower execution (marked with `@pytest.mark.slow`)
- **Dependencies**: Full system, real data processing

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-cov pytest-xdist
```

### Basic Test Execution
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=benchmarking_pipeline --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_data_loader.py

# Run specific test function
python -m pytest tests/unit/test_data_loader.py::TestDataLoader::test_load_single_chunk
```

### Using the Test Runner Script
```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --type unit

# Run with coverage
python run_tests.py --coverage

# Skip slow tests
python run_tests.py --fast

# Run in parallel
python run_tests.py --parallel
```

### Test Markers
```bash
# Run only unit tests
python -m pytest -m unit

# Run only integration tests
python -m pytest -m integration

# Run only end-to-end tests
python -m pytest -m e2e

# Skip slow tests
python -m pytest -m "not slow"

# Run smoke tests only
python -m pytest -m smoke
```

## Test Configuration

### pytest.ini
- Configures test discovery patterns
- Sets up test markers
- Configures coverage reporting
- Sets warning filters

### conftest.py
- Provides shared fixtures across all tests
- Configures pytest hooks
- Sets up test data and utilities

## Test Fixtures

### Available Fixtures
- `test_data_dir`: Temporary directory for test data
- `sample_csv_data`: Sample univariate time series data
- `sample_multivariate_data`: Sample multivariate time series data
- `mock_config`: Mock configuration for testing
- `beijing_dataset_sample`: Sample Beijing dataset structure

### Custom Test Data
The `tests/fixtures/test_data.py` module provides functions to create various types of test data:
- Univariate time series with trend and seasonality
- Multivariate time series with multiple targets and features
- Data with missing values
- Irregular timestamp data
- Long-horizon forecasting data
- Categorical and external regressor data
- Anomaly data

## Writing Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Test Structure
```python
class TestComponentName:
    """Test cases for ComponentName functionality."""
    
    def test_specific_functionality(self, fixture_name):
        """Test description."""
        # Arrange
        # Act
        # Assert
```

### Using Fixtures
```python
def test_example(self, sample_csv_data, mock_config):
    """Example test using fixtures."""
    # Use the fixtures
    data = sample_csv_data
    config = mock_config
    
    # Test logic here
    assert len(data) > 0
```

### Test Markers
```python
@pytest.mark.unit
def test_unit_functionality(self):
    """Unit test."""
    pass

@pytest.mark.integration
def test_integration_functionality(self):
    """Integration test."""
    pass

@pytest.mark.e2e
@pytest.mark.slow
def test_end_to_end_workflow(self):
    """End-to-end test."""
    pass
```

## Coverage Requirements

- **Minimum Coverage**: 80%
- **Coverage Reports**: HTML, XML, and terminal output
- **Coverage Location**: `htmlcov/` directory

## Best Practices

1. **Test Isolation**: Each test should be independent and not affect others
2. **Descriptive Names**: Test names should clearly describe what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Fixture Usage**: Use fixtures for common setup and teardown
5. **Mocking**: Mock external dependencies to isolate the unit under test
6. **Edge Cases**: Test boundary conditions and error scenarios
7. **Performance**: Mark slow tests appropriately

## Debugging Tests

### Verbose Output
```bash
python -m pytest -v
```

### Stop on First Failure
```bash
python -m pytest -x
```

### Debug Mode
```bash
python -m pytest --pdb
```

### Show Local Variables
```bash
python -m pytest -l
```

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:
- Fast execution for unit tests
- Comprehensive coverage reporting
- Clear test categorization
- Parallel execution support

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure the benchmarking_pipeline package is installed or in PYTHONPATH
2. **Fixture Errors**: Check that fixtures are properly defined in conftest.py
3. **Coverage Issues**: Verify that the coverage package is installed
4. **Slow Tests**: Use `--fast` flag to skip slow tests during development

### Getting Help
- Check pytest documentation: https://docs.pytest.org/
- Review existing test examples in the codebase
- Use `python -m pytest --help` for command-line options
