"""
Pytest configuration and shared fixtures for the benchmarking pipeline tests.
"""

import pytest
import tempfile
import os
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data that persists across the test session."""
    test_dir = tempfile.mkdtemp()
    
    # Create mock metadata.json for univariate testing
    univariate_metadata = {
        "variables": [
            {"var_name": "y", "target_index": 0}
        ]
    }
    
    # Create mock metadata.json for multivariate testing
    multivariate_metadata = {
        "variables": [
            {"var_name": "y", "target_index": 0},
            {"var_name": "z", "target_index": 1}
        ]
    }
    
    import json
    
    # Create univariate metadata
    univ_metadata_path = os.path.join(test_dir, 'univariate_metadata.json')
    with open(univ_metadata_path, 'w') as f:
        json.dump(univariate_metadata, f)
    
    # Create multivariate metadata
    mult_metadata_path = os.path.join(test_dir, 'multivariate_metadata.json')
    with open(mult_metadata_path, 'w') as f:
        json.dump(multivariate_metadata, f)
    
    # Create default metadata.json (for backward compatibility)
    default_metadata_path = os.path.join(test_dir, 'metadata.json')
    with open(default_metadata_path, 'w') as f:
        json.dump(univariate_metadata, f)
    
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
        'y': [1, 2, 3000, 7001] * 5
    })
    return data


@pytest.fixture
def sample_multivariate_data():
    """Create sample multivariate CSV data for testing."""
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
        'target_1': [1, 2, 3, 4] * 5,
        'target_2': [10, 20, 30, 40] * 5,
        'feature_1': [100, 200, 300, 400] * 5,
        'feature_2': [1000, 2000, 3000, 4000] * 5
    })
    return data


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    return {
        "dataset": {
            "path": "/tmp/test_dataset",
            "name": "test_dataset",
            "split_ratio": [0.8, 0.1, 0.1],
            "target_cols": ["y"],
            "forecast_horizon": [10, 25, 50],
            "frequency": "D",
            "normalize": False,
            "handle_missing": "interpolate",
            "chunks": 1
        },
        "model": {
            "test_model": {}  # New structure: models directly under model:
        },
        "training": {
            "epochs": 10,
            "batch_size": 32
        }
    }


@pytest.fixture
def beijing_dataset_sample():
    """Provide a sample of Beijing dataset for testing."""
    # This would contain actual Beijing dataset structure
    # For now, creating a mock structure
    return {
        'target': [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
            None,  # Some targets might be None
            [100.0, 200.0, 300.0, 400.0, 500.0]
        ]
    }


@pytest.fixture
def create_test_dataset():
    """Helper fixture to create test datasets with proper metadata."""
    import json
    
    def _create_dataset(test_dir, target_cols, is_multivariate=False):
        """Create a test dataset with proper metadata."""
        # Create metadata based on target columns
        variables = []
        for i, col_name in enumerate(target_cols):
            variables.append({"var_name": col_name, "target_index": i})
        
        metadata = {"variables": variables}
        
        # Create metadata.json in the test directory
        metadata_path = os.path.join(test_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return test_dir
    
    return _create_dataset


@pytest.fixture
def mock_multivariate_config():
    """Provide a mock multivariate configuration for testing."""
    return {
        "dataset": {
            "path": "/tmp/test_multivariate_dataset",
            "name": "test_multivariate_dataset",
            "split_ratio": [0.8, 0.1, 0.1],
            "target_cols": ["y", "z"],
            "forecast_horizon": [10, 25, 50],
            "frequency": "D",
            "normalize": False,
            "handle_missing": "interpolate",
            "chunks": 1
        },
        "model": {
            "lstm": {
                "hidden_size": [64, 128],
                "num_layers": [1, 2]
            }
        },
        "training": {
            "epochs": 10,
            "batch_size": 32
        }
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests for basic system health"
    )
