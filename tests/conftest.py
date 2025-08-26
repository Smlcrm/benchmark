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
            "split_ratio": [0.8, 0.1, 0.1]
        },
        "model": {
            "name": "test_model",
            "type": "univariate"
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
