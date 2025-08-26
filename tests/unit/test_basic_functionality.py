"""
Basic functionality tests to verify pytest setup.
"""

import pytest
import numpy as np


class TestBasicFunctionality:
    """Test basic pytest functionality."""
    
    @pytest.mark.unit
    def test_basic_assertion(self):
        """Test that basic assertions work."""
        assert 1 + 1 == 2
        assert "hello" + " world" == "hello world"
        assert [1, 2, 3] == [1, 2, 3]
    
    @pytest.mark.unit
    def test_fixture_usage(self, mock_config):
        """Test that fixtures are working."""
        assert mock_config is not None
        assert "dataset" in mock_config
        assert "model" in mock_config
        assert "training" in mock_config
    
    @pytest.mark.unit
    def test_mock_config(self, mock_config):
        """Test the mock_config fixture specifically."""
        assert mock_config["dataset"]["name"] == "test_dataset"
        assert mock_config["model"]["type"] == "univariate"
        assert mock_config["training"]["epochs"] == 10
    
    @pytest.mark.parametrize("input_val,expected", [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (-1, -1)
    ])
    @pytest.mark.unit
    def test_parametrized_test(self, input_val, expected):
        """Test parametrized tests."""
        assert input_val == expected
    
    @pytest.mark.unit
    def test_marker_functionality(self):
        """Test that pytest markers work."""
        # This test should be marked as unit
        assert True
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        # This test should be marked as both unit and slow
        assert True
