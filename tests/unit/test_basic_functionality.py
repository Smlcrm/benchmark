"""
Basic functionality tests to verify the test suite structure.
"""

import pytest


class TestBasicFunctionality:
    """Basic tests to verify the test suite is working."""
    
    def test_basic_assertion(self):
        """Test that basic assertions work."""
        assert True
        assert 1 + 1 == 2
        assert "hello" in "hello world"
    
    def test_fixture_usage(self, sample_csv_data):
        """Test that fixtures work correctly."""
        assert sample_csv_data is not None
        assert len(sample_csv_data) > 0
        assert 'timestamp' in sample_csv_data.columns
        assert 'y' in sample_csv_data.columns
    
    def test_mock_config(self, mock_config):
        """Test that mock config fixture works."""
        assert mock_config is not None
        assert 'dataset' in mock_config
        assert 'model' in mock_config
        assert mock_config['model']['type'] == 'univariate'
    
    @pytest.mark.parametrize("input_value,expected", [
        (1, 1),
        (2, 2),
        (3, 3),
        (0, 0),
        (-1, -1)
    ])
    def test_parametrized_test(self, input_value, expected):
        """Test that parametrized tests work."""
        assert input_value == expected
    
    def test_marker_functionality(self):
        """Test that test markers work."""
        # This test should be marked as unit
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        # This test is marked as slow
        assert True
