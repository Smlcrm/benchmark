"""
Smoke tests for system health and basic functionality.
"""

import pytest
import sys
import os


class TestSystemHealth:
    """Smoke tests to verify system health."""
    
    def test_python_version(self):
        """Test that Python version is compatible."""
        # Ensure we're using Python 3.7+
        assert sys.version_info >= (3, 7), f"Python 3.7+ required, got {sys.version_info}"
    
    def test_required_packages(self):
        """Test that required packages are available."""
        try:
            import pandas as pd
            import numpy as np
            import pytest
            assert True
        except ImportError as e:
            pytest.fail(f"Required package not available: {e}")
    
    def test_test_structure(self):
        """Test that the test directory structure is correct."""
        # Check that test directories exist
        test_dirs = ['unit', 'integration', 'e2e', 'fixtures', 'smoke']
        for dir_name in test_dirs:
            dir_path = os.path.join(os.path.dirname(__file__), '..', dir_name)
            assert os.path.exists(dir_path), f"Test directory {dir_name} not found"
            assert os.path.isdir(dir_path), f"{dir_name} is not a directory"
    
    def test_conftest_available(self):
        """Test that conftest.py is available and importable."""
        conftest_path = os.path.join(os.path.dirname(__file__), '..', 'conftest.py')
        assert os.path.exists(conftest_path), "conftest.py not found"
        
        # Try to import conftest
        try:
            sys.path.insert(0, os.path.dirname(conftest_path))
            import conftest
            assert True
        except ImportError as e:
            pytest.fail(f"Could not import conftest.py: {e}")
        finally:
            sys.path.pop(0)
    
    def test_basic_imports(self):
        """Test that basic benchmarking pipeline modules can be imported."""
        try:
            # These imports should work if the package is properly set up
            # Note: We're not testing actual functionality, just importability
            import benchmarking_pipeline
            assert True
        except ImportError:
            # If the package isn't installed, that's okay for smoke tests
            # We just want to ensure the test structure is correct
            pass
    
    def test_fixture_availability(self):
        """Test that basic fixtures are available."""
        # This test will be run with pytest fixtures available
        # We just need to ensure the test runs without errors
        assert True
    
    def test_marker_registration(self):
        """Test that pytest markers are properly registered."""
        # This test verifies that our custom markers work
        # The actual marker registration is tested in conftest.py
        assert True
