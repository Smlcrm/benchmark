"""
Unit tests for model configuration functionality.
"""

import pytest
import yaml
import os
from unittest.mock import patch, mock_open

from benchmarking_pipeline.models.model_router import ModelRouter


class TestModelConfigs:
    """Test cases for model configuration functionality."""
    
    @pytest.mark.unit
    def test_model_router_initialization(self):
        """Test ModelRouter initialization."""
        router = ModelRouter()
        assert router is not None
        assert hasattr(router, 'anyvariate_models')
        assert hasattr(router, 'multivariate_models')
        assert hasattr(router, 'univariate_models')
    
    @pytest.mark.unit
    def test_model_router_model_categories(self):
        """Test that ModelRouter has the expected model categories."""
        router = ModelRouter()
        
        # Check that expected model categories exist
        assert isinstance(router.anyvariate_models, set)
        assert isinstance(router.multivariate_models, set)
        assert isinstance(router.univariate_models, set)
        
        # Check that some expected models are present
        assert 'chronos' in router.anyvariate_models
        assert 'arima' in router.multivariate_models
        assert 'prophet' in router.univariate_models
    
    @pytest.mark.unit
    def test_parse_model_spec(self):
        """Test parsing of model specifications."""
        router = ModelRouter()
        
        # Test with model name
        model_name = router.parse_model_spec('arima')
        assert model_name == 'arima'
        
        # Test with just model name
        model_name = router.parse_model_spec('prophet')
        assert model_name == 'prophet'
        
        # Test with whitespace
        model_name = router.parse_model_spec(' arima ')
        assert model_name == 'arima'
    
    @pytest.mark.unit
    def test_get_model_path(self):
        """Test getting model paths."""
        router = ModelRouter()
        
        # Test univariate model
        folder_path, file_name, class_name = router.get_model_path('arima', {'target_cols': ['y']})
        assert 'univariate/arima' in folder_path
        assert file_name == 'arima_model'
        assert class_name == 'ArimaModel'
        
        # Test multivariate model
        folder_path, file_name, class_name = router.get_model_path('arima', {'target_cols': ['y', 'z']})
        assert 'multivariate/arima' in folder_path
        assert file_name == 'arima_model'
        assert class_name == 'ArimaModel'
        
        # Test anyvariate model
        folder_path, file_name, class_name = router.get_model_path('chronos', {'target_cols': ['y']})
        assert 'anyvariate/chronos' in folder_path
        assert file_name == 'chronos_model'
        assert class_name == 'ChronosModel'
    

    
    @pytest.mark.unit
    def test_get_model_info(self):
        """Test getting model information."""
        router = ModelRouter()
        
        # Test with valid model
        model_info = router.get_model_info('arima')
        assert model_info is not None
        assert 'folder_paths' in model_info
        assert 'name' in model_info
        assert 'category' in model_info
        
        # Test with anyvariate model
        model_info = router.get_model_info('chronos')
        assert model_info is not None
        assert 'anyvariate/chronos' in model_info['folder_paths']['anyvariate']
    
    @pytest.mark.unit
    def test_get_model_path_with_auto_detection(self):
        """Test getting model path with automatic variant detection."""
        router = ModelRouter()
        
        # Test auto-detection for anyvariate model
        folder_path, file_name, class_name = router.get_model_path_with_auto_detection('chronos', {'target_cols': ['y']})
        assert 'anyvariate/chronos' in folder_path
        assert file_name == 'chronos_model'
        assert class_name == 'ChronosModel'
    
    @pytest.mark.unit
    def test_invalid_model_names(self):
        """Test handling of invalid model names."""
        router = ModelRouter()
        
        # Test with non-existent model
        with pytest.raises(ValueError):
            router.get_model_path('non_existent_model', {'target_cols': ['y']})
        
        # Test with missing target_cols
        with pytest.raises(ValueError):
            router.get_model_path('arima', {})  # Missing target_cols
    
    @pytest.mark.unit
    def test_global_router_instance(self):
        """Test that global router instance works correctly."""
        # Test that we can create multiple instances
        router1 = ModelRouter()
        router2 = ModelRouter()
        
        assert router1 is not router2  # Should be different instances
        assert router1.anyvariate_models == router2.anyvariate_models  # But same configuration
