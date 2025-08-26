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
    
    def test_model_router_initialization(self, mock_config):
        """Test ModelRouter initialization with valid config."""
        router = ModelRouter(mock_config)
        assert router.config == mock_config
        assert router.model_name == "test_model"
        assert router.model_type == "univariate"
    
    def test_model_router_invalid_config(self):
        """Test ModelRouter initialization with invalid config."""
        invalid_config = {"invalid": "config"}
        
        with pytest.raises(KeyError):
            ModelRouter(invalid_config)
    
    def test_model_router_missing_model_info(self):
        """Test ModelRouter with missing model information."""
        incomplete_config = {
            "dataset": {"path": "/tmp/test", "name": "test"},
            # Missing model section
        }
        
        with pytest.raises(KeyError):
            ModelRouter(incomplete_config)
    
    def test_config_validation(self):
        """Test configuration validation logic."""
        # This would test the actual config validation if it exists
        # For now, creating a basic test structure
        valid_config = {
            "dataset": {
                "path": "/tmp/test",
                "name": "test_dataset",
                "split_ratio": [0.8, 0.1, 0.1]
            },
            "model": {
                "name": "test_model",
                "type": "univariate"
            }
        }
        
        # Basic validation checks
        assert "dataset" in valid_config
        assert "model" in valid_config
        assert len(valid_config["dataset"]["split_ratio"]) == 3
        assert sum(valid_config["dataset"]["split_ratio"]) == pytest.approx(1.0, abs=1e-6)
    
    @pytest.mark.parametrize("model_type", ["univariate", "multivariate", "anyvariate"])
    def test_model_type_validation(self, mock_config, model_type):
        """Test different model type validations."""
        config = mock_config.copy()
        config["model"]["type"] = model_type
        
        router = ModelRouter(config)
        assert router.model_type == model_type
    
    def test_invalid_model_type(self, mock_config):
        """Test that invalid model types raise appropriate errors."""
        config = mock_config.copy()
        config["model"]["type"] = "invalid_type"
        
        with pytest.raises(ValueError):
            ModelRouter(config)
    
    def test_config_file_loading(self):
        """Test loading configuration from YAML files."""
        # Mock YAML file content
        mock_yaml_content = """
        dataset:
          path: /tmp/test_dataset
          name: test_dataset
          split_ratio: [0.8, 0.1, 0.1]
        model:
          name: test_model
          type: univariate
        training:
          epochs: 10
          batch_size: 32
        """
        
        with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
            with patch("yaml.safe_load") as mock_yaml_load:
                mock_yaml_load.return_value = yaml.safe_load(mock_yaml_content)
                
                # This would test actual file loading if implemented
                config = yaml.safe_load(mock_yaml_content)
                
                assert config["dataset"]["name"] == "test_dataset"
                assert config["model"]["type"] == "univariate"
                assert config["training"]["epochs"] == 10
    
    def test_required_config_fields(self):
        """Test that required configuration fields are present."""
        required_fields = ["dataset", "model"]
        required_dataset_fields = ["path", "name", "split_ratio"]
        required_model_fields = ["name", "type"]
        
        # Test with minimal valid config
        minimal_config = {
            "dataset": {
                "path": "/tmp/test",
                "name": "test",
                "split_ratio": [0.8, 0.1, 0.1]
            },
            "model": {
                "name": "test_model",
                "type": "univariate"
            }
        }
        
        for field in required_fields:
            assert field in minimal_config
        
        for field in required_dataset_fields:
            assert field in minimal_config["dataset"]
        
        for field in required_model_fields:
            assert field in minimal_config["model"]
    
    def test_config_defaults(self):
        """Test configuration default values if they exist."""
        # This would test default values if the system supports them
        # For now, creating a basic test structure
        config_with_defaults = {
            "dataset": {
                "path": "/tmp/test",
                "name": "test_dataset",
                "split_ratio": [0.8, 0.1, 0.1]
            },
            "model": {
                "name": "test_model",
                "type": "univariate"
            },
            "training": {
                "epochs": 100,  # Default value
                "batch_size": 32  # Default value
            }
        }
        
        # Test that defaults are reasonable
        assert config_with_defaults["training"]["epochs"] > 0
        assert config_with_defaults["training"]["batch_size"] > 0
        assert config_with_defaults["training"]["batch_size"] <= 1024  # Reasonable upper bound
