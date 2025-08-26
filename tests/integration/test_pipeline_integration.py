"""
Integration tests for pipeline component interactions.
"""

import pytest
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import patch

from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.models.model_router import ModelRouter


class TestPipelineIntegration:
    """Test cases for pipeline component interactions."""
    
    def test_data_loader_with_model_router(self, test_data_dir, sample_csv_data, mock_config):
        """Test integration between DataLoader and ModelRouter."""
        # Create test data
        sample_csv_data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config with test directory
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        # Initialize components
        data_loader = DataLoader(config)
        model_router = ModelRouter(config)
        
        # Test that they work together
        chunk = data_loader.load_single_chunk(1)
        
        # Verify data structure is compatible with model requirements
        assert hasattr(chunk.train, 'targets')
        assert hasattr(chunk.validation, 'targets')
        assert hasattr(chunk.test, 'targets')
        
        # Verify model router can handle the config
        assert model_router.model_name == config["model"]["name"]
        assert model_router.model_type == config["model"]["type"]
    
    def test_multivariate_pipeline_integration(self, test_data_dir, sample_multivariate_data, mock_config):
        """Test integration with multivariate data."""
        # Update config for multivariate
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["model"]["type"] = "multivariate"
        
        # Create multivariate test data
        sample_multivariate_data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Initialize components
        data_loader = DataLoader(config)
        model_router = ModelRouter(config)
        
        # Test multivariate data loading
        chunk = data_loader.load_single_chunk(1)
        
        # Verify multivariate structure
        assert 'target_1' in chunk.train.targets
        assert 'target_2' in chunk.train.targets
        assert 'feature_1' in chunk.train.features
        assert 'feature_2' in chunk.train.features
        
        # Verify model router handles multivariate type
        assert model_router.model_type == "multivariate"
    
    def test_config_consistency_across_components(self, mock_config):
        """Test that configuration is consistent across all components."""
        # Test that all components can be initialized with the same config
        data_loader = DataLoader(mock_config)
        model_router = ModelRouter(mock_config)
        
        # Verify they use the same configuration
        assert data_loader.config == model_router.config
        
        # Verify they interpret the config consistently
        assert data_loader.config["dataset"]["name"] == model_router.config["dataset"]["name"]
        assert data_loader.config["model"]["type"] == model_router.config["model"]["type"]
    
    @pytest.mark.parametrize("split_ratio", [
        [0.7, 0.2, 0.1],
        [0.6, 0.2, 0.2],
        [0.8, 0.1, 0.1]
    ])
    def test_different_split_ratios_integration(self, test_data_dir, sample_csv_data, mock_config, split_ratio):
        """Test integration with different data split ratios."""
        # Create test data
        sample_csv_data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["dataset"]["split_ratio"] = split_ratio
        
        # Initialize components
        data_loader = DataLoader(config)
        model_router = ModelRouter(config)
        
        # Test data loading
        chunk = data_loader.load_single_chunk(1)
        
        # Verify split ratios are applied correctly
        total_length = len(sample_csv_data)
        expected_train_length = int(total_length * split_ratio[0])
        expected_val_length = int(total_length * split_ratio[1])
        expected_test_length = int(total_length * split_ratio[2])
        
        assert len(chunk.train.targets["y"]) == expected_train_length
        assert len(chunk.validation.targets["y"]) == expected_val_length
        assert len(chunk.test.targets["y"]) == expected_test_length
        
        # Verify model router still works with modified config
        assert model_router.config["dataset"]["split_ratio"] == split_ratio
    
    def test_error_propagation_between_components(self, mock_config):
        """Test that errors in one component properly propagate to others."""
        # Test with invalid config that should cause errors in multiple components
        invalid_config = mock_config.copy()
        invalid_config["dataset"]["split_ratio"] = [0.5, 0.5]  # Invalid: doesn't sum to 1
        
        # Both components should fail with invalid config
        with pytest.raises(ValueError):
            DataLoader(invalid_config)
        
        with pytest.raises(ValueError):
            ModelRouter(invalid_config)
    
    def test_data_flow_consistency(self, test_data_dir, sample_csv_data, mock_config):
        """Test that data flows consistently through the pipeline."""
        # Create test data
        sample_csv_data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        # Initialize components
        data_loader = DataLoader(config)
        model_router = ModelRouter(config)
        
        # Load data
        chunk = data_loader.load_single_chunk(1)
        
        # Verify data consistency
        train_data = chunk.train.targets["y"]
        val_data = chunk.validation.targets["y"]
        test_data = chunk.test.targets["y"]
        
        # All data should come from the same source
        assert len(train_data) + len(val_data) + len(test_data) == len(sample_csv_data)
        
        # Data should be continuous (no gaps in indices)
        all_indices = list(train_data.index) + list(val_data.index) + list(test_data.index)
        all_indices.sort()
        assert all_indices == list(range(len(sample_csv_data)))
