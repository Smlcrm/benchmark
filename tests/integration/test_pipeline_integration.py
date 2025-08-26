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
    
    @pytest.mark.integration
    def test_data_loader_with_model_router(self, test_data_dir, sample_csv_data, mock_config):
        """Test integration between DataLoader and ModelRouter."""
        # Create test data with the correct format
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3000, 7001, 1, 2, 3000, 7001, 1, 2, 3000, 7001, 1, 2, 3000, 7001, 1, 2, 3000, 7001]'],
            'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
        })
        
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config with test directory
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        # Initialize components
        data_loader = DataLoader(config)
        model_router = ModelRouter()  # No config needed
        
        # Test that they work together
        chunk = data_loader.load_single_chunk(1)
        
        # Verify data was loaded correctly
        assert chunk.train is not None
        assert chunk.validation is not None
        assert chunk.test is not None
        
        # Test ModelRouter functionality
        folder_path, file_name, class_name = model_router.get_model_path('arima', 'univariate')
        assert folder_path is not None
        assert file_name is not None
        assert class_name is not None
        assert 'univariate/arima' in folder_path
    
    @pytest.mark.integration
    def test_multivariate_pipeline_integration(self, test_data_dir, sample_multivariate_data, mock_config):
        """Test integration with multivariate data."""
        # Update config for multivariate
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["model"]["type"] = "multivariate"
        
        # Create multivariate test data with the correct format
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'],
            'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
        })
        
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Initialize components
        data_loader = DataLoader(config)
        model_router = ModelRouter()
        
        # Test multivariate data loading
        chunk = data_loader.load_single_chunk(1)
        
        # Verify multivariate data was loaded
        assert chunk.train is not None
        assert chunk.validation is not None
        assert chunk.test is not None
        
        # Test ModelRouter with multivariate model
        folder_path, file_name, class_name = model_router.get_model_path('arima', 'multivariate')
        assert folder_path is not None
        assert 'multivariate/arima' in folder_path
    
    @pytest.mark.integration
    def test_config_consistency_across_components(self, mock_config):
        """Test that configuration is consistent across components."""
        # Test that both components can work with the same config
        data_loader = DataLoader(mock_config)
        model_router = ModelRouter()
        
        # Verify config is properly used
        assert data_loader.config == mock_config
        assert data_loader.config["dataset"]["name"] == "test_dataset"
        assert data_loader.config["model"]["type"] == "univariate"
        
        # Verify ModelRouter has expected model categories
        assert hasattr(model_router, 'univariate_models')
        assert hasattr(model_router, 'multivariate_models')
        assert hasattr(model_router, 'anyvariate_models')
    
    @pytest.mark.parametrize("split_ratio", [
        [0.7, 0.2, 0.1],
        [0.6, 0.2, 0.2],
        [0.8, 0.1, 0.1]
    ])
    @pytest.mark.integration
    def test_different_split_ratios_integration(self, test_data_dir, sample_csv_data, mock_config, split_ratio):
        """Test integration with different data split ratios."""
        # Create test data with the correct format
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'],
            'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
        })
        
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["dataset"]["split_ratio"] = split_ratio
        
        # Initialize components
        data_loader = DataLoader(config)
        model_router = ModelRouter()
        
        # Test data loading
        chunk = data_loader.load_single_chunk(1)
        
        # Verify split ratios are applied correctly
        total_length = 20  # Length of target data
        expected_train_length = int(total_length * split_ratio[0])
        expected_val_length = int(total_length * split_ratio[1])
        expected_test_length = int(total_length * split_ratio[2])
        
        assert len(chunk.train.targets) == expected_train_length
        assert len(chunk.validation.targets) == expected_val_length
        assert len(chunk.test.targets) == expected_test_length
        
        # Verify ModelRouter still works with different split ratios
        folder_path, file_name, class_name = model_router.get_model_path('arima', 'univariate')
        assert folder_path is not None
    
    @pytest.mark.integration
    def test_error_propagation_between_components(self, mock_config):
        """Test that errors in one component properly propagate to others."""
        # Test with invalid config that should cause errors in DataLoader
        invalid_config = mock_config.copy()
        invalid_config["dataset"]["path"] = "/non/existent/path"  # Invalid path
        
        # DataLoader should fail with invalid path when trying to load data
        data_loader = DataLoader(invalid_config)
        with pytest.raises(FileNotFoundError):
            data_loader.load_single_chunk(1)
    
    @pytest.mark.integration
    def test_data_flow_consistency(self, test_data_dir, sample_csv_data, mock_config):
        """Test that data flows consistently through the pipeline."""
        # Create test data with the correct format
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'],
            'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
        })
        
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        # Initialize components
        data_loader = DataLoader(config)
        model_router = ModelRouter()
        
        # Load data
        chunk = data_loader.load_single_chunk(1)
        
        # Verify data consistency
        assert chunk.train is not None
        assert chunk.validation is not None
        assert chunk.test is not None
        
        # Verify that the same data can be processed by ModelRouter
        folder_path, file_name, class_name = model_router.get_model_path('arima', 'univariate')
        assert folder_path is not None
        
        # Verify data integrity
        total_loaded = len(chunk.train.targets) + len(chunk.validation.targets) + len(chunk.test.targets)
        assert total_loaded == 20  # Total should equal original data length
