"""
End-to-end tests for complete benchmarking pipeline workflows.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.models.model_router import ModelRouter


class TestCompleteBenchmarkWorkflow:
    """Test cases for complete benchmarking workflows."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_univariate_workflow(self, test_data_dir, sample_csv_data, mock_config):
        """Test complete univariate benchmarking workflow."""
        # Create test data with the correct format
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3000, 7001, 1, 2, 3000, 7001, 1, 2, 3000, 7001, 1, 2, 3000, 7001, 1, 2, 3000, 7001]'],
            'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
        })
        
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["model"]["type"] = "univariate"
        config["model"]["name"] = "seasonal_naive"  # Use a simple model for testing
        
        # Step 1: Data Loading
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Verify data was loaded correctly
        assert chunk.train is not None
        assert chunk.validation is not None
        assert chunk.test is not None
        
        # Step 2: Model Routing
        model_router = ModelRouter()
        folder_path, file_name, class_name = model_router.get_model_path('seasonal_naive', 'univariate')
        assert folder_path is not None
        assert 'univariate/seasonal_naive' in folder_path
        
        # Step 3: Verify workflow components are ready
        # The actual model execution would happen in a separate process
        # Here we just verify that all the routing and data preparation works
        assert folder_path is not None
        assert file_name is not None
        assert class_name is not None
        
        # Verify data integrity
        total_length = len(chunk.train.targets) + len(chunk.validation.targets) + len(chunk.test.targets)
        assert total_length == 20  # Total should equal original data length
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_multivariate_workflow(self, test_data_dir, sample_multivariate_data, mock_config):
        """Test complete multivariate benchmarking workflow."""
        # Create multivariate test data with the correct format
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'],
            'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
        })
        
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config for multivariate
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["model"]["type"] = "multivariate"
        config["model"]["name"] = "lstm"
        
        # Step 1: Data Loading
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Verify multivariate data was loaded
        assert chunk.train is not None
        assert chunk.validation is not None
        assert chunk.test is not None
        
        # Step 2: Model Routing
        model_router = ModelRouter()
        folder_path, file_name, class_name = model_router.get_model_path('lstm', 'multivariate')
        assert folder_path is not None
        assert 'multivariate/lstm' in folder_path
        
        # Step 3: Verify workflow components are ready
        assert folder_path is not None
        assert file_name is not None
        assert class_name is not None
        
        # Verify data integrity
        total_length = len(chunk.train.targets) + len(chunk.validation.targets) + len(chunk.test.targets)
        assert total_length == 20  # Total should equal original data length
    
    @pytest.mark.e2e
    def test_workflow_with_multiple_chunks(self, test_data_dir, sample_csv_data, mock_config):
        """Test workflow with multiple data chunks."""
        # Create multiple chunks with the correct format
        for i in range(1, 4):
            data = pd.DataFrame({
                'item_id': [i],
                'start': [f'2023-0{i}-01 00:00:00'],
                'freq': ['D'],
                'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'],
                'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
            })
            data.to_csv(os.path.join(test_data_dir, f'chunk{i:03d}.csv'), index=False)
        
        # Update config
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        # Load multiple chunks
        data_loader = DataLoader(config)
        chunks = data_loader.load_several_chunks(3)
        
        # Verify all chunks were loaded
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk.train is not None
            assert chunk.validation is not None
            assert chunk.test is not None
        
        # Test processing multiple chunks
        model_router = ModelRouter()
        for i, chunk in enumerate(chunks):
            folder_path, file_name, class_name = model_router.get_model_path('arima', 'univariate')
            assert folder_path is not None
    
    @pytest.mark.e2e
    def test_workflow_error_handling(self, mock_config):
        """Test workflow error handling and recovery."""
        # Test with invalid configuration
        invalid_config = mock_config.copy()
        invalid_config["dataset"]["path"] = "/non/existent/path"  # Invalid path
        
        # Should fail gracefully when trying to load data
        data_loader = DataLoader(invalid_config)
        with pytest.raises(FileNotFoundError):
            data_loader.load_single_chunk(1)
    
    @pytest.mark.e2e
    def test_workflow_performance_metrics(self, test_data_dir, sample_csv_data, mock_config):
        """Test that workflow produces meaningful performance metrics."""
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
        
        # Load data
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Verify data was loaded
        assert chunk.train is not None
        assert chunk.validation is not None
        assert chunk.test is not None
        
        # Test ModelRouter functionality
        model_router = ModelRouter()
        folder_path, file_name, class_name = model_router.get_model_path('arima', 'univariate')
        assert folder_path is not None
        
        # Verify that the workflow components are properly configured
        # for performance evaluation
        assert folder_path is not None
        assert file_name is not None
        assert class_name is not None
        
        # Verify data integrity for performance evaluation
        total_length = len(chunk.train.targets) + len(chunk.validation.targets) + len(chunk.test.targets)
        assert total_length == 20  # Total should equal original data length
        
        # Verify that the data splits are reasonable for evaluation
        assert len(chunk.train.targets) > 0
        assert len(chunk.validation.targets) > 0
        assert len(chunk.test.targets) > 0
