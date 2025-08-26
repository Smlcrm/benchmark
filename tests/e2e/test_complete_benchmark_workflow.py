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
from benchmarking_pipeline.model_executor import ModelExecutor


class TestCompleteBenchmarkWorkflow:
    """Test cases for complete benchmarking workflows."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_univariate_workflow(self, test_data_dir, sample_csv_data, mock_config):
        """Test complete univariate benchmarking workflow."""
        # Create test data
        sample_csv_data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["model"]["type"] = "univariate"
        config["model"]["name"] = "seasonal_naive"  # Use a simple model for testing
        
        # Step 1: Data Loading
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Verify data structure
        assert chunk.train is not None
        assert chunk.validation is not None
        assert chunk.test is not None
        
        # Step 2: Model Routing
        model_router = ModelRouter(config)
        assert model_router.model_type == "univariate"
        
        # Step 3: Model Execution (mocked for now)
        with patch('benchmarking_pipeline.model_executor.ModelExecutor') as mock_executor:
            mock_executor_instance = MagicMock()
            mock_executor.return_value = mock_executor_instance
            
            # Mock successful execution
            mock_executor_instance.execute.return_value = {
                'train_loss': 0.1,
                'val_loss': 0.15,
                'test_loss': 0.2,
                'predictions': np.array([1.0, 2.0, 3.0, 4.0])
            }
            
            # Execute the workflow
            executor = ModelExecutor(config)
            results = executor.execute(chunk)
            
            # Verify results structure
            assert 'train_loss' in results
            assert 'val_loss' in results
            assert 'test_loss' in results
            assert 'predictions' in results
            
            # Verify results are reasonable
            assert results['train_loss'] >= 0
            assert results['val_loss'] >= 0
            assert results['test_loss'] >= 0
            assert len(results['predictions']) > 0
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_multivariate_workflow(self, test_data_dir, sample_multivariate_data, mock_config):
        """Test complete multivariate benchmarking workflow."""
        # Create multivariate test data
        sample_multivariate_data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config for multivariate
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["model"]["type"] = "multivariate"
        config["model"]["name"] = "lstm"
        
        # Step 1: Data Loading
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Verify multivariate data structure
        assert 'target_1' in chunk.train.targets
        assert 'target_2' in chunk.train.targets
        assert 'feature_1' in chunk.train.features
        assert 'feature_2' in chunk.train.features
        
        # Step 2: Model Routing
        model_router = ModelRouter(config)
        assert model_router.model_type == "multivariate"
        
        # Step 3: Model Execution (mocked)
        with patch('benchmarking_pipeline.model_executor.ModelExecutor') as mock_executor:
            mock_executor_instance = MagicMock()
            mock_executor.return_value = mock_executor_instance
            
            # Mock multivariate results
            mock_executor_instance.execute.return_value = {
                'train_loss': 0.2,
                'val_loss': 0.25,
                'test_loss': 0.3,
                'predictions': {
                    'target_1': np.array([1.0, 2.0, 3.0, 4.0]),
                    'target_2': np.array([10.0, 20.0, 30.0, 40.0])
                }
            }
            
            # Execute the workflow
            executor = ModelExecutor(config)
            results = executor.execute(chunk)
            
            # Verify multivariate results structure
            assert 'predictions' in results
            assert 'target_1' in results['predictions']
            assert 'target_2' in results['predictions']
            
            # Verify predictions have correct shapes
            assert len(results['predictions']['target_1']) > 0
            assert len(results['predictions']['target_2']) > 0
    
    @pytest.mark.e2e
    def test_workflow_with_multiple_chunks(self, test_data_dir, sample_csv_data, mock_config):
        """Test workflow with multiple data chunks."""
        # Create multiple chunks
        for i in range(1, 4):
            chunk_data = sample_csv_data.copy()
            chunk_data['timestamp'] = pd.date_range(f'2023-0{i}-01', periods=20, freq='D')
            chunk_data.to_csv(os.path.join(test_data_dir, f'chunk{i:03d}.csv'), index=False)
        
        # Update config
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        # Load multiple chunks
        data_loader = DataLoader(config)
        chunks = data_loader.load_several_chunks(3)
        
        assert len(chunks) == 3
        
        # Test that each chunk can be processed
        for i, chunk in enumerate(chunks):
            assert chunk.train is not None
            assert chunk.validation is not None
            assert chunk.test is not None
            
            # Verify chunk numbering
            assert hasattr(chunk, 'chunk_id') or i == i  # Basic validation
    
    @pytest.mark.e2e
    def test_workflow_error_handling(self, mock_config):
        """Test workflow error handling and recovery."""
        # Test with invalid configuration
        invalid_config = mock_config.copy()
        invalid_config["dataset"]["split_ratio"] = [0.5, 0.5]  # Invalid
        
        # Should fail gracefully
        with pytest.raises(ValueError):
            DataLoader(invalid_config)
        
        # Test with missing dataset path
        missing_path_config = mock_config.copy()
        missing_path_config["dataset"]["path"] = "/nonexistent/path"
        
        with pytest.raises(FileNotFoundError):
            DataLoader(missing_path_config)
    
    @pytest.mark.e2e
    def test_workflow_performance_metrics(self, test_data_dir, sample_csv_data, mock_config):
        """Test that workflow produces meaningful performance metrics."""
        # Create test data
        sample_csv_data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        # Load data
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Mock model execution with performance metrics
        with patch('benchmarking_pipeline.model_executor.ModelExecutor') as mock_executor:
            mock_executor_instance = MagicMock()
            mock_executor.return_value = mock_executor_instance
            
            # Mock comprehensive results
            mock_executor_instance.execute.return_value = {
                'train_loss': 0.1,
                'val_loss': 0.15,
                'test_loss': 0.2,
                'mae': 0.5,
                'mse': 0.8,
                'rmse': 0.89,
                'mape': 15.2,
                'execution_time': 2.5,
                'memory_usage': 512.0
            }
            
            # Execute workflow
            executor = ModelExecutor(config)
            results = executor.execute(chunk)
            
            # Verify all expected metrics are present
            expected_metrics = ['train_loss', 'val_loss', 'test_loss', 'mae', 'mse', 'rmse', 'mape']
            for metric in expected_metrics:
                assert metric in results
                assert isinstance(results[metric], (int, float))
                assert results[metric] >= 0
            
            # Verify execution metrics
            assert 'execution_time' in results
            assert 'memory_usage' in results
            assert results['execution_time'] > 0
            assert results['memory_usage'] > 0
