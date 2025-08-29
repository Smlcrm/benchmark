"""
Unit tests for data loader functionality.
"""

import pytest
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import patch

from benchmarking_pipeline.pipeline.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader functionality."""
    
    @pytest.mark.unit
    def test_load_single_chunk(self, test_data_dir, sample_csv_data, mock_config):
        """Test loading a single chunk of data."""
        # Create test data files with the correct format
        data1 = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3000, 7001, 1, 2, 3000, 7001, 1, 2, 3000, 7001, 1, 2, 3000, 7001, 1, 2, 3000, 7001]'],
            'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
        })
        
        data1.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config with test directory
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        # Initialize DataLoader
        data_loader = DataLoader(config)
        
        # Load single chunk
        chunk = data_loader.load_single_chunk(1)
        
        # Verify chunk structure
        assert chunk.train is not None
        assert chunk.validation is not None
        assert chunk.test is not None
        
        # Verify data was loaded correctly
        assert len(chunk.train.targets) > 0
        assert len(chunk.validation.targets) > 0
        assert len(chunk.test.targets) > 0
    
    @pytest.mark.unit
    def test_load_several_chunks(self, test_data_dir, mock_config):
        """Test loading multiple chunks."""
        # Create test data for multiple chunks
        for i in range(1, 4):
            data = pd.DataFrame({
                'item_id': [i],
                'start': ['2023-01-01 00:00:00'],
                'freq': ['D'],
                'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'],
                'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
            })
            
            data.to_csv(os.path.join(test_data_dir, f'chunk{i:03d}.csv'), index=False)
        
        # Update config with path
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        # Initialize DataLoader
        data_loader = DataLoader(config)
        
        # Load multiple chunks
        chunks = data_loader.load_several_chunks(3)
        
        # Verify chunks were loaded
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk.train is not None
            assert chunk.validation is not None
            assert chunk.test is not None
    
    @pytest.mark.unit
    def test_invalid_chunk_number(self, test_data_dir, mock_config):
        """Test handling of invalid chunk numbers."""
        # Update config with path
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        # Initialize DataLoader
        data_loader = DataLoader(config)
        
        # Try to load non-existent chunk
        with pytest.raises(FileNotFoundError):
            data_loader.load_single_chunk(999)
    
    @pytest.mark.parametrize("split_ratio", [
        [0.7, 0.2, 0.1],
        [0.6, 0.2, 0.2],
        [0.8, 0.1, 0.1]
    ])
    @pytest.mark.unit
    def test_different_split_ratios(self, test_data_dir, sample_csv_data, mock_config, split_ratio):
        """Test different data split ratios."""
        # Create test data
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'],
            'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
        })
        
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config with custom split ratio
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["dataset"]["split_ratio"] = split_ratio
        
        # Initialize DataLoader
        data_loader = DataLoader(config)
        
        # Load chunk
        chunk = data_loader.load_single_chunk(1)
        
        # Verify split ratios are applied correctly
        total_length = 20  # Length of target data
        expected_train_length = int(total_length * split_ratio[0])
        expected_val_length = int(total_length * split_ratio[1])
        expected_test_length = int(total_length * split_ratio[2])
        
        assert len(chunk.train.targets) == expected_train_length
        assert len(chunk.validation.targets) == expected_val_length
        assert len(chunk.test.targets) == expected_test_length


class TestDataLoaderAutoDetection:
    """Tests for automatic target count detection and naming."""
    
    @pytest.mark.unit
    def test_infers_univariate(self, test_data_dir, mock_config):
        """Test that univariate data is correctly inferred."""
        # Create univariate test data
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]']
        })
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Load data; should infer univariate
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        assert chunk.metadata["num_targets"] == 1
        # Check that targets are raw arrays, not named columns
        assert isinstance(chunk.train.targets, list)
    
    @pytest.mark.unit
    def test_infers_multivariate(self, test_data_dir, mock_config):
        """Test that multivariate data is correctly inferred."""
        # Create multivariate test data
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]']
        })
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        assert chunk.metadata["num_targets"] == 2
        # Check that targets are raw arrays, not named columns
        assert isinstance(chunk.train.targets, list)
    
    @pytest.mark.unit
    def test_raw_target_arrays(self, test_data_dir, mock_config):
        """Test that targets are kept as raw arrays without artificial column naming."""
        # Create test data
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]']
        })
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Initialize DataLoader and ensure raw arrays are preserved
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Targets should be raw arrays, not DataFrames with named columns
        assert isinstance(chunk.train.targets, list)
        assert isinstance(chunk.validation.targets, list)
        assert isinstance(chunk.test.targets, list)
