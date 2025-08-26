"""
Unit tests for DataLoader functionality.
"""

import pytest
import pandas as pd
import os
import shutil
from unittest.mock import patch

from benchmarking_pipeline.pipeline.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader functionality."""
    
    def test_load_single_chunk(self, test_data_dir, sample_csv_data, mock_config):
        """Test loading a single chunk of data."""
        # Create test data files
        data1 = sample_csv_data.copy()
        data2 = sample_csv_data.copy()
        data2['timestamp'] = pd.date_range('2023-01-21', periods=20, freq='D')
        data3 = sample_csv_data.copy()
        data3['timestamp'] = pd.date_range('2023-02-10', periods=20, freq='D')
        
        # Save to CSV files
        data1.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        data2.to_csv(os.path.join(test_data_dir, 'chunk002.csv'), index=False)
        data3.to_csv(os.path.join(test_data_dir, 'chunk003.csv'), index=False)
        
        # Update config with test directory
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Test that chunk has expected structure
        assert chunk.train is not None
        assert chunk.validation is not None
        assert chunk.test is not None
        
        # Test target values (80% of 20 = 16 values for train)
        expected_train = pd.Series([1, 2, 3000, 7001] * 4)
        pd.testing.assert_series_equal(chunk.train.targets["y"], expected_train)
    
    def test_load_several_chunks(self, test_data_dir, sample_csv_data, mock_config):
        """Test loading multiple chunks of data."""
        # Create test data files
        data1 = sample_csv_data.copy()
        data2 = sample_csv_data.copy()
        data2['timestamp'] = pd.date_range('2023-01-21', periods=20, freq='D')
        data2['y'] = [8, 15, 7, 4] * 5
        
        # Save to CSV files
        data1.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        data2.to_csv(os.path.join(test_data_dir, 'chunk002.csv'), index=False)
        
        # Update config with test directory
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        data_loader = DataLoader(config)
        chunks = data_loader.load_several_chunks(2)
        
        assert len(chunks) == 2
        
        # Test first chunk validation targets (10% of 20 = 2 values)
        chunk_one = chunks[0]
        expected_validation = pd.Series([1, 2], index=[16, 17])
        pd.testing.assert_series_equal(chunk_one.validation.targets["y"], expected_validation)
        
        # Test second chunk test targets (10% of 20 = 2 values)
        chunk_two = chunks[1]
        expected_test = pd.Series([8, 15], index=[37, 38])
        pd.testing.assert_series_equal(chunk_two.test.targets["y"], expected_test)
    
    def test_invalid_chunk_number(self, mock_config):
        """Test that invalid chunk numbers raise appropriate errors."""
        data_loader = DataLoader(mock_config)
        
        with pytest.raises(ValueError):
            data_loader.load_single_chunk(0)  # Chunk numbers should start from 1
        
        with pytest.raises(ValueError):
            data_loader.load_single_chunk(-1)
    
    @pytest.mark.parametrize("split_ratio", [
        [0.7, 0.2, 0.1],
        [0.6, 0.2, 0.2],
        [0.8, 0.1, 0.1]
    ])
    def test_different_split_ratios(self, test_data_dir, sample_csv_data, mock_config, split_ratio):
        """Test DataLoader with different split ratios."""
        # Create test data
        sample_csv_data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Update config
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["dataset"]["split_ratio"] = split_ratio
        
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Verify split ratios are approximately correct
        total_length = len(sample_csv_data)
        expected_train_length = int(total_length * split_ratio[0])
        expected_val_length = int(total_length * split_ratio[1])
        expected_test_length = int(total_length * split_ratio[2])
        
        assert len(chunk.train.targets["y"]) == expected_train_length
        assert len(chunk.validation.targets["y"]) == expected_val_length
        assert len(chunk.test.targets["y"]) == expected_test_length
