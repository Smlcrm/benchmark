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
        
        # Update config with test directory and target_cols
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


class TestDataLoaderTargetColsEnforcement:
    """Test cases for mandatory target_cols enforcement."""
    
    @pytest.mark.unit
    def test_valid_target_cols_configuration(self, test_data_dir, sample_csv_data, mock_config):
        """Test data loader with valid target_cols configuration."""
        # Create test data
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'],
            'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
        })
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
    
        # Valid config with target_cols in dataset
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        
        # Should work without errors
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Verify target columns are correctly named
        assert list(chunk.train.targets.columns) == ["y"]
    
    @pytest.mark.unit
    def test_missing_target_cols(self, mock_config):
        """Test that missing target_cols in dataset raises appropriate error."""
        config = mock_config.copy()
        del config["dataset"]["target_cols"]  # Remove target_cols from dataset
    
        with pytest.raises(ValueError, match="target_cols must be defined in dataset configuration"):
            DataLoader(config)
    
    @pytest.mark.unit
    def test_none_target_cols(self, mock_config):
        """Test that None target_cols raises appropriate error."""
        config = mock_config.copy()
        config["dataset"]["target_cols"] = None  # Set target_cols to None
    
        with pytest.raises(ValueError, match="target_cols must be defined in dataset configuration"):
            DataLoader(config)
    
    @pytest.mark.unit
    def test_empty_list_target_cols(self, mock_config):
        """Test that empty list target_cols raises appropriate error."""
        config = mock_config.copy()
        config["dataset"]["target_cols"] = []  # Set target_cols to empty list
    
        with pytest.raises(ValueError, match="target_cols must be a non-empty list of column names"):
            DataLoader(config)
    
    @pytest.mark.unit
    def test_empty_model_parameters(self, test_data_dir, mock_config):
        """Test that empty model parameters doesn't affect target_cols validation."""
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir  # Use test_data_dir which has metadata.json
        config["model"]["parameters"] = {}  # Empty parameters
    
        # Should work since target_cols is in dataset
        data_loader = DataLoader(config)
        assert data_loader.target_cols == ["y"]
    
    @pytest.mark.unit
    def test_multiple_models_target_cols(self, test_data_dir, mock_config):
        """Test that data loader uses target_cols from dataset regardless of model parameters."""
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir  # Use test_data_dir which has metadata.json
        config["model"]["parameters"] = {
            "model1": {},
            "model2": {}
        }
        
        # Should use target_cols from dataset
        data_loader = DataLoader(config)
        assert data_loader.target_cols == ["y"]
    
    @pytest.mark.unit
    def test_data_structure_validation(self, test_data_dir, sample_csv_data, mock_config):
        """Test that data loader validates data structure against target_cols."""
        # Create univariate test data
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'],
            'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
        })
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Create metadata.json for multivariate target_cols
        import json
        metadata = {
            "variables": [
                {"var_name": "sales", "target_index": 0},
                {"var_name": "revenue", "target_index": 1},
                {"var_name": "customers", "target_index": 2}
            ]
        }
        metadata_path = os.path.join(test_data_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
        # Try to use multivariate target_cols with univariate data
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["dataset"]["target_cols"] = ["sales", "revenue", "customers"]  # 3 columns
    
        # Should raise error due to data structure mismatch during data loading
        data_loader = DataLoader(config)
        with pytest.raises(ValueError, match="Data is univariate but target_cols specifies 3 columns"):
            data_loader.load_single_chunk(1)
    
    @pytest.mark.unit
    def test_column_naming_respects_target_cols(self, test_data_dir, sample_csv_data, mock_config):
        """Test that column names are correctly applied from target_cols configuration."""
        # Create test data
        data = pd.DataFrame({
            'item_id': [1],
            'start': ['2023-01-01 00:00:00'],
            'freq': ['D'],
            'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'],
            'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
        })
        data.to_csv(os.path.join(test_data_dir, 'chunk001.csv'), index=False)
        
        # Create metadata.json for custom column name
        import json
        metadata = {
            "variables": [
                {"var_name": "custom_column_name", "target_index": 0}
            ]
        }
        metadata_path = os.path.join(test_data_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
        # Config with custom column names
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["dataset"]["target_cols"] = ["custom_column_name"]
        
        # Initialize DataLoader
        data_loader = DataLoader(config)
        
        # Load chunk
        chunk = data_loader.load_single_chunk(1)
        
        # Verify column names are correctly applied
        assert list(chunk.train.targets.columns) == ["custom_column_name"]
        assert list(chunk.validation.targets.columns) == ["custom_column_name"]
        assert list(chunk.test.targets.columns) == ["custom_column_name"]
