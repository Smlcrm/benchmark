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
        config["model"]["parameters"] = {
            "test_model": {
                "target_cols": ["y"]
            }
        }
        
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
    def test_load_several_chunks(self, test_data_dir, sample_csv_data, mock_config):
        """Test loading multiple chunks."""
        # Create multiple test data files
        for i in range(1, 4):
            data = pd.DataFrame({
                'item_id': [i],
                'start': [f'2023-0{i}-01 00:00:00'],
                'freq': ['D'],
                'target': ['[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]'],
                'past_feat_dynamic_real': ['[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]']
            })
            data.to_csv(os.path.join(test_data_dir, f'chunk{i:03d}.csv'), index=False)
        
        # Update config with target_cols
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["model"]["parameters"] = {
            "test_model": {
                "target_cols": ["y"]
            }
        }
        
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
        # Update config with target_cols
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["model"]["parameters"] = {
            "test_model": {
                "target_cols": ["y"]
            }
        }
        
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
        
        # Update config with custom split ratio and target_cols
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["dataset"]["split_ratio"] = split_ratio
        config["model"]["parameters"] = {
            "test_model": {
                "target_cols": ["y"]
            }
        }
        
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
        
        # Valid config with target_cols
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["model"]["parameters"] = {
            "test_model": {
                "target_cols": ["sales"]
            }
        }
        
        # Should work without errors
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Verify correct column names
        assert list(chunk.train.targets.columns) == ["sales"]
        assert chunk.train.targets.shape[1] == 1  # Single column
    
    @pytest.mark.unit
    def test_missing_target_cols(self, mock_config):
        """Test that missing target_cols raises appropriate error."""
        config = mock_config.copy()
        config["model"]["parameters"] = {
            "test_model": {
                # Missing target_cols
            }
        }
        
        with pytest.raises(ValueError, match="target_cols must be defined in model parameters"):
            DataLoader(config)
    
    @pytest.mark.unit
    def test_none_target_cols(self, mock_config):
        """Test that None target_cols raises appropriate error."""
        config = mock_config.copy()
        config["model"]["parameters"] = {
            "test_model": {
                "target_cols": None
            }
        }
        
        with pytest.raises(ValueError, match="target_cols cannot be None"):
            DataLoader(config)
    
    @pytest.mark.unit
    def test_empty_list_target_cols(self, mock_config):
        """Test that empty list target_cols raises appropriate error."""
        config = mock_config.copy()
        config["model"]["parameters"] = {
            "test_model": {
                "target_cols": []
            }
        }
        
        with pytest.raises(ValueError, match="non-empty list"):
            DataLoader(config)
    
    @pytest.mark.unit
    def test_empty_model_parameters(self, mock_config):
        """Test that empty model parameters raises appropriate error."""
        config = mock_config.copy()
        config["model"]["parameters"] = {}  # Empty parameters
        
        with pytest.raises(ValueError, match="target_cols must be defined in model parameters"):
            DataLoader(config)
    
    @pytest.mark.unit
    def test_multiple_models_target_cols(self, mock_config):
        """Test that data loader uses target_cols from first available model."""
        config = mock_config.copy()
        config["model"]["parameters"] = {
            "model1": {
                "target_cols": ["sales"]
            },
            "model2": {
                "target_cols": ["revenue", "customers"]
            }
        }
        
        # Should use target_cols from first model
        data_loader = DataLoader(config)
        assert data_loader.target_cols == ["sales"]
    
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
        
        # Try to use multivariate target_cols with univariate data
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["model"]["parameters"] = {
            "test_model": {
                "target_cols": ["sales", "revenue", "customers"]  # 3 columns
            }
        }
        
        data_loader = DataLoader(config)
        
        # Should raise error for data structure mismatch
        with pytest.raises(ValueError, match="univariate but target_cols specifies"):
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
        
        # Config with custom column names
        config = mock_config.copy()
        config["dataset"]["path"] = test_data_dir
        config["model"]["parameters"] = {
            "test_model": {
                "target_cols": ["custom_column_name"]
            }
        }
        
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Verify custom column name is used
        assert list(chunk.train.targets.columns) == ["custom_column_name"]
        assert "y" not in chunk.train.targets.columns  # Should not use default names
