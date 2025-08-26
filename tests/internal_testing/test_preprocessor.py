import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):
    """Test cases for preprocessor functionality."""
    
    def setUp(self):
        """Set up test data and components."""
        self.test_dir = tempfile.mkdtemp()
        self.create_test_data()
        
    def tearDown(self):
        """Clean up test data directory."""
        shutil.rmtree(self.test_dir)
        
    def create_test_data(self):
        """Create sample test data files with missing values and outliers."""
        # Create sample time series data with missing values and outliers
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create data with some missing values and outliers
        values = np.random.randn(100) * 10 + 100
        
        # Add some missing values
        missing_indices = [10, 25, 50, 75]
        values[missing_indices] = np.nan
        
        # Add some outliers
        outlier_indices = [15, 30, 60, 85]
        values[outlier_indices] = [1000, -500, 800, -300]
        
        data = pd.DataFrame({
            'timestamp': dates,
            'y': values
        })
        
        # Save to CSV file
        data.to_csv(os.path.join(self.test_dir, 'chunk001.csv'), index=False)
        
    def test_preprocessor_no_normalization(self):
        """Test preprocessor without normalization."""
        config = {
            "dataset": {
                "path": self.test_dir,
                "name": "test_dataset",
                "split_ratio": [0.8, 0.1, 0.1]
            }
        }
        
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Test preprocessor with no normalization
        preprocessor = Preprocessor({
            "dataset": {
                "normalize": False,
                "handle_missing": "interpolate",
                "remove_outliers": False
            }
        })
        
        preprocessed_chunk = preprocessor.preprocess(chunk)
        
        # Basic validation
        self.assertIsNotNone(preprocessed_chunk)
        self.assertIsNotNone(preprocessed_chunk.data)
        self.assertIsNotNone(preprocessed_chunk.data.train)
        self.assertIsNotNone(preprocessed_chunk.data.validation)
        self.assertIsNotNone(preprocessed_chunk.data.test)
        
    def test_preprocessor_with_normalization(self):
        """Test preprocessor with normalization."""
        config = {
            "dataset": {
                "path": self.test_dir,
                "name": "test_dataset",
                "split_ratio": [0.8, 0.1, 0.1]
            }
        }
        
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Test preprocessor with normalization
        preprocessor = Preprocessor({
            "dataset": {
                "normalize": True,
                "normalization_method": "standard",
                "handle_missing": "interpolate",
                "remove_outliers": False
            }
        })
        
        preprocessed_chunk = preprocessor.preprocess(chunk)
        
        # Basic validation
        self.assertIsNotNone(preprocessed_chunk)
        self.assertIsNotNone(preprocessed_chunk.data)
        
        # Check that data is normalized (should have mean close to 0 and std close to 1)
        train_data = preprocessed_chunk.data.train.targets['y'].values
        if not np.all(np.isnan(train_data)):
            mean = np.nanmean(train_data)
            std = np.nanstd(train_data)
            self.assertAlmostEqual(mean, 0, delta=0.1)
            self.assertAlmostEqual(std, 1, delta=0.1)
        
    def test_preprocessor_missing_value_handling(self):
        """Test preprocessor missing value handling."""
        config = {
            "dataset": {
                "path": self.test_dir,
                "name": "test_dataset",
                "split_ratio": [0.8, 0.1, 0.1]
            }
        }
        
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Test preprocessor with median imputation
        preprocessor = Preprocessor({
            "dataset": {
                "normalize": False,
                "handle_missing": "median",
                "remove_outliers": False
            }
        })
        
        preprocessed_chunk = preprocessor.preprocess(chunk)
        
        # Check that no missing values remain
        train_data = preprocessed_chunk.data.train.targets['y']
        validation_data = preprocessed_chunk.data.validation.targets['y']
        test_data = preprocessed_chunk.data.test.targets['y']
        
        self.assertFalse(train_data.isna().any())
        self.assertFalse(validation_data.isna().any())
        self.assertFalse(test_data.isna().any())
        
    def test_preprocessor_outlier_removal(self):
        """Test preprocessor outlier removal."""
        config = {
            "dataset": {
                "path": self.test_dir,
                "name": "test_dataset",
                "split_ratio": [0.8, 0.1, 0.1]
            }
        }
        
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Test preprocessor with outlier removal
        preprocessor = Preprocessor({
            "dataset": {
                "normalize": False,
                "handle_missing": "interpolate",
                "remove_outliers": True,
                "outlier_threshold": 3
            }
        })
        
        preprocessed_chunk = preprocessor.preprocess(chunk)
        
        # Basic validation
        self.assertIsNotNone(preprocessed_chunk)
        self.assertIsNotNone(preprocessed_chunk.data)
        
        # Check that extreme outliers are handled (this is a basic check)
        train_data = preprocessed_chunk.data.train.targets['y'].values
        if len(train_data) > 0:
            # Check that no extremely large values remain
            self.assertLess(np.max(np.abs(train_data)), 1000)


if __name__ == "__main__":
    unittest.main()
