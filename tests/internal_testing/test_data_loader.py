import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch

from benchmarking_pipeline.pipeline.data_loader import DataLoader
from .testing_utilities import value_equality


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader functionality."""
    
    def setUp(self):
        """Set up test data directory with sample CSV files."""
        self.test_dir = tempfile.mkdtemp()
        self.create_test_data()
        
    def tearDown(self):
        """Clean up test data directory."""
        shutil.rmtree(self.test_dir)
        
    def create_test_data(self):
        """Create sample test data files."""
        # Create sample data for testing
        data1 = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=20, freq='D'),
            'y': [1, 2, 3000, 7001] * 5
        })
        
        data2 = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-21', periods=20, freq='D'),
            'y': [8, 15, 7, 4] * 5
        })
        
        data3 = pd.DataFrame({
            'timestamp': pd.date_range('2023-02-10', periods=20, freq='D'),
            'y': [5, 10, 15, 20] * 5
        })
        
        # Save to CSV files
        data1.to_csv(os.path.join(self.test_dir, 'chunk001.csv'), index=False)
        data2.to_csv(os.path.join(self.test_dir, 'chunk002.csv'), index=False)
        data3.to_csv(os.path.join(self.test_dir, 'chunk003.csv'), index=False)
        
    def test_load_single_chunk(self):
        """Test loading a single chunk of data."""
        config = {
            "dataset": {
                "path": self.test_dir,
                "name": "test_dataset",
                "split_ratio": [0.8, 0.1, 0.1]
            }
        }
        
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Test that chunk has expected structure
        self.assertIsNotNone(chunk.train)
        self.assertIsNotNone(chunk.validation)
        self.assertIsNotNone(chunk.test)
        
        # Test target values
        expected_train = pd.Series([1, 2, 3000, 7001] * 4)  # 80% of 20 = 16 values
        self.assertTrue(value_equality(chunk.train.targets["y"], expected_train))
        
    def test_load_several_chunks(self):
        """Test loading multiple chunks of data."""
        config = {
            "dataset": {
                "path": self.test_dir,
                "name": "test_dataset",
                "split_ratio": [0.8, 0.1, 0.1]
            }
        }
        
        data_loader = DataLoader(config)
        chunks = data_loader.load_several_chunks(2)
        
        self.assertEqual(len(chunks), 2)
        
        # Test first chunk validation targets
        chunk_one = chunks[0]
        expected_validation = pd.Series([1, 2], index=[16, 17])
        self.assertTrue(value_equality(chunk_one.validation.targets["y"], expected_validation))
        
        # Test second chunk test targets
        chunk_two = chunks[1]
        expected_test = pd.Series([8, 15, 7, 4, 8], index=[37, 38, 39, 40, 41])
        self.assertTrue(value_equality(chunk_two.test.targets["y"], expected_test))


if __name__ == "__main__":
    unittest.main()
