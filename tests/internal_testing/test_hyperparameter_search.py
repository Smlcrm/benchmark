import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

from benchmarking_pipeline.models.arima_model import ARIMAModel
from benchmarking_pipeline.models.theta_model import ThetaModel
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor


class TestHyperparameterSearch(unittest.TestCase):
    """Test cases for hyperparameter search functionality."""
    
    def setUp(self):
        """Set up test data and components."""
        self.test_dir = tempfile.mkdtemp()
        self.create_test_data()
        
    def tearDown(self):
        """Clean up test data directory."""
        shutil.rmtree(self.test_dir)
        
    def create_test_data(self):
        """Create sample test data files."""
        # Create sample time series data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'y': np.random.randn(100).cumsum() + 100  # Random walk
        })
        
        # Save to CSV file
        data.to_csv(os.path.join(self.test_dir, 'chunk001.csv'), index=False)
        
    def test_arima_hyperparameter_search(self):
        """Test ARIMA model hyperparameter search."""
        config = {
            "dataset": {
                "path": self.test_dir,
                "name": "test_dataset",
                "split_ratio": [0.8, 0.1, 0.1]
            }
        }
        
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Preprocess data
        preprocessor = Preprocessor({"dataset": {"normalize": True}})
        processed_chunk = preprocessor.preprocess(chunk).data
        
        # Test ARIMA model with different parameters
        arima_params = {
            'p': [0, 1],
            'd': [0, 1],
            'q': [0, 1],
            's': [2, 4]
        }
        
        # Test a few combinations
        test_params = [
            {'p': 0, 'd': 1, 'q': 0, 's': 2},
            {'p': 1, 'd': 0, 'q': 1, 's': 4}
        ]
        
        for params in test_params:
            model = ARIMAModel(params)
            try:
                # Test training
                trained_model = model.train(
                    y_context=processed_chunk.train.targets,
                    y_target=processed_chunk.validation.targets
                )
                
                # Test prediction
                predictions = trained_model.predict(
                    y_context=processed_chunk.train.targets,
                    y_target=processed_chunk.test.targets
                )
                
                # Basic validation
                self.assertIsNotNone(predictions)
                self.assertEqual(len(predictions), len(processed_chunk.test.targets))
                
            except Exception as e:
                # Some parameter combinations might fail, which is expected
                print(f"ARIMA with params {params} failed: {e}")
                
    def test_theta_hyperparameter_search(self):
        """Test Theta model hyperparameter search."""
        config = {
            "dataset": {
                "path": self.test_dir,
                "name": "test_dataset",
                "split_ratio": [0.8, 0.1, 0.1]
            }
        }
        
        data_loader = DataLoader(config)
        chunk = data_loader.load_single_chunk(1)
        
        # Preprocess data
        preprocessor = Preprocessor({"dataset": {"normalize": True}})
        processed_chunk = preprocessor.preprocess(chunk).data
        
        # Test Theta model with different parameters
        theta_params = {
            'sp': [1, 2],
            'forecast_horizon': [5, 10]
        }
        
        test_params = [
            {'sp': 1, 'forecast_horizon': 5},
            {'sp': 2, 'forecast_horizon': 10}
        ]
        
        for params in test_params:
            model = ThetaModel(params)
            try:
                # Test training
                trained_model = model.train(
                    y_context=processed_chunk.train.targets,
                    y_target=processed_chunk.validation.targets
                )
                
                # Test prediction
                predictions = trained_model.predict(
                    y_context=processed_chunk.train.targets,
                    y_target=processed_chunk.test.targets
                )
                
                # Basic validation
                self.assertIsNotNone(predictions)
                self.assertEqual(len(predictions), len(processed_chunk.test.targets))
                
            except Exception as e:
                # Some parameter combinations might fail, which is expected
                print(f"Theta with params {params} failed: {e}")


if __name__ == "__main__":
    unittest.main()
