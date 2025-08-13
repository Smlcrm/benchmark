#!/usr/bin/env python3
"""
Test script to verify MultivariateLSTM compatibility with hyperparameter tuning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from benchmarking_pipeline.models.anyvariate.lstm_model import MultivariateLSTMModel
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.pipeline.data_types import DatasetSplit, Dataset

def create_test_multivariate_data():
    """Create test multivariate dataset."""
    # Create synthetic multivariate time series data
    np.random.seed(42)
    n_samples = 100
    n_targets = 2
    
    # Generate correlated time series
    base_series = np.cumsum(np.random.randn(n_samples))
    target1 = base_series + 0.1 * np.random.randn(n_samples)
    target2 = base_series * 0.8 + 0.2 * np.random.randn(n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'target_0': target1,
        'target_1': target2
    })
    
    # Create timestamps
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size + val_size]
    test_data = data.iloc[train_size + val_size:]
    
    train_timestamps = timestamps[:train_size]
    val_timestamps = timestamps[train_size:train_size + val_size]
    test_timestamps = timestamps[train_size + val_size:]
    
    # Create dataset splits
    train_split = DatasetSplit(
        targets=train_data,
        timestamps=train_timestamps.values,
        features=None
    )
    
    val_split = DatasetSplit(
        targets=val_data,
        timestamps=val_timestamps.values,
        features=None
    )
    
    test_split = DatasetSplit(
        targets=test_data,
        timestamps=test_timestamps.values,
        features=None
    )
    
    # Create dataset
    dataset = Dataset(
        train=train_split,
        validation=val_split,
        test=test_split,
        name="test_multivariate",
        metadata={
            "start": "2023-01-01",
            "freq": "D"
        }
    )
    
    return dataset

def test_multivariate_lstm_compatibility():
    """Test that MultivariateLSTM works with hyperparameter tuning."""
    print("Testing MultivariateLSTM compatibility with hyperparameter tuning...")
    
    # Create test data
    dataset = create_test_multivariate_data()
    print(f"Created test dataset with shape: {dataset.train.targets.shape}")
    print(f"Target columns: {list(dataset.train.targets.columns)}")
    
    # Create MultivariateLSTM model
    config = {
        'units': 20,
        'layers': 1,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 5,  # Small number for testing
        'sequence_length': 5,
        'forecast_horizon': 3,
        'target_cols': ['target_0', 'target_1'],
        'loss_functions': ['mae'],
        'primary_loss': 'mae'
    }
    
    model = MultivariateLSTMModel(config)
    print(f"Created MultivariateLSTM model with {model.n_targets} targets")
    
    # Test basic training and prediction
    print("\nTesting basic training and prediction...")
    model.train(
        y_context=dataset.train.targets,
        y_target=dataset.validation.targets
    )
    
    predictions = model.predict(
        y_context=dataset.train.targets,
        y_target=dataset.test.targets
    )
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Expected shape: ({len(dataset.test.targets)}, {model.n_targets})")
    
    # Test hyperparameter tuning compatibility
    print("\nTesting hyperparameter tuning compatibility...")
    
    # Define hyperparameter ranges
    hyperparameter_ranges = {
        'units': [10, 20],
        'dropout': [0.1, 0.2],
        'learning_rate': [0.001, 0.01]
    }
    
    # Create tuner
    tuner = HyperparameterTuner(model, hyperparameter_ranges, use_exog=False)
    
    # Test grid search (with reduced iterations for testing)
    print("Running hyperparameter grid search...")
    try:
        validation_score, best_hyperparameters = tuner.hyperparameter_grid_search_several_time_series([dataset])
        print(f"Grid search completed successfully!")
        print(f"Best validation score: {validation_score}")
        print(f"Best hyperparameters: {best_hyperparameters}")
        
        # Test final evaluation
        print("\nTesting final evaluation...")
        results = tuner.final_evaluation(best_hyperparameters, [dataset])
        print(f"Final evaluation results: {results}")
        
        print("\n✅ All tests passed! MultivariateLSTM is compatible with hyperparameter tuning.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_multivariate_lstm_compatibility()
    if success:
        print("\n🎉 MultivariateLSTM compatibility test completed successfully!")
    else:
        print("\n💥 MultivariateLSTM compatibility test failed!")
        sys.exit(1) 