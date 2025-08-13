#!/usr/bin/env python3
"""
Test script to verify MultivariateLSTM works with Beijing dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.models.multivariate.lstm_model import MultivariateLSTMModel

def test_multivariate_lstm_beijing():
    """Test MultivariateLSTM with Beijing dataset."""
    
    # Load the config
    config_path = "benchmarking_pipeline/configs/multivariate_forecast_horizon_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=== Testing MultivariateLSTM with Beijing Dataset ===")
    print(f"Config loaded from: {config_path}")
    
    # Load dataset
    dataset_cfg = config['dataset']
    data_loader = DataLoader({"dataset": {
        "path": dataset_cfg['path'],
        "name": dataset_cfg['name'],
        "split_ratio": dataset_cfg['split_ratio']
    }})
    
    all_dataset_chunks = data_loader.load_several_chunks(dataset_cfg['chunks'])
    print(f"Loaded {len(all_dataset_chunks)} chunks")
    
    # Preprocess
    preprocessor = Preprocessor({"dataset": {"normalize": dataset_cfg.get('normalize', False)}})
    all_dataset_chunks = [preprocessor.preprocess(chunk).data for chunk in all_dataset_chunks]
    
    # Check dataset structure
    chunk = all_dataset_chunks[0]
    print(f"\nDataset structure:")
    print(f"Train targets shape: {chunk.train.targets.shape}")
    print(f"Train targets columns: {list(chunk.train.targets.columns)}")
    print(f"Validation targets shape: {chunk.validation.targets.shape}")
    print(f"Test targets shape: {chunk.test.targets.shape}")
    
    # Create MultivariateLSTM model
    lstm_params = config['model']['parameters']['LSTM']
    model_params = {}
    for k, v in lstm_params.items():
        if isinstance(v, list):
            model_params[k] = v[0]
        else:
            model_params[k] = v
    
    print(f"\nModel parameters: {model_params}")
    
    model = MultivariateLSTMModel(model_params)
    print(f"Created MultivariateLSTM model")
    print(f"Model target_cols: {model.target_cols}")
    
    # Test training
    print(f"\n=== Testing Training ===")
    y_context = chunk.train.targets
    y_target = chunk.validation.targets
    
    print(f"Training with y_context shape: {y_context.shape}")
    print(f"Training with y_target shape: {y_target.shape}")
    
    try:
        trained_model = model.train(y_context=y_context, y_target=y_target)
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False
    
    # Test prediction
    print(f"\n=== Testing Prediction ===")
    try:
        predictions = trained_model.predict(y_context=y_context, y_target=chunk.test.targets)
        print(f"✓ Prediction completed successfully")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Expected shape: {chunk.test.targets.shape}")
        
        if predictions.shape == chunk.test.targets.shape:
            print("✓ Prediction shape matches expected shape")
        else:
            print("✗ Prediction shape mismatch")
            return False
            
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False
    
    # Test loss computation
    print(f"\n=== Testing Loss Computation ===")
    try:
        loss = trained_model.compute_loss(chunk.test.targets.values, predictions)
        print(f"✓ Loss computation completed successfully")
        print(f"Loss: {loss}")
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False
    
    print(f"\n=== All Tests Passed! ===")
    return True

if __name__ == "__main__":
    success = test_multivariate_lstm_beijing()
    if success:
        print("MultivariateLSTM is working correctly with Beijing dataset!")
    else:
        print("MultivariateLSTM has issues with Beijing dataset.") 