#!/usr/bin/env python3
"""
Test script to verify MultivariateThetaModel works with Beijing dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import numpy as np
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.models.multivariate.theta_model import MultivariateTheta

def test_multivariate_theta_beijing():
    """Test MultivariateTheta with Beijing dataset."""
    
    # Load the config
    config_path = "benchmarking_pipeline/configs/multivariate_forecast_horizon_config_theta.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=== Testing MultivariateTheta with Beijing Dataset ===")
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
    
    # Test different configurations
    test_configs = [
        {
            'sp': 1,
            'theta_method': 'least_squares',
            'use_reduced_rank': False,
            'forecast_horizon': 50
        },
        {
            'sp': 48,  # Daily seasonality in 30min data (48 = 24*2)
            'theta_method': 'correlation_optimal',
            'use_reduced_rank': False,
            'forecast_horizon': 50
        }
    ]
    
    for i, model_params in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"Testing Configuration {i+1}: {model_params}")
        print(f"{'='*60}")
        
        # Create MultivariateTheta model
        model = MultivariateTheta(model_params)
        print(f"Created MultivariateTheta model")
        print(f"Model target_cols: {model.target_cols}")
        print(f"Model n_targets: {model.n_targets}")
        
        # Test training
        print(f"\n=== Testing Training ===")
        y_context = chunk.train.targets
        y_target = chunk.validation.targets
        
        print(f"Training with y_context shape: {y_context.shape}")
        print(f"Training with y_target shape: {y_target.shape}")
        print(f"Y context columns: {list(y_context.columns)}")
        
        try:
            trained_model = model.train(y_context=y_context, y_target=y_target)
            print("✓ Training completed successfully")
            print(f"✓ Model is_fitted: {trained_model.is_fitted}")
            print(f"✓ Estimated Θ matrix shape: {trained_model.theta_matrix.shape}")
            print(f"✓ Drift vector: {trained_model.drift_vector}")
            print(f"✓ Number of univariate models: {len(trained_model.univariate_models)}")
        except Exception as e:
            print(f"✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Test prediction
        print(f"\n=== Testing Prediction ===")
        try:
            # Test with explicit forecast_horizon
            predictions_fh = trained_model.predict(
                y_context=y_context, 
                forecast_horizon=chunk.test.targets.shape[0]
            )
            print(f"✓ Prediction with forecast_horizon completed successfully")
            print(f"Predictions shape: {predictions_fh.shape}")
            print(f"Expected shape: {chunk.test.targets.shape}")
            
            # Test with x_target for length determination
            predictions_xt = trained_model.predict(
                y_context=y_context,
                x_target=np.zeros((chunk.test.targets.shape[0], 1))  # Dummy x_target
            )
            print(f"✓ Prediction with x_target completed successfully")
            print(f"Predictions shape: {predictions_xt.shape}")
            
            if predictions_fh.shape == chunk.test.targets.shape:
                print("✓ Prediction shape matches expected shape")
            else:
                print("✗ Prediction shape mismatch")
                print(f"Expected: {chunk.test.targets.shape}, Got: {predictions_fh.shape}")
                continue
                
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Test loss computation
        print(f"\n=== Testing Loss Computation ===")
        try:
            loss = trained_model.compute_loss(chunk.test.targets.values, predictions_fh)
            print(f"✓ Loss computation completed successfully")
            print(f"Loss metrics: {loss}")
            
            # Check if all metrics are reasonable (not NaN/inf)
            all_metrics_valid = all(np.isfinite(v) and not np.isnan(v) for v in loss.values())
            if all_metrics_valid:
                print("✓ All loss metrics are valid")
            else:
                print("✗ Some loss metrics are invalid (NaN/inf)")
                
        except Exception as e:
            print(f"✗ Loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Test parameter methods
        print(f"\n=== Testing Parameter Methods ===")
        try:
            params = trained_model.get_params()
            print(f"✓ get_params() successful")
            print(f"Parameters keys: {list(params.keys())}")
            
            # Test set_params
            trained_model.set_params(forecast_horizon=25)
            print(f"✓ set_params() successful")
            
        except Exception as e:
            print(f"✗ Parameter methods failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Test save/load
        print(f"\n=== Testing Save/Load ===")
        try:
            model_path = f"test_multivariate_theta_config_{i+1}.pkl"
            trained_model.save(model_path)
            print(f"✓ Model saved to {model_path}")
            
            # Create new model and load
            new_model = MultivariateTheta()
            new_model.load(model_path)
            print(f"✓ Model loaded successfully")
            print(f"✓ Loaded model is_fitted: {new_model.is_fitted}")
            
            # Test prediction with loaded model
            loaded_predictions = new_model.predict(
                y_context=y_context,
                forecast_horizon=10
            )
            print(f"✓ Prediction with loaded model successful")
            print(f"Loaded predictions shape: {loaded_predictions.shape}")
            
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"✓ Cleaned up {model_path}")
                
        except Exception as e:
            print(f"✗ Save/Load failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print(f"\n✓ Configuration {i+1} - All Tests Passed!")
    
    print(f"\n{'='*60}")
    print(f"=== All MultivariateTheta Tests Completed! ===")
    print(f"{'='*60}")
    return True



if __name__ == "__main__":
    print("Starting MultivariateTheta Testing...")
    
    # Test with multivariate data
    success_multivariate = test_multivariate_theta_beijing()
    
    if success_multivariate:
        print("\n All MultivariateTheta tests passed!")
        print("The model is working correctly with multivariate data!")
    else:
        print("\n❌ MultivariateTheta tests failed.")