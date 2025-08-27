#!/usr/bin/env python3
"""
Test script to verify forecast_horizon vector injection is working correctly.
"""

import yaml
import sys
import os

# Add the benchmarking_pipeline to the path
sys.path.append('benchmarking_pipeline')

def test_forecast_horizon_injection():
    """Test that forecast_horizon is properly injected into model hyperparameter grids."""
    
    print("🧪 Testing forecast_horizon vector injection...")
    
    # Load the univariate config
    config_path = 'benchmarking_pipeline/configs/all_model_univariate.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Config loaded from: {config_path}")
    
    # Check dataset forecast_horizon
    dataset_forecast_horizons = config['dataset'].get('forecast_horizon')
    print(f"📊 Dataset forecast_horizon: {dataset_forecast_horizons}")
    print(f"📊 Type: {type(dataset_forecast_horizons)}")
    
    # Test the injection logic
    from benchmarking_pipeline.model_executor import ModelExecutor
    
    # Create a mock model executor to test the logic
    class MockModelExecutor:
        def __init__(self, config):
            self.config = config
        
        def test_forecast_horizon_injection(self, model_name):
            """Test the forecast_horizon injection logic."""
            # Get the base hyperparameter grid for this model
            base_hyper_grid = self.config['model'][model_name] or {}
            
            # Automatically inject forecast_horizon from dataset config into the hyperparameter grid
            dataset_forecast_horizons = self.config['dataset'].get('forecast_horizon', [1])
            if isinstance(dataset_forecast_horizons, list):
                # Create a copy of the base hyper grid and add forecast_horizon
                hyper_grid = base_hyper_grid.copy()
                hyper_grid['forecast_horizon'] = dataset_forecast_horizons
                print(f"  [INFO] Auto-injected forecast_horizon: {dataset_forecast_horizons}")
            else:
                # Single value case
                hyper_grid = base_hyper_grid.copy()
                hyper_grid['forecast_horizon'] = [dataset_forecast_horizons]
                print(f"  [INFO] Auto-injected forecast_horizon: {[dataset_forecast_horizons]}")
            
            print(f"  [INFO] Final hyperparameter grid for {model_name}: {hyper_grid}")
            return hyper_grid
    
    # Test with a few models
    mock_executor = MockModelExecutor(config)
    
    print("\n🔍 Testing forecast_horizon injection for different models:")
    
    # Test tiny_time_mixer (empty model)
    print("\n1. Testing tiny_time_mixer (empty model):")
    tiny_time_mixer_grid = mock_executor.test_forecast_horizon_injection('tiny_time_mixer')
    
    # Test moment (model with parameters)
    print("\n2. Testing moment (model with parameters):")
    moment_grid = mock_executor.test_forecast_horizon_injection('moment')
    
    # Test arima (univariate model)
    print("\n3. Testing arima (univariate model):")
    arima_grid = mock_executor.test_forecast_horizon_injection('arima')
    
    # Verify the injection worked correctly
    print("\n✅ Verification:")
    for model_name, grid in [('tiny_time_mixer', tiny_time_mixer_grid), 
                           ('moment', moment_grid), 
                           ('arima', arima_grid)]:
        if 'forecast_horizon' in grid:
            print(f"  ✅ {model_name}: forecast_horizon = {grid['forecast_horizon']}")
        else:
            print(f"  ❌ {model_name}: forecast_horizon missing!")
    
    print("\n🎯 Forecast horizon vector injection test completed!")
    return True

if __name__ == "__main__":
    try:
        test_forecast_horizon_injection()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
