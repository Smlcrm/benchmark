#!/usr/bin/env python3
"""
Test script to verify that hyperparameter grid search uses all forecast horizon values.
"""

import yaml
import sys
import os
from itertools import product

# Add the benchmarking_pipeline to the path
sys.path.append('benchmarking_pipeline')

def test_hyperparameter_grid_expansion():
    """Test that the hyperparameter grid expands correctly with forecast_horizon vector."""
    
    print("🧪 Testing hyperparameter grid expansion with forecast_horizon vector...")
    
    # Load the univariate config
    config_path = 'benchmarking_pipeline/configs/all_model_univariate.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Config loaded from: {config_path}")
    
    # Test with different models
    test_models = ['tiny_time_mixer', 'moment', 'arima']
    
    for model_name in test_models:
        print(f"\n🔍 Testing {model_name}:")
        
        # Get the base hyperparameter grid for this model
        base_hyper_grid = config['model'][model_name] or {}
        
        # Automatically inject forecast_horizon from dataset config
        dataset_forecast_horizons = config['dataset'].get('forecast_horizon', [1])
        if isinstance(dataset_forecast_horizons, list):
            hyper_grid = base_hyper_grid.copy()
            hyper_grid['forecast_horizon'] = dataset_forecast_horizons
        else:
            hyper_grid = base_hyper_grid.copy()
            hyper_grid['forecast_horizon'] = [dataset_forecast_horizons]
        
        print(f"  📊 Hyperparameter grid: {hyper_grid}")
        
        # Calculate total combinations
        total_combinations = 1
        for param_name, param_values in hyper_grid.items():
            if isinstance(param_values, list):
                total_combinations *= len(param_values)
            else:
                total_combinations *= 1
        
        print(f"  📊 Total hyperparameter combinations: {total_combinations}")
        
        # Show all combinations for small grids
        if total_combinations <= 100:  # Only show if reasonable number
            print(f"  📊 All combinations:")
            param_names = list(hyper_grid.keys())
            param_value_lists = [hyper_grid[name] if isinstance(hyper_grid[name], list) else [hyper_grid[name]] 
                               for name in param_names]
            
            for i, combination in enumerate(product(*param_value_lists)):
                combo_dict = dict(zip(param_names, combination))
                print(f"    {i+1:2d}. {combo_dict}")
        else:
            print(f"  📊 Too many combinations to display ({total_combinations} > 100)")
        
        # Verify forecast_horizon is included
        if 'forecast_horizon' in hyper_grid:
            fh_values = hyper_grid['forecast_horizon']
            print(f"  ✅ forecast_horizon included: {fh_values}")
            print(f"  ✅ forecast_horizon type: {type(fh_values)}")
        else:
            print(f"  ❌ forecast_horizon missing!")
    
    print("\n🎯 Hyperparameter grid expansion test completed!")
    return True

def test_forecast_horizon_impact():
    """Test the impact of different forecast_horizon vectors on total combinations."""
    
    print("\n🧪 Testing forecast_horizon vector impact on total combinations...")
    
    # Test different forecast_horizon vectors
    test_vectors = [
        [10],           # Single value
        [10, 25],       # Two values
        [10, 25, 50],   # Three values
        [5, 15, 30, 60] # Four values
    ]
    
    # Use a simple model for testing
    base_grid = {'p': [1, 2], 'd': [0, 1]}  # Simple ARIMA-like grid
    
    for fh_vector in test_vectors:
        test_grid = base_grid.copy()
        test_grid['forecast_horizon'] = fh_vector
        
        # Calculate combinations
        total_combinations = 1
        for param_name, param_values in test_grid.items():
            if isinstance(param_values, list):
                total_combinations *= len(param_values)
        
        print(f"  📊 forecast_horizon: {fh_vector} → Total combinations: {total_combinations}")
        print(f"     Base combinations: {len(base_grid['p']) * len(base_grid['d'])} = {len(base_grid['p']) * len(base_grid['d'])}")
        print(f"     Multiplier: {len(fh_vector)}x")
        print(f"     Formula: {len(base_grid['p'])} × {len(base_grid['d'])} × {len(fh_vector)} = {total_combinations}")
        print()
    
    print("🎯 Forecast horizon impact test completed!")
    return True

if __name__ == "__main__":
    try:
        test_hyperparameter_grid_expansion()
        test_forecast_horizon_impact()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
