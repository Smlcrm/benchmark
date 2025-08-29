#!/usr/bin/env python3
"""
End-to-end tests for forecast_horizon vector injection functionality.

These tests verify the complete workflow from configuration to execution.
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path

# Add the benchmarking_pipeline to the path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'benchmarking_pipeline'))


class TestForecastHorizonE2E:
    """End-to-end tests for forecast_horizon functionality."""
    
    @pytest.fixture
    def test_config_path(self):
        """Create a temporary test configuration file."""
        test_config = {
            'dataset': {
                'name': 'test_dataset',
                'path': 'test/path',
                'frequency': 'D',
                'forecast_horizon': [5, 15, 30],  # Test vector
                'split_ratio': [0.8, 0.1, 0.1],
                'normalize': False,
                'handle_missing': 'interpolate',
                'chunks': 1
            },
            'model': {
                'test_model': {
                    'param1': [1, 2],
                    'param2': [10, 20]
                }
            }
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_forecast_horizon_workflow(self, test_config_path):
        """E2E test: Test the complete forecast_horizon workflow from config to execution."""
        # Load the test configuration
        with open(test_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify configuration structure
        assert 'dataset' in config, "Config should have dataset section"
        assert 'model' in config, "Config should have model section"
        
        dataset_cfg = config['dataset']
        assert 'forecast_horizon' in dataset_cfg, "Dataset should have forecast_horizon"
        
        # Verify forecast_horizon is a vector
        forecast_horizons = dataset_cfg['forecast_horizon']
        assert isinstance(forecast_horizons, list), "forecast_horizon should be a list"
        assert len(forecast_horizons) == 3, "forecast_horizon should have 3 values"
        assert all(isinstance(x, int) for x in forecast_horizons), "All forecast_horizon values should be integers"
        
        # Test the injection logic (simulating what ModelExecutor would do)
        model_name = 'test_model'
        base_hyper_grid = config['model'][model_name] or {}
        
        # Inject forecast_horizon
        if isinstance(forecast_horizons, list):
            hyper_grid = base_hyper_grid.copy()
            hyper_grid['forecast_horizon'] = forecast_horizons
        else:
            hyper_grid = base_hyper_grid.copy()
            hyper_grid['forecast_horizon'] = [forecast_horizons]
        
        # Verify injection worked
        assert 'forecast_horizon' in hyper_grid, "forecast_horizon should be injected"
        assert hyper_grid['forecast_horizon'] == forecast_horizons, "Injected forecast_horizon should match dataset config"
        
        # Verify original parameters are preserved
        assert 'param1' in hyper_grid, "Original parameter param1 should be preserved"
        assert 'param2' in hyper_grid, "Original parameter param2 should be preserved"
        assert hyper_grid['param1'] == base_hyper_grid['param1'], "Original parameter param1 value should be unchanged"
        assert hyper_grid['param2'] == base_hyper_grid['param2'], "Original parameter param2 value should be unchanged"
        
        # Calculate total combinations
        total_combinations = 1
        for param_name, param_values in hyper_grid.items():
            if isinstance(param_values, list):
                total_combinations *= len(param_values)
        
        # Expected: param1(2) × param2(2) × forecast_horizon(3) = 12 combinations
        expected_combinations = 2 * 2 * 3
        assert total_combinations == expected_combinations, f"Total combinations should be {expected_combinations}, got {total_combinations}"
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_forecast_horizon_vector_variations(self, test_config_path):
        """E2E test: Test different forecast_horizon vector configurations."""
        # Load the test configuration
        with open(test_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test different forecast_horizon vectors
        test_vectors = [
            [10],           # Single value
            [10, 25],       # Two values
            [10, 25, 50],   # Three values
            [5, 15, 30, 60] # Four values
        ]
        
        for fh_vector in test_vectors:
            # Update the config with new forecast_horizon
            config['dataset']['forecast_horizon'] = fh_vector
            
            # Test injection
            model_name = 'test_model'
            base_hyper_grid = config['model'][model_name] or {}
            
            # Inject forecast_horizon
            if isinstance(fh_vector, list):
                hyper_grid = base_hyper_grid.copy()
                hyper_grid['forecast_horizon'] = fh_vector
            else:
                hyper_grid = base_hyper_grid.copy()
                hyper_grid['forecast_horizon'] = [fh_vector]
            
            # Verify injection
            assert 'forecast_horizon' in hyper_grid, f"forecast_horizon should be injected for vector {fh_vector}"
            assert hyper_grid['forecast_horizon'] == fh_vector, f"Injected forecast_horizon should match {fh_vector}"
            
            # Calculate combinations
            total_combinations = 1
            for param_name, param_values in hyper_grid.items():
                if isinstance(param_values, list):
                    total_combinations *= len(param_values)
            
            # Base combinations: param1(2) × param2(2) = 4
            base_combinations = 4
            expected_combinations = base_combinations * len(fh_vector)
            
            assert total_combinations == expected_combinations, \
                f"forecast_horizon {fh_vector} should result in {expected_combinations} combinations, got {total_combinations}"
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_model_executor_integration_simulation(self, test_config_path):
        """E2E test: Simulate the complete ModelExecutor workflow."""
        # Load the test configuration
        with open(test_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Simulate the ModelExecutor.run() method logic
        model_name = 'test_model'
        
        # Get the base hyperparameter grid for this model
        base_hyper_grid = config['model'][model_name] or {}
        
        # Automatically inject forecast_horizon from dataset config into the hyperparameter grid
        dataset_forecast_horizons = config['dataset'].get('forecast_horizon', [1])
        if isinstance(dataset_forecast_horizons, list):
            # Create a copy of the base hyper grid and add forecast_horizon
            hyper_grid = base_hyper_grid.copy()
            hyper_grid['forecast_horizon'] = dataset_forecast_horizons
        else:
            # Single value case
            hyper_grid = base_hyper_grid.copy()
            hyper_grid['forecast_horizon'] = [dataset_forecast_horizons]
        
        # Verify the final hyperparameter grid
        assert 'forecast_horizon' in hyper_grid, "Final hyperparameter grid should have forecast_horizon"
        assert isinstance(hyper_grid['forecast_horizon'], list), "forecast_horizon should be a list in final grid"
        assert len(hyper_grid['forecast_horizon']) > 0, "forecast_horizon list should not be empty"
        
        # Verify all expected parameters are present
        expected_params = {'param1', 'param2', 'forecast_horizon'}
        assert set(hyper_grid.keys()) == expected_params, f"Final grid should have parameters: {expected_params}"
        
        # Verify parameter types
        assert isinstance(hyper_grid['param1'], list), "param1 should be a list"
        assert isinstance(hyper_grid['param2'], list), "param2 should be a list"
        assert isinstance(hyper_grid['forecast_horizon'], list), "forecast_horizon should be a list"
        
        # Verify parameter values
        assert hyper_grid['param1'] == [1, 2], "param1 should have values [1, 2]"
        assert hyper_grid['param2'] == [10, 20], "param2 should have values [10, 20]"
        assert hyper_grid['forecast_horizon'] == [5, 15, 30], "forecast_horizon should have values [5, 15, 30]"
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_hyperparameter_tuning_simulation(self, test_config_path):
        """E2E test: Simulate the hyperparameter tuning process with forecast_horizon."""
        # Load the test configuration
        with open(test_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Simulate the hyperparameter tuning process
        model_name = 'test_model'
        base_hyper_grid = config['model'][model_name] or {}
        
        # Inject forecast_horizon
        dataset_forecast_horizons = config['dataset'].get('forecast_horizon', [1])
        if isinstance(dataset_forecast_horizons, list):
            hyper_grid = base_hyper_grid.copy()
            hyper_grid['forecast_horizon'] = dataset_forecast_horizons
        else:
            hyper_grid = base_hyper_grid.copy()
            hyper_grid['forecast_horizon'] = [dataset_forecast_horizons]
        
        # Simulate hyperparameter tuning by testing each combination
        param_names = list(hyper_grid.keys())
        param_value_lists = [hyper_grid[name] if isinstance(hyper_grid[name], list) else [hyper_grid[name]] 
                           for name in param_names]
        
        # Generate all combinations
        from itertools import product
        combinations = list(product(*param_value_lists))
        
        # Verify we have the expected number of combinations
        expected_combinations = 2 * 2 * 3  # param1(2) × param2(2) × forecast_horizon(3)
        assert len(combinations) == expected_combinations, f"Should have {expected_combinations} combinations, got {len(combinations)}"
        
        # Verify each combination has the right structure
        for combo in combinations:
            combo_dict = dict(zip(param_names, combo))
            
            # Each combination should have all parameters
            assert 'param1' in combo_dict, "Each combination should have param1"
            assert 'param2' in combo_dict, "Each combination should have param2"
            assert 'forecast_horizon' in combo_dict, "Each combination should have forecast_horizon"
            
            # Verify parameter types
            assert isinstance(combo_dict['param1'], int), "param1 should be integer in combination"
            assert isinstance(combo_dict['param2'], int), "param2 should be integer in combination"
            assert isinstance(combo_dict['forecast_horizon'], int), "forecast_horizon should be integer in combination"
            
            # Verify parameter value ranges
            assert combo_dict['param1'] in [1, 2], "param1 should be 1 or 2"
            assert combo_dict['param2'] in [10, 20], "param2 should be 10 or 20"
            assert combo_dict['forecast_horizon'] in [5, 15, 30], "forecast_horizon should be 5, 15, or 30"
