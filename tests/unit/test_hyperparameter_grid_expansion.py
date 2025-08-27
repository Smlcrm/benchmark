#!/usr/bin/env python3
"""
Unit tests for hyperparameter grid expansion with forecast_horizon vector.

These tests verify that hyperparameter grids expand correctly when forecast_horizon is injected.
"""

import pytest
import yaml
from pathlib import Path
from itertools import product

# Add the benchmarking_pipeline to the path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'benchmarking_pipeline'))


class TestHyperparameterGridExpansion:
    """Test class for hyperparameter grid expansion functionality."""
    
    @pytest.fixture
    def univariate_config(self):
        """Load the univariate configuration for testing."""
        config_path = Path(__file__).parent.parent.parent / 'benchmarking_pipeline/configs/all_model_univariate.yaml'
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def mock_hyperparameter_injector(self, univariate_config):
        """Create a mock hyperparameter injector for testing."""
        def inject_forecast_horizon(model_name):
            """Inject forecast_horizon into a model's hyperparameter grid."""
            base_hyper_grid = univariate_config['model'][model_name] or {}
            
            # Automatically inject forecast_horizon from dataset config
            dataset_forecast_horizons = univariate_config['dataset'].get('forecast_horizon', [1])
            if isinstance(dataset_forecast_horizons, list):
                hyper_grid = base_hyper_grid.copy()
                hyper_grid['forecast_horizon'] = dataset_forecast_horizons
            else:
                hyper_grid = base_hyper_grid.copy()
                hyper_grid['forecast_horizon'] = [dataset_forecast_horizons]
            
            return hyper_grid
        
        return inject_forecast_horizon
    
    def test_hyperparameter_grid_expansion_for_empty_model(self, mock_hyperparameter_injector):
        """Test hyperparameter grid expansion for models with no parameters."""
        grid = mock_hyperparameter_injector('tiny_time_mixer')
        
        # Should only have forecast_horizon
        assert len(grid) == 1, "Empty model should only have forecast_horizon parameter"
        assert 'forecast_horizon' in grid, "forecast_horizon should be present"
        
        # Calculate total combinations
        total_combinations = 1
        for param_name, param_values in grid.items():
            if isinstance(param_values, list):
                total_combinations *= len(param_values)
        
        assert total_combinations == 3, "Empty model with 3 forecast horizons should have 3 combinations"
    
    def test_hyperparameter_grid_expansion_for_foundation_model(self, mock_hyperparameter_injector):
        """Test hyperparameter grid expansion for foundation models."""
        grid = mock_hyperparameter_injector('moment')
        
        # Should have all original parameters plus forecast_horizon
        expected_params = {'model_path', 'context_length', 'fine_tune_epochs', 'batch_size', 'learning_rate', 'forecast_horizon'}
        assert set(grid.keys()) == expected_params, "Foundation model should have all expected parameters"
        
        # Calculate total combinations
        total_combinations = 1
        for param_name, param_values in grid.items():
            if isinstance(param_values, list):
                total_combinations *= len(param_values)
        
        # moment model: 1×1×1×1×1×3 = 3 combinations
        assert total_combinations == 3, "Foundation model should have 3 combinations (3 forecast horizons)"
    
    def test_hyperparameter_grid_expansion_for_univariate_model(self, mock_hyperparameter_injector):
        """Test hyperparameter grid expansion for univariate models."""
        grid = mock_hyperparameter_injector('arima')
        
        # Should have all original parameters plus forecast_horizon
        expected_params = {'p', 'd', 'q', 's', 'loss_function', 'forecast_horizon'}
        assert set(grid.keys()) == expected_params, "Univariate model should have all expected parameters"
        
        # Calculate total combinations
        total_combinations = 1
        for param_name, param_values in grid.items():
            if isinstance(param_values, list):
                total_combinations *= len(param_values)
        
        # arima model: 3×2×3×2×1×3 = 108 combinations
        assert total_combinations == 108, "ARIMA model should have 108 combinations"
    
    def test_forecast_horizon_vector_impact_on_combinations(self):
        """Test how different forecast_horizon vectors affect total combinations."""
        # Use a simple base grid for testing
        base_grid = {'p': [1, 2], 'd': [0, 1]}
        base_combinations = len(base_grid['p']) * len(base_grid['d'])  # 2 × 2 = 4
        
        # Test different forecast_horizon vectors
        test_vectors = [
            [10],           # Single value
            [10, 25],       # Two values
            [10, 25, 50],   # Three values
            [5, 15, 30, 60] # Four values
        ]
        
        for fh_vector in test_vectors:
            test_grid = base_grid.copy()
            test_grid['forecast_horizon'] = fh_vector
            
            # Calculate total combinations
            total_combinations = 1
            for param_name, param_values in test_grid.items():
                if isinstance(param_values, list):
                    total_combinations *= len(param_values)
            
            expected_combinations = base_combinations * len(fh_vector)
            assert total_combinations == expected_combinations, \
                f"forecast_horizon {fh_vector} should multiply combinations by {len(fh_vector)}"
    
    def test_hyperparameter_grid_combinations_display_logic(self, mock_hyperparameter_injector):
        """Test the logic for displaying hyperparameter combinations."""
        grid = mock_hyperparameter_injector('tiny_time_mixer')
        
        # Calculate total combinations
        total_combinations = 1
        for param_name, param_values in grid.items():
            if isinstance(param_values, list):
                total_combinations *= len(param_values)
        
        # tiny_time_mixer has 3 combinations (≤ 100), so should be displayable
        assert total_combinations <= 100, "tiny_time_mixer should have ≤ 100 combinations for display"
        
        # Verify all combinations can be generated
        param_names = list(grid.keys())
        param_value_lists = [grid[name] if isinstance(grid[name], list) else [grid[name]] 
                           for name in param_names]
        
        combinations = list(product(*param_value_lists))
        assert len(combinations) == total_combinations, "Generated combinations should match calculated total"
        
        # Verify each combination has the right structure
        for combo in combinations:
            combo_dict = dict(zip(param_names, combo))
            assert 'forecast_horizon' in combo_dict, "Each combination should have forecast_horizon"
            assert isinstance(combo_dict['forecast_horizon'], int), "forecast_horizon should be integer in combination"
    
    def test_hyperparameter_grid_parameter_preservation(self, mock_hyperparameter_injector, univariate_config):
        """Test that hyperparameter grid injection preserves all original parameters."""
        original_grid = univariate_config['model']['arima'].copy()
        injected_grid = mock_hyperparameter_injector('arima')
        
        # All original parameters should be preserved
        for param_name, param_value in original_grid.items():
            assert param_name in injected_grid, f"Original parameter {param_name} should be preserved"
            assert injected_grid[param_name] == param_value, f"Original parameter {param_name} value should be unchanged"
        
        # forecast_horizon should be added
        assert 'forecast_horizon' in injected_grid, "forecast_horizon should be added"
        assert isinstance(injected_grid['forecast_horizon'], list), "Added forecast_horizon should be a list"
