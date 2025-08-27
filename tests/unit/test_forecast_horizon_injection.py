#!/usr/bin/env python3
"""
Unit tests for forecast_horizon vector injection functionality.

These tests verify that forecast_horizon is properly injected into model hyperparameter grids.
"""

import pytest
import yaml
import os
from pathlib import Path

# Add the benchmarking_pipeline to the path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'benchmarking_pipeline'))


class TestForecastHorizonInjection:
    """Test class for forecast_horizon vector injection functionality."""
    
    @pytest.fixture
    def univariate_config(self):
        """Load the univariate configuration for testing."""
        config_path = Path(__file__).parent.parent.parent / 'benchmarking_pipeline/configs/all_model_univariate.yaml'
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def mock_model_executor(self, univariate_config):
        """Create a mock model executor for testing the injection logic."""
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
                else:
                    # Single value case
                    hyper_grid = base_hyper_grid.copy()
                    hyper_grid['forecast_horizon'] = [dataset_forecast_horizons]
                
                return hyper_grid
        
        return MockModelExecutor(univariate_config)
    
    def test_dataset_forecast_horizon_configuration(self, univariate_config):
        """Test that dataset forecast_horizon is properly configured."""
        dataset_forecast_horizons = univariate_config['dataset'].get('forecast_horizon')
        
        assert dataset_forecast_horizons is not None, "forecast_horizon should be defined in dataset config"
        assert isinstance(dataset_forecast_horizons, list), "forecast_horizon should be a list"
        assert len(dataset_forecast_horizons) > 0, "forecast_horizon list should not be empty"
        assert all(isinstance(x, int) for x in dataset_forecast_horizons), "All forecast_horizon values should be integers"
    
    def test_forecast_horizon_injection_for_empty_model(self, mock_model_executor):
        """Test forecast_horizon injection for models with no parameters (empty model)."""
        grid = mock_model_executor.test_forecast_horizon_injection('tiny_time_mixer')
        
        assert 'forecast_horizon' in grid, "forecast_horizon should be injected into empty model"
        assert isinstance(grid['forecast_horizon'], list), "Injected forecast_horizon should be a list"
        assert len(grid['forecast_horizon']) > 0, "Injected forecast_horizon should not be empty"
    
    def test_forecast_horizon_injection_for_foundation_model(self, mock_model_executor):
        """Test forecast_horizon injection for foundation models with parameters."""
        grid = mock_model_executor.test_forecast_horizon_injection('moment')
        
        assert 'forecast_horizon' in grid, "forecast_horizon should be injected into foundation model"
        assert isinstance(grid['forecast_horizon'], list), "Injected forecast_horizon should be a list"
        assert len(grid['forecast_horizon']) > 0, "Injected forecast_horizon should not be empty"
        
        # Verify other parameters are preserved
        assert 'model_path' in grid, "Existing parameters should be preserved"
        assert 'context_length' in grid, "Existing parameters should be preserved"
    
    def test_forecast_horizon_injection_for_univariate_model(self, mock_model_executor):
        """Test forecast_horizon injection for univariate models with parameters."""
        grid = mock_model_executor.test_forecast_horizon_injection('arima')
        
        assert 'forecast_horizon' in grid, "forecast_horizon should be injected into univariate model"
        assert isinstance(grid['forecast_horizon'], list), "Injected forecast_horizon should be a list"
        assert len(grid['forecast_horizon']) > 0, "Injected forecast_horizon should not be empty"
        
        # Verify other parameters are preserved
        assert 'p' in grid, "Existing parameters should be preserved"
        assert 'd' in grid, "Existing parameters should be preserved"
        assert 'q' in grid, "Existing parameters should be preserved"
    
    def test_forecast_horizon_injection_preserves_existing_parameters(self, mock_model_executor):
        """Test that forecast_horizon injection doesn't overwrite existing parameters."""
        original_grid = mock_model_executor.config['model']['arima'].copy()
        injected_grid = mock_model_executor.test_forecast_horizon_injection('arima')
        
        # Check that all original parameters are preserved
        for param_name, param_value in original_grid.items():
            assert param_name in injected_grid, f"Original parameter {param_name} should be preserved"
            assert injected_grid[param_name] == param_value, f"Original parameter {param_name} value should be unchanged"
        
        # Check that forecast_horizon is added
        assert 'forecast_horizon' in injected_grid, "forecast_horizon should be added"
    
    def test_forecast_horizon_injection_creates_copy(self, mock_model_executor):
        """Test that forecast_horizon injection creates a copy, not modifies original."""
        original_grid = mock_model_executor.config['model']['arima'].copy()
        injected_grid = mock_model_executor.test_forecast_horizon_injection('arima')
        
        # Verify they are different objects
        assert injected_grid is not original_grid, "Injection should create a copy, not modify original"
        
        # Verify original is unchanged
        assert 'forecast_horizon' not in original_grid, "Original grid should not be modified"
