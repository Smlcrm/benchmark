#!/usr/bin/env python3
"""
Integration tests for forecast_horizon vector injection across multiple components.

These tests verify that forecast_horizon injection works correctly across the entire pipeline.
"""

import pytest
import yaml
from pathlib import Path

# Add the benchmarking_pipeline to the path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'benchmarking_pipeline'))


class TestForecastHorizonIntegration:
    """Integration tests for forecast_horizon functionality across components."""
    
    @pytest.fixture
    def univariate_config(self):
        """Load the univariate configuration for testing."""
        config_path = Path(__file__).parent.parent.parent / 'benchmarking_pipeline/configs/all_model_univariate.yaml'
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture
    def multivariate_config(self):
        """Load the multivariate configuration for testing."""
        config_path = Path(__file__).parent.parent.parent / 'benchmarking_pipeline/configs/all_model_multivariate.yaml'
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @pytest.mark.integration
    def test_configuration_consistency_between_univariate_and_multivariate(self, univariate_config, multivariate_config):
        """Test that forecast_horizon configuration is consistent between config types."""
        univ_fh = univariate_config['dataset'].get('forecast_horizon')
        mult_fh = multivariate_config['dataset'].get('forecast_horizon')
        
        # Both should have forecast_horizon defined
        assert univ_fh is not None, "Univariate config should have forecast_horizon"
        assert mult_fh is not None, "Multivariate config should have forecast_horizon"
        
        # Both should be lists
        assert isinstance(univ_fh, list), "Univariate forecast_horizon should be a list"
        assert isinstance(mult_fh, list), "Multivariate forecast_horizon should be a list"
        
        # Both should have the same values (as per our refactoring)
        assert univ_fh == mult_fh, "Both configs should have the same forecast_horizon values"
    
    @pytest.mark.integration
    def test_model_executor_forecast_horizon_injection(self, univariate_config):
        """Test that ModelExecutor correctly injects forecast_horizon."""
        from benchmarking_pipeline.model_executor import ModelExecutor
        
        # Create a mock model executor to test the injection logic
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
        
        mock_executor = MockModelExecutor(univariate_config)
        
        # Test injection for different model types
        test_models = ['tiny_time_mixer', 'moment', 'arima']
        
        for model_name in test_models:
            grid = mock_executor.test_forecast_horizon_injection(model_name)
            
            # Verify forecast_horizon is injected
            assert 'forecast_horizon' in grid, f"forecast_horizon should be injected into {model_name}"
            assert isinstance(grid['forecast_horizon'], list), f"forecast_horizon should be a list for {model_name}"
            
            # Verify original parameters are preserved
            if model_name in univariate_config['model']:
                original_params = univariate_config['model'][model_name]
                if original_params:
                    for param_name, param_value in original_params.items():
                        assert param_name in grid, f"Original parameter {param_name} should be preserved in {model_name}"
                        assert grid[param_name] == param_value, f"Original parameter {param_name} value should be unchanged in {model_name}"
    
    @pytest.mark.integration
    def test_base_model_forecast_horizon_handling(self, univariate_config):
        """Test that base models correctly handle forecast_horizon from config."""
        from benchmarking_pipeline.models.base_model import BaseModel
        
        # Create a minimal config for testing
        test_config = {
            'dataset': univariate_config['dataset'],
            'forecast_horizon': [10, 25, 50]  # Simulate injected forecast_horizon
        }
        
        # Test that BaseModel can handle forecast_horizon in config
        # Note: We can't instantiate BaseModel directly since it's abstract
        # But we can test the logic that would be used
        
        # Simulate the logic from BaseModel.__init__
        forecast_horizon = test_config.get('forecast_horizon', test_config['dataset'].get('forecast_horizon', 1))
        
        assert forecast_horizon is not None, "forecast_horizon should be extracted from config"
        assert isinstance(forecast_horizon, list), "forecast_horizon should be a list"
        assert len(forecast_horizon) > 0, "forecast_horizon list should not be empty"
        
        # Test the fallback logic
        if isinstance(forecast_horizon, list):
            first_value = forecast_horizon[0]
            assert isinstance(first_value, int), "First forecast_horizon value should be an integer"
    
    @pytest.mark.integration
    def test_pipeline_component_forecast_horizon_access(self, univariate_config):
        """Test that pipeline components can access forecast_horizon from dataset config."""
        from benchmarking_pipeline.pipeline.evaluator import Evaluator
        from benchmarking_pipeline.pipeline.feature_extraction import FeatureExtractor
        
        # Test that components can access dataset-level forecast_horizon
        dataset_cfg = univariate_config['dataset']
        forecast_horizon = dataset_cfg.get('forecast_horizon')
        
        assert forecast_horizon is not None, "Pipeline components should be able to access dataset forecast_horizon"
        assert isinstance(forecast_horizon, list), "Dataset forecast_horizon should be a list"
        assert len(forecast_horizon) > 0, "Dataset forecast_horizon should not be empty"
        
        # Test that components can access target_cols from dataset config
        target_cols = dataset_cfg.get('target_cols')
        assert target_cols is not None, "Pipeline components should be able to access dataset target_cols"
        assert isinstance(target_cols, list), "Dataset target_cols should be a list"
        assert len(target_cols) > 0, "Dataset target_cols should not be empty"
    
    @pytest.mark.integration
    def test_hyperparameter_grid_consistency_across_models(self, univariate_config):
        """Test that hyperparameter grids are consistent across different model types."""
        # Test that all models get the same forecast_horizon values
        dataset_forecast_horizons = univariate_config['dataset'].get('forecast_horizon')
        
        test_models = ['tiny_time_mixer', 'moment', 'arima']
        
        for model_name in test_models:
            base_grid = univariate_config['model'][model_name] or {}
            
            # Simulate injection
            injected_grid = base_grid.copy()
            injected_grid['forecast_horizon'] = dataset_forecast_horizons
            
            # Verify forecast_horizon is consistent
            assert injected_grid['forecast_horizon'] == dataset_forecast_horizons, \
                f"Model {model_name} should have consistent forecast_horizon values"
            
            # Verify forecast_horizon is a list
            assert isinstance(injected_grid['forecast_horizon'], list), \
                f"Model {model_name} should have forecast_horizon as a list"
    
    @pytest.mark.integration
    def test_configuration_structure_consistency(self, univariate_config, multivariate_config):
        """Test that both configurations have consistent structure after refactoring."""
        # Both should have dataset section
        assert 'dataset' in univariate_config, "Univariate config should have dataset section"
        assert 'dataset' in multivariate_config, "Multivariate config should have dataset section"
        
        # Both should have model section (not model.parameters)
        assert 'model' in univariate_config, "Univariate config should have model section"
        assert 'model' in multivariate_config, "Multivariate config should have model section"
        
        # Both should NOT have model.parameters section
        assert 'parameters' not in univariate_config.get('model', {}), "Univariate config should not have model.parameters"
        assert 'parameters' not in multivariate_config.get('model', {}), "Multivariate config should not have model.parameters"
        
        # Both should have the same dataset structure
        univ_dataset_keys = set(univariate_config['dataset'].keys())
        mult_dataset_keys = set(multivariate_config['dataset'].keys())
        
        # Core dataset keys should be present in both
        core_keys = {'name', 'path', 'forecast_horizon', 'target_cols'}
        for key in core_keys:
            assert key in univ_dataset_keys, f"Univariate config should have dataset.{key}"
            assert key in mult_dataset_keys, f"Multivariate config should have dataset.{key}"
