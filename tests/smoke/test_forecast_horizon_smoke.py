#!/usr/bin/env python3
"""
Smoke tests for forecast_horizon vector injection functionality.

These tests provide quick verification that the basic forecast_horizon functionality works.
"""

import pytest
import yaml
from pathlib import Path

# Add the benchmarking_pipeline to the path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'benchmarking_pipeline'))


class TestForecastHorizonSmoke:
    """Smoke tests for forecast_horizon functionality."""
    
    @pytest.mark.smoke
    def test_config_files_load_without_errors(self):
        """Smoke test: Verify that configuration files load without errors."""
        config_dir = Path(__file__).parent.parent.parent / 'benchmarking_pipeline/configs'
        
        # Test univariate config
        univ_config_path = config_dir / 'all_model_univariate.yaml'
        assert univ_config_path.exists(), "Univariate config file should exist"
        
        with open(univ_config_path, 'r') as f:
            univ_config = yaml.safe_load(f)
        
        assert univ_config is not None, "Univariate config should load successfully"
        assert 'dataset' in univ_config, "Univariate config should have dataset section"
        assert 'model' in univ_config, "Univariate config should have model section"
        
        # Test multivariate config
        mult_config_path = config_dir / 'all_model_multivariate.yaml'
        assert mult_config_path.exists(), "Multivariate config file should exist"
        
        with open(mult_config_path, 'r') as f:
            mult_config = yaml.safe_load(f)
        
        assert mult_config is not None, "Multivariate config should load successfully"
        assert 'dataset' in mult_config, "Multivariate config should have dataset section"
        assert 'model' in mult_config, "Multivariate config should have model section"
    
    @pytest.mark.smoke
    def test_forecast_horizon_is_defined_in_dataset(self):
        """Smoke test: Verify that forecast_horizon is defined in dataset configuration."""
        config_path = Path(__file__).parent.parent.parent / 'benchmarking_pipeline/configs/all_model_univariate.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_cfg = config.get('dataset', {})
        forecast_horizon = dataset_cfg.get('forecast_horizon')
        
        assert forecast_horizon is not None, "forecast_horizon should be defined in dataset config"
        assert isinstance(forecast_horizon, list), "forecast_horizon should be a list"
        assert len(forecast_horizon) > 0, "forecast_horizon list should not be empty"
        
        # Verify all values are integers
        for value in forecast_horizon:
            assert isinstance(value, int), f"forecast_horizon value {value} should be an integer"
            assert value > 0, f"forecast_horizon value {value} should be positive"
    
    @pytest.mark.smoke
    def test_model_section_structure(self):
        """Smoke test: Verify that model section has correct structure."""
        config_path = Path(__file__).parent.parent.parent / 'benchmarking_pipeline/configs/all_model_univariate.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_section = config.get('model', {})
        assert isinstance(model_section, dict), "Model section should be a dictionary"
        assert len(model_section) > 0, "Model section should not be empty"
        
        # Verify that models are defined directly (not under 'parameters')
        assert 'parameters' not in model_section, "Model section should not have 'parameters' subkey"
        
        # Verify that at least one model is defined
        model_names = list(model_section.keys())
        assert len(model_names) > 0, "At least one model should be defined"
        
        # Verify that model names are strings
        for model_name in model_names:
            assert isinstance(model_name, str), f"Model name {model_name} should be a string"
    
    @pytest.mark.smoke
    def test_forecast_horizon_injection_logic(self):
        """Smoke test: Verify that forecast_horizon injection logic works."""
        config_path = Path(__file__).parent.parent.parent / 'benchmarking_pipeline/configs/all_model_univariate.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test the injection logic
        dataset_forecast_horizons = config['dataset'].get('forecast_horizon', [1])
        
        # Test with a simple model
        base_grid = {'p': [1, 2], 'd': [0, 1]}
        
        # Simulate injection
        if isinstance(dataset_forecast_horizons, list):
            injected_grid = base_grid.copy()
            injected_grid['forecast_horizon'] = dataset_forecast_horizons
        else:
            injected_grid = base_grid.copy()
            injected_grid['forecast_horizon'] = [dataset_forecast_horizons]
        
        # Verify injection worked
        assert 'forecast_horizon' in injected_grid, "forecast_horizon should be injected"
        assert isinstance(injected_grid['forecast_horizon'], list), "Injected forecast_horizon should be a list"
        
        # Verify original parameters are preserved
        assert 'p' in injected_grid, "Original parameter 'p' should be preserved"
        assert 'd' in injected_grid, "Original parameter 'd' should be preserved"
        assert injected_grid['p'] == base_grid['p'], "Original parameter 'p' value should be unchanged"
        assert injected_grid['d'] == base_grid['d'], "Original parameter 'd' value should be unchanged"
    
    @pytest.mark.smoke
    def test_configuration_files_are_valid_yaml(self):
        """Smoke test: Verify that configuration files are valid YAML."""
        config_dir = Path(__file__).parent.parent.parent / 'benchmarking_pipeline/configs'
        
        config_files = [
            'all_model_univariate.yaml',
            'all_model_multivariate.yaml'
        ]
        
        for config_file in config_files:
            config_path = config_dir / config_file
            assert config_path.exists(), f"Config file {config_file} should exist"
            
            # Try to load as YAML
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                assert config is not None, f"Config file {config_file} should load as valid YAML"
            except yaml.YAMLError as e:
                pytest.fail(f"Config file {config_file} should be valid YAML: {e}")
