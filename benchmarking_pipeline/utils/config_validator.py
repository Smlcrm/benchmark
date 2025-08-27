"""
Configuration validation utility for the benchmarking pipeline.

This module provides comprehensive validation of configuration files to ensure
they comply with the expected schema before execution.
"""

import yaml
import os
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    
    def __init__(self, message: str, field_path: str = "", details: str = ""):
        self.message = message
        self.field_path = field_path
        self.details = details
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        if self.field_path:
            return f"Config validation error at '{self.field_path}': {self.message}"
        return f"Config validation error: {self.message}"


class ConfigValidator:
    """
    Validates configuration files against the expected schema.
    
    Ensures all required fields are present, data types are correct,
    and values are within acceptable ranges.
    """
    
    def __init__(self):
        # Define the expected schema
        self.schema = {
            'test_type': {
                'type': str,
                'required': True,
                'allowed_values': [
                    'univariate', 'multivariate', 'additive', 'multiplicative',
                    'deterministic', 'probabilistic', 'irregular', 'regular'
                ]
            },
            'tensorboard': {
                'type': bool,
                'required': False,
                'default': False
            },
            'dataset': {
                'type': dict,
                'required': True,
                'fields': {
                    'name': {'type': str, 'required': True},
                    'path': {'type': str, 'required': True},
                    'frequency': {'type': str, 'required': True},
                    'forecast_horizon': {'type': int, 'required': True, 'min': 1},
                    'split_ratio': {
                        'type': list,
                        'required': True,
                        'min_length': 3,
                        'max_length': 3,
                        'element_type': float,
                        'custom_validator': self._validate_split_ratio
                    },
                    'normalize': {'type': bool, 'required': True},
                    'handle_missing': {
                        'type': str,
                        'required': True,
                        'allowed_values': [
                            'interpolate', 'mean', 'median', 'drop',
                            'forward_fill', 'backward_fill'
                        ]
                    },
                    'chunks': {'type': int, 'required': False, 'min': 1, 'default': 1},
                    'num_targets': {'type': int, 'required': False, 'min': 1, 'default': 1}
                }
            },
            'model': {
                'type': dict,
                'required': True,
                'fields': {
                    'name': {
                        'type': list,
                        'required': True,
                        'min_length': 1,
                        'element_type': str,
                        'custom_validator': self._validate_model_names
                    },
                    'parameters': {
                        'type': dict,
                        'required': True,
                        'custom_validator': self._validate_model_parameters
                    }
                }
            },
            'evaluation': {
                'type': dict,
                'required': True,
                'fields': {
                    'type': {
                        'type': str,
                        'required': True,
                        'allowed_values': ['deterministic', 'probabilistic']
                    },
                    'metrics': {
                        'type': list,
                        'required': True,
                        'min_length': 1,
                        'element_type': str,
                        'custom_validator': self._validate_metrics
                    }
                }
            }
        }
        
        # Define required model parameters for each model type
        self.model_parameter_schemas = {
            'arima': {
                'p': {'type': list, 'element_type': int, 'min': 0},
                'd': {'type': list, 'element_type': int, 'min': 0},
                'q': {'type': list, 'element_type': int, 'min': 0},
                's': {'type': list, 'element_type': int, 'min': 1},
                'maxlags': {'type': list, 'element_type': int, 'min': 1},
                'target_cols': {'type': list, 'element_type': str, 'required': True},
                'loss_functions': {'type': list, 'element_type': str, 'required': True},
                'primary_loss': {'type': list, 'element_type': str, 'required': True},
                'forecast_horizon': {'type': list, 'element_type': int, 'min': 1}
            },
            'lstm': {
                'units': {'type': list, 'element_type': int, 'min': 1},
                'layers': {'type': list, 'element_type': int, 'min': 1},
                'dropout': {'type': list, 'element_type': float, 'min': 0.0, 'max': 1.0},
                'learning_rate': {'type': list, 'element_type': float, 'min': 0.0},
                'batch_size': {'type': list, 'element_type': int, 'min': 1},
                'epochs': {'type': list, 'element_type': int, 'min': 1},
                'sequence_length': {'type': list, 'element_type': int, 'min': 1},
                'target_cols': {'type': list, 'element_type': str, 'required': True},
                'loss_functions': {'type': list, 'element_type': str, 'required': True},
                'primary_loss': {'type': list, 'element_type': str, 'required': True},
                'forecast_horizon': {'type': list, 'element_type': int, 'min': 1}
            },
            'xgboost': {
                'lookback_window': {'type': list, 'element_type': int, 'min': 1},
                'forecast_horizon': {'type': list, 'element_type': int, 'min': 1},
                'n_estimators': {'type': list, 'element_type': int, 'min': 1},
                'max_depth': {'type': list, 'element_type': int, 'min': 1},
                'learning_rate': {'type': list, 'element_type': float, 'min': 0.0},
                'subsample': {'type': list, 'element_type': float, 'min': 0.0, 'max': 1.0},
                'colsample_bytree': {'type': list, 'element_type': float, 'min': 0.0, 'max': 1.0},
                'random_state': {'type': list, 'element_type': int},
                'n_jobs': {'type': list, 'element_type': int},
                'target_cols': {'type': list, 'element_type': str, 'required': True}
            },
            'theta': {
                'sp': {'type': list, 'element_type': int, 'min': 1},
                'forecast_horizon': {'type': list, 'element_type': int, 'min': 1},
                'method': {'type': list, 'element_type': str, 'allowed_values': ['additive', 'multiplicative']},
                'target_cols': {'type': list, 'element_type': str, 'required': True},
                'loss_functions': {'type': list, 'element_type': str, 'required': True},
                'primary_loss': {'type': list, 'element_type': str, 'required': True},
                'use_reduced_rank': {'type': list, 'element_type': bool},
                'theta_method': {'type': list, 'element_type': str, 'allowed_values': ['least_squares', 'correlation_optimal']}
            },
            'svr': {
                'kernel': {'type': list, 'element_type': str, 'allowed_values': ['rbf', 'linear', 'poly', 'sigmoid']},
                'C': {'type': list, 'element_type': float, 'min': 0.0},
                'epsilon': {'type': list, 'element_type': float, 'min': 0.0},
                'gamma': {'type': list, 'element_type': str, 'allowed_values': ['scale', 'auto']},
                'lookback_window': {'type': list, 'element_type': int, 'min': 1},
                'forecast_horizon': {'type': list, 'element_type': int, 'min': 1},
                'max_iter': {'type': list, 'element_type': int, 'min': 1},
                'random_state': {'type': list, 'element_type': int},
                'target_cols': {'type': list, 'element_type': str, 'required': True}
            }
        }
        
        # Define allowed metrics for each evaluation type
        self.allowed_metrics = {
            'deterministic': ['mae', 'rmse', 'mape', 'smape', 'mase'],
            'probabilistic': ['crps', 'quantile_loss', 'interval_score', 'mae', 'rmse']
        }
    
    def validate_config(self, config: Dict[str, Any], config_path: str = "") -> bool:
        """
        Validate a configuration dictionary against the schema.
        
        Args:
            config: Configuration dictionary to validate
            config_path: Path to the config file (for error reporting)
            
        Returns:
            True if validation passes
            
        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            self._validate_dict(config, self.schema, "")
            logger.info(f"Configuration validation passed for {config_path or 'config'}")
            return True
        except ConfigValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def validate_config_file(self, config_path: str) -> bool:
        """
        Load and validate a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            True if validation passes
            
        Raises:
            ConfigValidationError: If validation fails
            FileNotFoundError: If config file doesn't exist
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ConfigValidationError("Configuration file is empty or invalid YAML")
            
            return self.validate_config(config, config_path)
            
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML format: {e}")
    
    def _validate_dict(self, data: Dict[str, Any], schema: Dict[str, Any], path: str):
        """Validate a dictionary against its schema."""
        if not isinstance(data, dict):
            raise ConfigValidationError(f"Expected dict, got {type(data).__name__}", path)
        
        # Check required fields
        for field_name, field_schema in schema.items():
            if field_schema.get('required', False) and field_name not in data:
                raise ConfigValidationError(f"Required field '{field_name}' is missing", path)
        
        # Validate each field
        for field_name, field_value in data.items():
            if field_name not in schema:
                logger.warning(f"Unknown field '{field_name}' in {path or 'root'}")
                continue
            
            field_schema = schema[field_name]
            field_path = f"{path}.{field_name}" if path else field_name
            
            self._validate_field(field_value, field_schema, field_path)
    
    def _validate_field(self, value: Any, schema: Dict[str, Any], path: str):
        """Validate a single field against its schema."""
        expected_type = schema.get('type')
        
        # Type validation
        if expected_type and not isinstance(value, expected_type):
            raise ConfigValidationError(
                f"Expected {expected_type.__name__}, got {type(value).__name__}", 
                path
            )
        
        # Handle different types
        if expected_type == dict:
            if 'fields' in schema:
                self._validate_dict(value, schema['fields'], path)
            elif 'custom_validator' in schema:
                schema['custom_validator'](value, path)
        
        elif expected_type == list:
            self._validate_list(value, schema, path)
        
        elif expected_type == str:
            if 'allowed_values' in schema and value not in schema['allowed_values']:
                raise ConfigValidationError(
                    f"Value '{value}' not allowed. Allowed values: {schema['allowed_values']}", 
                    path
                )
        
        elif expected_type == int:
            if 'min' in schema and value < schema['min']:
                raise ConfigValidationError(f"Value must be >= {schema['min']}, got {value}", path)
            if 'max' in schema and value > schema['max']:
                raise ConfigValidationError(f"Value must be <= {schema['max']}, got {value}", path)
        
        elif expected_type == float:
            if 'min' in schema and value < schema['min']:
                raise ConfigValidationError(f"Value must be >= {schema['min']}, got {value}", path)
            if 'max' in schema and value > schema['max']:
                raise ConfigValidationError(f"Value must be <= {schema['max']}, got {value}", path)
        
        # Custom validation
        if 'custom_validator' in schema:
            schema['custom_validator'](value, path)
    
    def _validate_list(self, data: List[Any], schema: Dict[str, Any], path: str):
        """Validate a list against its schema."""
        if not isinstance(data, list):
            raise ConfigValidationError(f"Expected list, got {type(data).__name__}", path)
        
        # Length validation
        if 'min_length' in schema and len(data) < schema['min_length']:
            raise ConfigValidationError(
                f"List must have at least {schema['min_length']} elements, got {len(data)}", 
                path
            )
        
        if 'max_length' in schema and len(data) > schema['max_length']:
            raise ConfigValidationError(
                f"List must have at most {schema['max_length']} elements, got {len(data)}", 
                path
            )
        
        # Element type validation
        if 'element_type' in schema:
            for i, element in enumerate(data):
                if not isinstance(element, schema['element_type']):
                    raise ConfigValidationError(
                        f"Element {i} must be {schema['element_type'].__name__}, got {type(element).__name__}", 
                        f"{path}[{i}]"
                    )
    
    def _validate_split_ratio(self, value: List[float], path: str):
        """Validate that split ratios sum to 1.0."""
        if abs(sum(value) - 1.0) > 1e-6:
            raise ConfigValidationError(
                f"Split ratios must sum to 1.0, got {sum(value)}", 
                path
            )
        
        if any(ratio <= 0 for ratio in value):
            raise ConfigValidationError("All split ratios must be positive", path)
    
    def _validate_model_names(self, value: List[str], path: str):
        """Validate model names against known models."""
        known_models = {
            'arima', 'lstm', 'svr', 'theta', 'xgboost', 'prophet', 'random_forest',
            'seasonal_naive', 'tabpfn', 'deepar', 'chronos', 'moment', 'lagllama',
            'toto', 'moirai', 'moirai_moe', 'tiny_time_mixer', 'timesfm',
            'croston_classic', 'exponential_smoothing'
        }
        
        unknown_models = [model for model in value if model not in known_models]
        if unknown_models:
            raise ConfigValidationError(
                f"Unknown models: {unknown_models}. Known models: {sorted(known_models)}", 
                path
            )
    
    def _validate_model_parameters(self, value: Dict[str, Any], path: str):
        """Validate model parameters against their schemas."""
        if not isinstance(value, dict):
            raise ConfigValidationError(f"Expected dict, got {type(value).__name__}", path)
        
        # Check that all models in the name list have parameters
        # This would require access to the full config, so we'll do basic validation here
        
        for model_name, model_params in value.items():
            model_path = f"{path}.{model_name}"
            
            if not isinstance(model_params, dict):
                raise ConfigValidationError(
                    f"Model parameters must be a dict, got {type(model_params).__name__}", 
                    model_path
                )
            
            # Validate against known schemas if available
            if model_name in self.model_parameter_schemas:
                self._validate_model_specific_params(model_params, model_name, model_path)
    
    def _validate_model_specific_params(self, params: Dict[str, Any], model_name: str, path: str):
        """Validate parameters for a specific model."""
        schema = self.model_parameter_schemas[model_name]
        
        # First check for missing required parameters
        for param_name, param_schema in schema.items():
            if param_schema.get('required', False) and param_name not in params:
                raise ConfigValidationError(f"Required parameter '{param_name}' is missing", f"{path}.{param_name}")
        
        # Then validate existing parameters
        for param_name, param_value in params.items():
            if param_name not in schema:
                logger.warning(f"Unknown parameter '{param_name}' for model '{model_name}' in {path}")
                continue
            
            param_schema = schema[param_name]
            param_path = f"{path}.{param_name}"
            
            # Validate parameter value
            self._validate_field(param_value, param_schema, param_path)
    
    def _validate_metrics(self, value: List[str], path: str):
        """Validate metrics against evaluation type."""
        # This would require access to the evaluation.type field
        # For now, we'll do basic validation
        allowed_metrics = set()
        for metrics_list in self.allowed_metrics.values():
            allowed_metrics.update(metrics_list)
        
        unknown_metrics = [metric for metric in value if metric not in allowed_metrics]
        if unknown_metrics:
            raise ConfigValidationError(
                f"Unknown metrics: {unknown_metrics}. Allowed metrics: {sorted(allowed_metrics)}", 
                path
            )


def validate_config_file(config_path: str) -> bool:
    """
    Convenience function to validate a config file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        True if validation passes
        
    Raises:
        ConfigValidationError: If validation fails
        FileNotFoundError: If config file doesn't exist
    """
    validator = ConfigValidator()
    return validator.validate_config_file(config_path)


def validate_config_dict(config: Dict[str, Any]) -> bool:
    """
    Convenience function to validate a config dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if validation passes
        
    Raises:
        ConfigValidationError: If validation fails
    """
    validator = ConfigValidator()
    return validator.validate_config(config)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python config_validator.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        validate_config_file(config_file)
        print(f"✅ Configuration file '{config_file}' is valid!")
        sys.exit(0)
    except ConfigValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
