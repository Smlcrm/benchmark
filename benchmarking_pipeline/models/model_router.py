"""
Model Router for handling model_name:variant format and routing to appropriate folders.

This module provides intelligent routing for models based on their location and variant type.
"""

import os
from typing import Tuple, Dict, Any, Optional
from pathlib import Path


class ModelRouter:
    """
    Routes model requests to appropriate folders based on model_name:variant format.
    
    Handles three cases:
    1. anyvariate models: Route to univariate/multivariate implementation based on user choice
    2. non-anyvariate models: Route directly to specified variant folder
    3. legacy models: Route to univariate folder (backward compatibility)
    """
    
    def __init__(self):
        # Define which models are truly anyvariate (can handle both cases)
        self.anyvariate_models = {
            'chronos', 'moment', 'lagllama', 'toto', 'moirai', 'moirai_moe', 
            'tiny_time_mixer', 'timesfm', 'croston_classic', 'exponential_smoothing'
        }
        
        # Define which models have separate multivariate implementations
        self.multivariate_models = {
            'arima', 'lstm', 'svr', 'theta', 'xgboost'
        }
        
        # Define which models are univariate-only
        self.univariate_models = {
            'prophet', 'random_forest', 'seasonal_naive', 'tabpfn', 'deepar'
        }
    
    def parse_model_spec(self, model_spec: str) -> Tuple[str, str]:
        """
        Parse model specification in format 'model_name:variant'.
        
        Args:
            model_spec: Model specification string (e.g., 'arima:multivariate', 'chronos:univariate')
            
        Returns:
            Tuple of (model_name, variant)
            
        Examples:
            >>> router = ModelRouter()
            >>> router.parse_model_spec('arima:multivariate')
            ('arima', 'multivariate')
            >>> router.parse_model_spec('chronos:univariate')
            ('chronos', 'univariate')
            >>> router.parse_model_spec('prophet')
            ('prophet', 'univariate')
        """
        if ':' in model_spec:
            model_name, variant = model_spec.split(':', 1)
            return model_name.strip(), variant.strip()
        else:
            # Legacy format: default to univariate
            return model_spec.strip(), 'univariate'
    
    def get_model_path(self, model_name: str, variant: str) -> Tuple[str, str, str]:
        """
        Get the appropriate model path, file name, and class name.
        
        Args:
            model_name: Name of the model (e.g., 'arima', 'chronos')
            variant: Variant type ('univariate', 'multivariate')
            
        Returns:
            Tuple of (folder_path, file_name, class_name)
            
        Examples:
            >>> router = ModelRouter()
            >>> router.get_model_path('arima', 'multivariate')
            ('benchmarking_pipeline/models/multivariate/arima', 'arima_model', 'ArimaModel')
            >>> router.get_model_path('chronos', 'univariate')
            ('benchmarking_pipeline/models/anyvariate/chronos', 'chronos_model', 'ChronosModel')
        """
        # Handle anyvariate models
        if model_name in self.anyvariate_models:
            folder_path = f"benchmarking_pipeline/models/anyvariate/{model_name}"
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name)
            return folder_path, file_name, class_name
        
        # Handle models with separate multivariate implementations
        elif model_name in self.multivariate_models:
            if variant == 'multivariate':
                folder_path = f"benchmarking_pipeline/models/multivariate/{model_name}"
            else:
                folder_path = f"benchmarking_pipeline/models/univariate/{model_name}"
            
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name)
            return folder_path, file_name, class_name
        
        # Handle univariate-only models
        elif model_name in self.univariate_models:
            if variant == 'multivariate':
                raise ValueError(f"Model '{model_name}' does not support multivariate variant. "
                               f"Available variants: univariate")
            
            folder_path = f"benchmarking_pipeline/models/univariate/{model_name}"
            file_name = f"{model_name}_model"
            class_name = self._generate_class_name(model_name)
            return folder_path, file_name, class_name
        
        # Handle special cases
        elif model_name in ['foundation_model', 'base_model']:
            folder_path = "benchmarking_pipeline/models"
            file_name = model_name
            class_name = self._generate_class_name(model_name)
            return folder_path, file_name, class_name
        
        else:
            raise ValueError(f"Unknown model '{model_name}'. Available models: "
                           f"{sorted(self.anyvariate_models | self.multivariate_models | self.univariate_models | {'foundation_model', 'base_model'})}")
    
    def _generate_class_name(self, model_name: str) -> str:
        """
        Generate the class name from model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Class name in PascalCase
        """
        # Handle special cases
        if model_name == 'foundation_model':
            return 'FoundationModel'
        elif model_name == 'base_model':
            return 'BaseModel'
        elif model_name == 'moirai_moe':
            return 'MoiraiMoEModel'
        elif model_name == 'tiny_time_mixer':
            return 'TinyTimeMixerModel'
        elif model_name == 'seasonal_naive':
            return 'SeasonalNaiveModel'
        elif model_name == 'random_forest':
            return 'RandomForestModel'
        elif model_name == 'exponential_smoothing':
            return 'ExponentialSmoothingModel'
        elif model_name == 'croston_classic':
            return 'CrostonClassicModel'
        elif model_name == 'tabpfn':
            return 'TabPFNModel'
        elif model_name == 'deepar':
            return 'DeepARModel'
        elif model_name == 'timesfm':
            return 'TimesFMModel'
        else:
            # Default: capitalize first letter and add 'Model' suffix
            return f"{model_name.title()}Model"
    
    def validate_model_spec(self, model_spec: str) -> bool:
        """
        Validate that a model specification is valid.
        
        Args:
            model_spec: Model specification string
            
        Returns:
            True if valid, False otherwise
        """
        try:
            model_name, variant = self.parse_model_spec(model_spec)
            
            # Check if variant is valid
            if variant not in ['univariate', 'multivariate']:
                return False
            
            # Check if model supports the requested variant
            if model_name in self.univariate_models and variant == 'multivariate':
                return False
            
            # Check if folder exists
            folder_path, _, _ = self.get_model_path(model_name, variant)
            return os.path.exists(folder_path)
            
        except (ValueError, Exception):
            return False
    
    def get_available_models(self) -> Dict[str, list]:
        """
        Get all available models and their supported variants.
        
        Returns:
            Dictionary mapping model categories to lists of models
        """
        return {
            'anyvariate': sorted(self.anyvariate_models),
            'multivariate': sorted(self.multivariate_models),
            'univariate_only': sorted(self.univariate_models),
            'base_models': ['foundation_model', 'base_model']
        }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        info = {
            'name': model_name,
            'category': None,
            'supported_variants': [],
            'folder_paths': {}
        }
        
        if model_name in self.anyvariate_models:
            info['category'] = 'anyvariate'
            info['supported_variants'] = ['univariate', 'multivariate']
            info['folder_paths'] = {
                'anyvariate': f"benchmarking_pipeline/models/anyvariate/{model_name}"
            }
        elif model_name in self.multivariate_models:
            info['category'] = 'multivariate'
            info['supported_variants'] = ['univariate', 'multivariate']
            info['folder_paths'] = {
                'univariate': f"benchmarking_pipeline/models/univariate/{model_name}",
                'multivariate': f"benchmarking_pipeline/models/multivariate/{model_name}"
            }
        elif model_name in self.univariate_models:
            info['category'] = 'univariate_only'
            info['supported_variants'] = ['univariate']
            info['folder_paths'] = {
                'univariate': f"benchmarking_pipeline/models/univariate/{model_name}"
            }
        elif model_name in ['foundation_model', 'base_model']:
            info['category'] = 'base_model'
            info['supported_variants'] = ['base']
            info['folder_paths'] = {
                'base': f"benchmarking_pipeline/models/{model_name}"
            }
        
        return info


# Global router instance
model_router = ModelRouter()
