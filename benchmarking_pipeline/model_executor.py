"""
Model executor for isolated benchmarking runs.

This module provides the ModelExecutor class that runs individual models
in isolated conda environments to avoid dependency conflicts.
It handles model loading, hyperparameter tuning, and evaluation.
"""

import argparse
import yaml
import pickle
import importlib

from benchmarking_pipeline.models.base_model import BaseModel
from benchmarking_pipeline.models.foundation_model import FoundationModel

from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.trainer.foundation_model_tuning import FoundationModelTuner

class ModelExecutor:

    def __init__(self, config, chunk_path, model_folder_name, model_file_name, model_class_name):
        self.config = config
        self.chunk_path = chunk_path
        self.model_folder_name = model_folder_name
        self.model_file_name = model_file_name
        self.model_class_name = model_class_name

    def run(self):
        # Extract the model name from the folder path for parameter lookup
        # The folder path is now like 'benchmarking_pipeline/models/multivariate/arima'
        # We need to extract just 'arima' for parameter lookup
        model_name = self.model_folder_name.split('/')[-1]
        
        # Build the module path for import
        if self.model_folder_name.startswith('benchmarking_pipeline/models/'):
            # Remove the prefix and convert to module path
            relative_path = self.model_folder_name.replace('benchmarking_pipeline/models/', '')
            # Replace forward slashes with dots for proper Python module path
            relative_path = relative_path.replace('/', '.')
            module_path = f"benchmarking_pipeline.models.{relative_path}.{self.model_file_name}"
        else:
            # Fallback for backward compatibility
            module_path = f"benchmarking_pipeline.models.{self.model_folder_name}.{self.model_file_name}"
        
        print(f"[INFO] Importing module: {module_path}")
        module = importlib.import_module(module_path)
        model_class = getattr(module, self.model_class_name)
        print(f"[INFO] Executing {model_name} model...")
        
        with open(self.chunk_path, 'rb') as f:
          all_dataset_chunks = pickle.load(f)
        
        if issubclass(model_class, BaseModel):
            print(f"{model_name} is a Base Model!")
            hyper_grid = config['model']['parameters'][model_name]
            print(f"{model_name} hyper grid: {hyper_grid}")

            model_params = {k: v[0] if isinstance(v, list) else v for k, v in hyper_grid.items()}
            print(f"{model_name} initial model_params: {model_params}")

            base_model = model_class(model_params)

            model_hyperparameter_tuner = HyperparameterTuner(base_model, hyper_grid, False)
            validation_score_hyperparameter_tuple = model_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
            print(f"{model_name} validation score hyperparameter tuple: {validation_score_hyperparameter_tuple}")
            best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
            print(f"{model_name} best hyperparameters dict: {best_hyperparameters_dict}")
            results = model_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)

            print(f"{model_name} results: {results}")
            print(f"[SUCCESS] {model_name} execution completed successfully!")
        elif issubclass(model_class, FoundationModel):
            print(f"{model_name} is a Foundation Model!")
            
            hyper_grid = config['model']['parameters'][model_name]
            model_params = {k: v[0] if isinstance(v, list) else v for k, v in hyper_grid.items()}

            foundation_model = model_class(model_params)

            model_hyperparameter_tuner = FoundationModelTuner(foundation_model, hyper_grid, False)
            validation_score_hyperparameter_tuple = model_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
            best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
            print(f"{model_name} best hyperparameters dict: {best_hyperparameters_dict}")
            results = model_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)

            print(f"{model_name} results: {results}")
            print(f"[SUCCESS] {model_name} execution completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute a single model in an isolated environment for benchmarking.")
    parser.add_argument('--config', type=str, help='Path to the config YAML file')
    parser.add_argument('--chunk_path', type=str, help='Path to the temporary pickle file containing dataset chunks.')
    parser.add_argument('--model_folder_name', type=str, help='Name of the model folder to reference.')
    parser.add_argument('--model_file_name', type=str, help='Name of the model file containing the model class.')
    parser.add_argument('--model_class_name', type=str, help='Name of the model class to instantiate.')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_executor = ModelExecutor(config=config, 
                                   chunk_path=args.chunk_path, 
                                   model_folder_name=args.model_folder_name,
                                   model_file_name=args.model_file_name,
                                   model_class_name=args.model_class_name)
    print(f"[INFO] Config: {args.config}")
    print(f"[INFO] Chunk Path: {args.chunk_path}")
    print(f"[INFO] Model Folder: {args.model_folder_name}")
    print(f"[INFO] Model File: {args.model_file_name}")
    print(f"[INFO] Model Class: {args.model_class_name}")
    model_executor.run() 