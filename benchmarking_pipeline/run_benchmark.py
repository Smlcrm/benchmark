import argparse
import subprocess
import importlib
import os
import sys
import yaml
import json
import pickle
import tempfile

# Save to temp file

from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
import datetime

class BenchmarkRunner:
    
    def __init__(self, config, config_path=None):
        """
        Initialize benchmark runner with configuration.
        Args:
            config: Configuration dictionary for the pipeline
            config_path: Path to the config file used
        """
        self.config = config
        self.config_path = config_path
        
    def _analyze_dataset(self, dataset_chunk):
        """
        Analyze dataset chunk to determine properties for auto-detection.
        
        Args:
            dataset_chunk: Dataset chunk object with train/test data
            
        Returns:
            Dictionary with dataset analysis information
        """
        # Prefer explicit config when available
        num_targets_cfg = self.config.get('dataset', {}).get('num_targets')

        # Use training targets to determine number of target columns
        train_targets = dataset_chunk.train.targets
        if train_targets is None:
            raise ValueError("Dataset analysis failed: train.targets is None")
        if not hasattr(train_targets, 'shape') or len(getattr(train_targets, 'shape')) < 2:
            raise ValueError("Dataset analysis failed: train.targets has no valid 2D shape")

        if isinstance(num_targets_cfg, int) and num_targets_cfg > 0:
            target_columns = num_targets_cfg
        else:
            target_columns = train_targets.shape[1]

        # Basic shape info of targets
        data_shape = train_targets.shape
        total_columns = data_shape[1]

        # More sophisticated detection could be added here
        has_multiple_targets = target_columns > 1
        
        dataset_info = {
            'target_columns': target_columns,
            'total_columns': total_columns,
            'has_multiple_targets': has_multiple_targets,
            'data_shape': data_shape
        }
        
        print(f"[DEBUG] Dataset analysis: shape={dataset_info['data_shape']}, targets={target_columns}")
        return dataset_info
    
    def run(self):
        """Execute the end-to-end benchmarking pipeline."""
        # Determine config file name for logging
        config_file_name = os.path.splitext(os.path.basename(self.config_path))[0] if self.config_path else 'unknown_config'
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Load dataset config
        dataset_cfg = self.config['dataset']
        dataset_path = dataset_cfg['path']
        dataset_name = dataset_cfg['name']
        split_ratio = dataset_cfg.get('split_ratio', [0.8, 0.1, 0.1])

        # Use 'chunks' from config, default to 1 if not present
        num_chunks = dataset_cfg.get('chunks', 1)
        data_loader = DataLoader({"dataset": {
            "path": dataset_path,
            "name": dataset_name,
            "split_ratio": split_ratio
        }})

        # Preprocess all dataset chunks
        all_dataset_chunks = data_loader.load_several_chunks(num_chunks)
        preprocessor = Preprocessor({"dataset": {"normalize": dataset_cfg.get('normalize', False)}})
        all_dataset_chunks = [preprocessor.preprocess(chunk).data for chunk in all_dataset_chunks]

        # Temporary file to access data across different subprocesses
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
          pickle.dump(all_dataset_chunks, tmp)
          chunk_path = tmp.name
          print("[DEBUG] Temporary file path", chunk_path)

        
        with open(chunk_path, 'rb') as f:
          yuh = pickle.load(f)
        print(f"[DEBUG] Number of chunks: {len(all_dataset_chunks)}")
        if len(all_dataset_chunks) > 0:
            first = all_dataset_chunks[0]
            targets_shape = getattr(first.train.targets, 'shape', None)
            features_shape = getattr(first.train.features, 'shape', None) if first.train.features is not None else None
            print(f"[DEBUG] First chunk train targets shape: {targets_shape}")
            print(f"[DEBUG] First chunk train features shape: {features_shape}")

        # Load config
        config_path = self.config_path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_names = config['model']['name']

        # Import the model router
        from benchmarking_pipeline.models.model_router import model_router
        
        # Analyze dataset to determine properties for auto-detection
        print("[INFO] Analyzing dataset for auto-detection...")
        dataset_info = self._analyze_dataset(all_dataset_chunks[0])
        print(f"[INFO] Dataset analysis: {dataset_info}")
        
        # Run each model we have individually
        for model_spec in model_names:
            # Parse the model specification (e.g., 'arima', 'chronos')
            model_name, variant = model_router.parse_model_spec(model_spec)
            
            # Get the appropriate model path with auto-detection
            folder_path, model_file_name, model_class_name = model_router.get_model_path_with_auto_detection(
                model_name, variant, dataset_info
            )
            
            print(f"[INFO] Processing model: {model_spec}")
            print(f"[INFO] Model name: {model_name}, Variant: {variant}")
            print(f"[INFO] Folder path: {folder_path}")
            print(f"[INFO] File name: {model_file_name}")
            print(f"[INFO] Class name: {model_class_name}")
            
            # Get parameters for the base model name (without variant)
            if model_name in config['model']['parameters']:
                model_params = config['model']['parameters'][model_name]
            else:
                print(f"[WARNING] No parameters found for {model_name}, using defaults")
                model_params = {}

            requirements_path = f"{folder_path}/requirements.txt"

            # Create conda environment name based on model name to avoid conflicts
            # The variant is now auto-detected, so we use just the model name
            conda_env_name = model_name

            # Something on my mind: uv could probably make this process a LOT quicker - definitely something to explore.
        
            # Create the conda environment.
            subprocess.run(["conda", "create", "-n", conda_env_name, "python=3.10", "-y"], check=True)
            print(f"[SUCCESS] Conda environment for {conda_env_name} has been made!")

            # Install the dependencies of the corresponding model without activating the model.
            subprocess.run(["conda", "run", "-n", conda_env_name, "pip", "install", "-r", requirements_path], check=True)
            print(f"[SUCCESS] Dependencies for {conda_env_name} have been installed in the proper conda environment!")

            subprocess.run([
                "conda", "run", "-n", conda_env_name, "python3", "-m", "benchmarking_pipeline.model_executor",
                "--config", args.config,
                "--chunk_path", chunk_path,
                "--model_folder_name", folder_path,
                "--model_file_name", model_file_name,
                "--model_class_name", model_class_name
            ], check=True)

            # Get rid of the environment when we're done with the current model.
            subprocess.run(["conda", "remove", "-n", conda_env_name, "--all", "-y"], check=True)
            print(f"[SUCCESS] Conda environment for {conda_env_name} has been deleted!")

            
        os.remove(chunk_path)
        print("All model files ran!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking pipeline with specified config file.")
    parser.add_argument('--config', type=str, default='benchmarking_pipeline/configs/all_model_univariate.yaml', help='Path to the config YAML file')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    runner = BenchmarkRunner(config=config, config_path=config_path)
    runner.run() 