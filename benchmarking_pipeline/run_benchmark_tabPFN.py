"""
Main script for orchestrating the end-to-end benchmarking pipeline for TabPFN.
"""

from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from torch.utils.tensorboard import SummaryWriter
import time
from benchmarking_pipeline.models.tabpfn.tabpfn_model import TabpfnModel
import numpy as np
import pandas as pd
import os
import datetime
import yaml
import argparse

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
        
    def run(self):
        """Execute the end-to-end benchmarking pipeline."""
        config_file_name = os.path.splitext(os.path.basename(self.config_path))[0] if self.config_path else 'unknown_config'
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=f"runs/benchmark_runner_{config_file_name}_{timestamp}")
        # Load dataset config
        dataset_cfg = self.config['dataset']
        dataset_path = dataset_cfg['path']
        dataset_name = dataset_cfg['name']
        split_ratio = dataset_cfg.get('split_ratio', [0.8, 0.1, 0.1])
        num_chunks = dataset_cfg.get('chunks', 1)
        data_loader = DataLoader({"dataset": {
            "path": dataset_path,
            "name": dataset_name,
            "split_ratio": split_ratio
        }})
        all_dataset_chunks = data_loader.load_several_chunks(num_chunks)
        preprocessor = Preprocessor({"dataset": {"normalize": dataset_cfg.get('normalize', False)}})
        all_dataset_chunks = [preprocessor.preprocess(chunk).data for chunk in all_dataset_chunks]
        print(f"[DEBUG] Number of chunks: {len(all_dataset_chunks)}")
        if len(all_dataset_chunks) > 0:
            print(f"[DEBUG] First chunk train shape: {all_dataset_chunks[0].train.features.shape}")
        config_path = self.config_path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_names = config['model']['name']
        for model_name in model_names:
            func_name = f"run_{model_name.lower()}"
            if func_name in globals():
                globals()[func_name](all_dataset_chunks, writer, self.config, self.config_path)
        writer.close()
        print(f"\nTensorBoard logs saved in 'runs/benchmark_runner_{config_file_name}_{timestamp}/'. To view, run: tensorboard --logdir runs/benchmark_runner\n")

def run_tabpfn(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    tabpfn_params = config['model']['parameters']['TabPFN']
    n_ensemble_configs = tabpfn_params.get('n_ensemble_configs', 32)
    device = tabpfn_params.get('device', 'cpu')
    prediction_length = tabpfn_params.get('forecast_horizon', 5)
    target_col = tabpfn_params.get('target_col', 'y')

    tabpfn_model = TabpfnModel(
        n_ensemble_configs=n_ensemble_configs,
        device=device
    )

    # Use the first chunk for demonstration (adapt as needed for your pipeline)
    chunk = all_dataset_chunks[0]
    if hasattr(chunk.train, "features"):
        df = chunk.train.features
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
    else:
        df = pd.DataFrame(chunk.train.labels)  # fallback if features not present

    # Run prediction
    forecast = tabpfn_model.predict(df, target_col=target_col, prediction_length=prediction_length)
    print(f"TabPFN predictions for {dataset_name}:")
    print(f"  {target_col}: {forecast[:5]} ...")  # show first 5 predictions

    # Optionally log to TensorBoard
    if writer is not None:
        writer.add_scalar(f'TabPFN/{target_col}_first_pred', float(np.asarray(forecast).flatten()[0]), 12)
        # If you have true values for comparison, plot them
        if hasattr(chunk.test, "features"):
            y_true = chunk.test.features[target_col].values.flatten()[:prediction_length]
            y_pred = np.asarray(forecast).flatten()[:prediction_length]
            # Optionally implement log_preds_vs_true if you want to plot
            # log_preds_vs_true(writer, 'TabPFN', y_true, y_pred, 12)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame({target_col: forecast}).to_csv(f'results/TabPFN_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("TabPFN WORKS!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking pipeline with specified config file.")
    parser.add_argument('--config', type=str, default='benchmarking_pipeline/configs/tabPFN_test.yaml', help='Path to the config YAML file')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    runner = BenchmarkRunner(config=config, config_path=config_path)
    runner.run()