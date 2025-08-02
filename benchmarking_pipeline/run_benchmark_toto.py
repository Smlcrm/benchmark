"""
Main script for orchestrating the end-to-end benchmarking pipeline.
"""

from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
#from benchmarking_pipeline.pipeline.logger import Logger
from torch.utils.tensorboard import SummaryWriter
import time
from benchmarking_pipeline.models.toto.toto_model import TotoModel
from benchmarking_pipeline.trainer.foundation_model_tuning import FoundationModelTuner
import numpy as np
import pandas as pd
import re
import os
import json
import datetime
#import matplotlib.pyplot as plt
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
        #self.logger = Logger(config)
        
    def run(self):
        """Execute the end-to-end benchmarking pipeline."""
        # Determine config file name for logging
        config_file_name = os.path.splitext(os.path.basename(self.config_path))[0] if self.config_path else 'unknown_config'
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=f"runs/benchmark_runner_{config_file_name}_{timestamp}")
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
        all_dataset_chunks = data_loader.load_several_chunks(num_chunks)
        preprocessor = Preprocessor({"dataset": {"normalize": dataset_cfg.get('normalize', False)}})
        all_dataset_chunks = [preprocessor.preprocess(chunk).data for chunk in all_dataset_chunks]
        print(f"[DEBUG] Number of chunks: {len(all_dataset_chunks)}")
        if len(all_dataset_chunks) > 0:
            print(f"[DEBUG] First chunk train shape: {all_dataset_chunks[0].train.features.shape}")
        # Load config
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
        #self.logger.log_metrics({"status": "Pipeline completed"}, step=0)
  
def run_toto(all_dataset_chunks, writer=None, config=None, config_path=None):

    dataset_name = config['dataset']['name']
    toto_params = config['model']['parameters']['Toto']
    model_params = {k: v[0] if isinstance(v, list) else v for k, v in toto_params.items()}


    toto_model = TotoModel(model_params)

    hyper_grid = {k: v for k, v in toto_params.items() if isinstance(v, list)}

    moirai_hyperparameter_tuner = FoundationModelTuner(toto_model, hyper_grid, False)
    validation_score_hyperparameter_tuple = moirai_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
    results = moirai_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Toto results: {results}")

    print("TOTO WORKS!")



def _extract_number_before_capital(freq_str):
    match = re.match(r'(\d+)?[A-Z]', freq_str)
    if match:
        return int(match.group(1)) if match.group(1) else 1
    else:
        raise ValueError(f"Invalid frequency string: {freq_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking pipeline with specified config file.")
    parser.add_argument('--config', type=str, default='benchmarking_pipeline/configs/univariate_config.yaml', help='Path to the config YAML file')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    runner = BenchmarkRunner(config=config, config_path=config_path)
    runner.run() 