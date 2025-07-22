import argparse
import matplotlib.pyplot as plt
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
import yaml
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a time series dataset after preprocessing.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--config', type=str, default=None, help='Optional config YAML for preprocessing')
    args = parser.parse_args()

    # Minimal config for DataLoader and Preprocessor
    config = {
        'dataset': {
            'path': args.dataset_path,
            'name': os.path.basename(args.dataset_path),
            'split_ratio': [0.8, 0.1, 0.1],
            'normalize': True,
            'handle_missing': 'interpolate',
            'chunks': 1
        }
    }
    if args.config:
        with open(args.config, 'r') as f:
            user_config = yaml.safe_load(f)
        config['dataset'].update(user_config.get('dataset', {}))

    # Load and preprocess
    data_loader = DataLoader(config)
    dataset_chunks = data_loader.load_several_chunks(1)
    preprocessor = Preprocessor(config)
    preprocessed = preprocessor.preprocess(dataset_chunks[0])
    data = preprocessed.data

    # Concatenate all splits in order
    all_timestamps = np.concatenate([data.train.timestamps, data.validation.timestamps, data.test.timestamps])
    col = data.train.features.columns[0]
    all_values = np.concatenate([
        data.train.features[col].values,
        data.validation.features[col].values,
        data.test.features[col].values
    ])

    plt.figure(figsize=(12, 6))
    plt.plot(all_timestamps, all_values, label='full series')
    plt.title(f"Dataset: {config['dataset']['name']} (full series)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show() 