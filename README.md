# Benchmarking Pipeline

A flexible and modular benchmarking pipeline for machine learning models.

## Project Structure

```
benchmarking_pipeline/
├── configs/                # YAML/JSON files for experiment configuration (models, datasets, hyperparameters)
├── datasets/               # Datasets (organized in chunks for time series experiments)
├── models/                 # Model implementations (each model as a Python class)
│   ├── arima_model.py          # ARIMA model wrapper
│   ├── croston_classic_model.py# Croston's method for intermittent demand
│   ├── exponential_smoothing_model.py # Exponential Smoothing model
│   ├── lstm_model.py           # LSTM (deep learning) model
│   ├── prophet_model.py        # Prophet model
│   ├── random_forest_model.py  # Random Forest regressor
│   ├── SVR_model.py            # Support Vector Regression
│   ├── theta_model.py          # Theta model (sktime)
│   ├── xgboost_model.py        # XGBoost regressor
│   └── ...                     # Other models
├── pipeline/                # Core pipeline components
│   ├── data_loader.py           # Loads datasets, handles chunking and splits
│   ├── preprocessor.py          # Data preprocessing (normalization, missing values, outliers)
│   ├── feature_extraction.py    # Feature engineering for ML models
│   ├── trainer.py               # Model training logic
│   ├── evaluator.py             # Model evaluation and metrics
│   └── logger.py                # Logging utilities (TensorBoard, CSV, etc.)
├── trainer/                 # Hyperparameter tuning and training utilities
│   └── hyperparameter_tuning.py # Grid/random search, evaluation logic
├── test_plot.py             # Script to quickly plot a dataset after preprocessing
├── run_benchmark.py         # Main script: runs the full benchmarking pipeline
├── cli.py                   # (Optional) Command-line interface for running experiments
└── ...                      # Other utility scripts and files
```

### Key Files Explained

- **run_benchmark.py**: Main entry point for running the benchmarking pipeline. Loads config, runs models, logs results.
- **test_plot.py**: Standalone script to visualize a dataset (after preprocessing) for quick inspection.
- **configs/**: Contains YAML/JSON files specifying which models to run, their hyperparameters, and dataset details.
- **models/**: Contains Python classes for each supported model. Each model implements a common interface for training and prediction.
- **pipeline/data_loader.py**: Loads datasets, handles chunking, and splits into train/val/test.
- **pipeline/preprocessor.py**: Handles normalization, missing value imputation, and outlier removal.
- **pipeline/feature_extraction.py**: Creates features for ML models (e.g., lagged values for XGBoost, Random Forest).
- **pipeline/trainer.py**: Contains logic for training models on the data.
- **pipeline/evaluator.py**: Evaluates model predictions and computes metrics.
- **pipeline/logger.py**: Handles logging of metrics and results (e.g., to TensorBoard).
- **trainer/hyperparameter_tuning.py**: Implements grid/random search for hyperparameter optimization and final evaluation.

For more details, see the docstrings in each file or explore the example configs in `benchmarking_pipeline/configs/`.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Smlcrm/benchmark.git
cd benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Create or modify a configuration file in the `configs/` directory.

2. Run the benchmark using the CLI:
```bash
python benchmarking_pipeline/cli.py --config configs/default_config.yaml
```

## How to Run the Benchmarking Pipeline

To run the benchmarking pipeline with a specific configuration file:

```bash
python benchmarking_pipeline/run_benchmark.py --config benchmarking_pipeline/configs/your_config.yaml
```

- Replace `your_config.yaml` with the path to your desired YAML config file.
- The config file controls which models are run, their hyperparameters, and the dataset used.
- Results and TensorBoard logs will be saved in the `runs/` and `results/` directories.

## How to Plot a Dataset

To quickly visualize a dataset (after preprocessing) using the provided script:

```bash
python test_plot.py --dataset_path benchmarking_pipeline/datasets/your_dataset_folder
```

- Replace `your_dataset_folder` with the path to your dataset directory.
- Optionally, you can pass a custom config for preprocessing:

```bash
python test_plot.py --dataset_path benchmarking_pipeline/datasets/your_dataset_folder --config benchmarking_pipeline/configs/your_config.yaml
```

This will display a plot of the full time series after preprocessing.

---

For more details on configuration options and model-specific settings, see the example YAML files in `benchmarking_pipeline/configs/`.

## Configuration

The pipeline is configured using YAML files in the `configs/` directory. See `configs/default_config.yaml` for an example configuration.

## Features

- Flexible model benchmarking pipeline
- Support for HuggingFace datasets
- Modular preprocessing and feature extraction
- Comprehensive logging and metrics tracking
- Configurable model training and evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
