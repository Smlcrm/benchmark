# Benchmarking Pipeline

A flexible and modular benchmarking pipeline for machine learning models with comprehensive support for both univariate and multivariate time series forecasting.

## Project Structure

```
benchmarking_pipeline/
├── configs/                # YAML/JSON files for experiment configuration (models, datasets, hyperparameters)
├── datasets/               # Datasets (organized in chunks for time series experiments)
├── models/                 # Model implementations (each model as a Python class)
│   ├── multivariate/           # Multivariate model implementations
│   │   ├── multivariate_svr_model.py      # Multivariate SVR with MultiOutputRegressor
│   │   ├── multivariate_xgboost_model.py  # Multivariate XGBoost with gradient boosting
│   │   └── lstm_model.py                  # Multivariate LSTM with flattened output
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
├── test_multivariate_svr_beijing.py       # Test suite for Multivariate SVR
├── test_multivariate_xgboost_beijing.py   # Test suite for Multivariate XGBoost
└── ...                      # Other utility scripts and files
```

### Key Files Explained

- **run_benchmark.py**: Main entry point for running the benchmarking pipeline. Loads config, runs models, logs results.
- **test_plot.py**: Standalone script to visualize a dataset (after preprocessing) for quick inspection.
- **configs/**: Contains YAML/JSON files specifying which models to run, their hyperparameters, and dataset details.
- **models/**: Contains Python classes for each supported model. Each model implements a common interface for training and prediction.
- **models/multivariate/**: Specialized implementations for multivariate time series forecasting with cross-series dependency modeling.
- **pipeline/data_loader.py**: Loads datasets, handles chunking, and splits into train/val/test.
- **pipeline/preprocessor.py**: Handles normalization, missing value imputation, and outlier removal.
- **pipeline/feature_extraction.py**: Creates features for ML models (e.g., lagged values for XGBoost, Random Forest).
- **pipeline/trainer.py**: Contains logic for training models on the data.
- **pipeline/evaluator.py**: Evaluates model predictions and computes metrics.
- **pipeline/logger.py**: Handles logging of metrics and results (e.g., to TensorBoard).
- **trainer/hyperparameter_tuning.py**: Implements grid/random search for hyperparameter optimization and final evaluation.

For more details, see the docstrings in each file or explore the example configs in `benchmarking_pipeline/configs/`.

## Multivariate Models

The pipeline supports advanced multivariate time series forecasting with comprehensive implementations for handling multiple target variables simultaneously.

### Available Multivariate Models

#### **MultivariateLSTM** (`models/multivariate/lstm_model.py`)

- **Architecture**: Extends univariate LSTM with flattened output layer predicting `forecast_horizon × n_targets` values
- **Technique**: Uses same LSTM architecture but with multivariate input/output dimensions
- **Cross-Dependencies**: Learns relationships between targets through shared hidden states
- **Library**: TensorFlow/Keras

#### **MultivariateSVR** (`models/multivariate/multivariate_svr_model.py`)

- **Architecture**: Uses `MultiOutputRegressor` wrapper around sklearn's SVR
- **Feature Engineering**: Flattened lag features from all target variables `(lookback_window × n_targets)`
- **Cross-Dependencies**: Learns through comprehensive feature combinations across all targets
- **Preprocessing**: Integrated StandardScaler for SVR sensitivity requirements
- **Library**: scikit-learn

#### **MultivariateXGBoost** (`models/multivariate/multivariate_xgboost_model.py`)

- **Architecture**: Uses `MultiOutputRegressor` wrapper around XGBRegressor
- **Feature Engineering**: Simplified lag-based features consistent with SVR approach
- **Cross-Dependencies**: Gradient boosting learns complex non-linear multivariate relationships
- **Performance**: Handles large feature spaces efficiently with ensemble learning
- **Library**: xgboost

### Key Multivariate Features

- **Cross-Series Dependencies**: All models learn relationships between multiple time series simultaneously
- **Unified Interface**: Same API as univariate models (`train()`, `predict()`, `compute_loss()`) for seamless integration
- **Advanced Feature Engineering**: Sophisticated lag features and statistical representations across all target variables
- **Rolling Prediction**: Autoregressive forecasting that maintains multivariate structure throughout prediction horizon
- **Pipeline Integration**: Full compatibility with existing benchmarking infrastructure and configuration system
- **Robust Error Handling**: Comprehensive validation, NaN handling, and graceful degradation mechanisms

### Multivariate Model Testing

Each multivariate model includes comprehensive test suites validated on real datasets:

```bash
# Test MultivariateSVR with Beijing Subway dataset
python test_multivariate_svr_beijing.py

# Test MultivariateXGBoost with Beijing Subway dataset
python test_multivariate_xgboost_beijing.py
```

**Test Coverage**:

- Real multivariate dataset validation (Beijing Subway ridership)
- Full pipeline testing: training → prediction → loss computation → save/load
- Shape validation ensuring `(forecast_steps, n_targets)` output consistency
- Edge case handling and error condition validation
- Parameter management and dynamic configuration updates

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Smlcrm/benchmark.git
cd benchmark
```

2. Install dependencies:

```bash
pip install -r requirements.txt

# Additional dependencies for specific models:
conda install -c conda-forge xgboost  # For XGBoost (macOS users)
pip install statsmodels              # For statistical models
```

## Usage

### Basic Benchmarking

1. Create or modify a configuration file in the `configs/` directory.

2. Run the benchmark using the CLI:

```bash
python benchmarking_pipeline/cli.py --config configs/default_config.yaml
```

### Multivariate Forecasting

1. Update your config to include multivariate models:

```yaml
model:
  name: ["lstm", "svr", "xgboost"] # Include multivariate models
  parameters:
    SVR:
      kernel: ["rbf", "linear"]
      C: [0.1, 1.0, 10.0]
      lookback_window: [50, 100]
      forecast_horizon: [50]
    XGBoost:
      n_estimators: [50, 100]
      max_depth: [3, 6]
      lookback_window: [50, 100]
      forecast_horizon: [50]
```

2. Run multivariate benchmarking:

```bash
python benchmarking_pipeline/run_benchmark.py --config benchmarking_pipeline/configs/multivariate_forecast_horizon_config.yaml
```

## How to Run the Benchmarking Pipeline

To run the benchmarking pipeline with a specific configuration file:

```bash
python benchmarking_pipeline/run_benchmark.py --config benchmarking_pipeline/configs/your_config.yaml
```

- Replace `your_config.yaml` with the path to your desired YAML config file.
- The config file controls which models are run, their hyperparameters, and the dataset used.
- Results and TensorBoard logs will be saved in the `runs/` and `results/` directories.

### Multivariate-Specific Configuration

For multivariate forecasting, ensure your config includes:

- `num_targets: N` in the dataset section
- Appropriate `lookback_window` and `forecast_horizon` parameters
- Model-specific hyperparameters for multivariate implementations

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

This will display a plot of the full time series after preprocessing, supporting both univariate and multivariate visualizations.

---

For more details on configuration options and model-specific settings, see the example YAML files in `benchmarking_pipeline/configs/`.

## Configuration

The pipeline is configured using YAML files in the `configs/` directory. See `configs/default_config.yaml` for an example configuration.

### Multivariate Configuration Example

```yaml
test_type: deterministic
dataset:
  name: BEIJING_SUBWAY_30MIN
  path: benchmarking_pipeline/datasets/BEIJING_SUBWAY_30MIN
  frequency: 30T
  forecast_horizon: 50
  split_ratio: [0.8, 0.1, 0.1]
  normalize: true
  num_targets: 2

model:
  name: ["svr", "xgboost"]
  parameters:
    SVR:
      kernel: ["rbf", "linear"]
      C: [0.1, 1.0, 10.0]
      epsilon: [0.01, 0.1, 0.2]
      lookback_window: [50, 100]
      forecast_horizon: [50]
    XGBoost:
      n_estimators: [50, 100]
      max_depth: [3, 6]
      learning_rate: [0.1, 0.3]
      lookback_window: [50, 100]
      forecast_horizon: [50]

evaluation:
  type: deterministic
  metrics: [mae, rmse]
```

## Features

- **Flexible Model Benchmarking**: Support for both univariate and multivariate time series models
- **Comprehensive Multivariate Support**: Advanced implementations for SVR, XGBoost, and LSTM with cross-series dependency modeling
- **HuggingFace Dataset Integration**: Seamless dataset loading and preprocessing
- **Modular Preprocessing**: Configurable normalization, missing value handling, and feature extraction
- **Advanced Feature Engineering**: Lag-based features, statistical representations, and cross-correlation analysis
- **Comprehensive Logging**: TensorBoard integration, CSV exports, and detailed metrics tracking
- **Configurable Training**: Hyperparameter tuning, grid search, and evaluation pipelines
- **Robust Testing**: Comprehensive test suites for all model implementations
- **Rolling Prediction**: Advanced autoregressive forecasting for extended horizons
- **Pipeline Integration**: Unified interface across all model types for seamless comparison

## Model Performance Validation

All multivariate models have been validated on real datasets:

- **MultivariateSVR**: MAE=0.0841, RMSE=0.1467 on Beijing Subway dataset
- **MultivariateXGBoost**: Successful training with simplified feature engineering
- **MultivariateLSTM**: Cross-series dependency learning with flattened output architecture

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/multivariate-implementation`)
3. Commit your changes (`git commit -am 'Add multivariate SVR implementation'`)
4. Push to the branch (`git push origin feature/multivariate-implementation`)
5. Create a Pull Request

### Development Guidelines

- Follow the established `BaseModel` interface for all new model implementations
- Include comprehensive test suites for new models (`test_model_name_dataset.py`)
- Update configuration examples when adding new models
- Maintain consistency in feature engineering approaches across similar models
- Document multivariate-specific parameters and usage patterns

## License

This project is licensed under the MIT License - see the LICENSE file for details.

\
