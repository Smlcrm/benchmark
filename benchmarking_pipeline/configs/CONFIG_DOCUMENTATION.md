# Configuration Documentation

This document provides comprehensive documentation for the benchmarking pipeline configuration system.

## Overview

The benchmarking pipeline uses YAML configuration files to control all aspects of the benchmarking process, including:
- Dataset selection and preprocessing
- Model selection and hyperparameters
- Evaluation metrics and validation strategies
- Training parameters and optimization settings

## Configuration Structure

### Top-Level Configuration

```yaml
test_type: deterministic          # Type of test (deterministic, probabilistic)
tensorboard: true                # Enable TensorBoard logging
dataset:                         # Dataset configuration
  # ... dataset settings
model:                          # Model configuration
  # ... model settings
evaluation:                     # Evaluation configuration
  # ... evaluation settings
```

### Dataset Configuration

```yaml
dataset:
  name: china_air_quality       # Dataset name (for reference)
  path: benchmarking_pipeline/datasets/china_air_quality  # Path to dataset
  frequency: H                   # Data frequency (H=hourly, D=daily, M=monthly)
  forecast_horizon: [10, 25, 50] # Forecast horizons to test
  split_ratio: [0.8, 0.1, 0.1]  # Train/validation/test split ratios
  normalize: false               # Whether to normalize data
  handle_missing: interpolate    # Missing value strategy (interpolate, delete, forward_fill)
  validate_ranges: false         # Enable additional data range validation (optional)
  chunks: 2                      # Number of data chunks to load
```

#### Frequency Options

- `H`: Hourly
- `D`: Daily
- `W`: Weekly
- `M`: Monthly
- `Q`: Quarterly
- `Y`: Yearly
- `15T`: 15 minutes
- `30T`: 30 minutes

#### Missing Value Strategies

- `interpolate`: Linear interpolation between known values
- `delete`: Remove rows with missing values
- `forward_fill`: Use previous value to fill missing values
- `backward_fill`: Use next value to fill missing values
- `mean`: Fill with mean of the series
- `median`: Fill with median of the series

#### Data Quality Validation

The pipeline automatically validates data quality before passing it to models:

- **NaN Detection**: Ensures no NaN values remain after preprocessing
- **Infinite Values**: Checks for division by zero or overflow issues
- **Data Types**: Verifies all target data is numeric
- **Empty Data**: Prevents empty datasets from reaching models
- **Constant Data**: Detects columns with no variance (forecasting issues)
- **Range Validation**: Optional additional validation for data ranges and variance

This enforces the contract: "no NaNs or inconsistencies when data gets to models".

### Model Configuration

The model configuration section supports both traditional models and foundation models. Each model type has its own configuration block.

#### Foundation Models

Foundation models are large pre-trained models that can be fine-tuned for specific tasks.

##### Chronos

```yaml
chronos:
  model_size: ['small', 'base', 'large']  # Model size options
  context_length: [8, 16, 32]            # Number of past time steps for context
  num_samples: [5, 10, 20]               # Number of predictive samples
  batch_size: [8, 16]                    # Batch size for inference
  learning_rate: [0.0001, 0.00001]       # Learning rate for fine-tuning
  fine_tune_epochs: [0, 1, 2]            # Number of fine-tuning epochs
```

**Model Sizes**: `tiny`, `mini`, `small`, `base`, `large`

##### LagLlama

```yaml
lagllama:
  context_length: [4, 8, 16]             # Context length for the model
  num_samples: [5, 10, 20]               # Number of samples to generate
  batch_size: [4, 8, 16]                 # Batch size for processing
```

##### Moirai

```yaml
moirai:
  size: ["small", "base", "large"]        # Model size
  psz: [16, 32, 64]                      # Patch size
  bsz: [8, 16, 32]                       # Batch size
  num_samples: [5, 10, 20]               # Number of samples
```

##### TimesFM

```yaml
timesfm:
  per_core_batch_size: [2, 4]            # Batch size per core
  horizon_len: [20, 40]                  # Forecast horizon length
  num_layers: [2, 4]                     # Number of layers
  context_len: [2, 4]                    # Context length
  use_positional_embedding: [False, True] # Whether to use positional embeddings
```

##### Toto

```yaml
toto:
  num_samples: [20, 40, 80]              # Number of samples
  samples_per_batch: [20, 40, 80]        # Samples per batch
```

#### Traditional Models

Traditional models are statistical and machine learning models that require training from scratch.

##### ARIMA

```yaml
arima:
  p: [0, 1, 2]                           # AR order (autoregressive)
  d: [0, 1]                              # Differencing order (integration)
  q: [0, 1, 2]                           # MA order (moving average)
  s: [2, 4, 12]                          # Seasonality period
  maxlags: [10, 20]                      # Maximum lags for auto-selection
  training_loss: ["mae", "mse"]           # Loss function for training
```

##### LSTM

```yaml
lstm:
  units: [32, 64, 128]                   # Number of LSTM units
  layers: [1, 2, 3]                      # Number of LSTM layers
  dropout: [0.1, 0.2, 0.3]              # Dropout rate
  learning_rate: [0.01, 0.001, 0.0001]   # Learning rate
  batch_size: [16, 32, 64]               # Batch size
  epochs: [20, 50, 100]                  # Training epochs
  sequence_length: [10, 20, 50]          # Input sequence length
  training_loss: ["mae", "mse"]           # Loss function
```

##### XGBoost

```yaml
xgboost:
  lookback_window: [50, 100, 200]        # Number of past values to use as features
  n_estimators: [50, 100, 200]           # Number of boosting rounds
  max_depth: [5, 6, 10]                  # Maximum tree depth
  learning_rate: [0.1, 0.3, 0.5]        # Learning rate
  subsample: [0.8, 1.0]                  # Subsample ratio
  colsample_bytree: [0.8, 1.0]           # Column subsample ratio
  random_state: [42]                      # Random seed
  n_jobs: [-1]                           # Number of parallel jobs (-1 for all)
```

##### SVR (Support Vector Regression)

```yaml
svr:
  kernel: ["rbf", "linear", "poly"]       # Kernel function
  C: [0.1, 1.0, 10.0]                   # Regularization parameter
  epsilon: [0.01, 0.1, 0.2]              # Epsilon in epsilon-SVR
  gamma: ["scale", "auto"]                # Kernel coefficient
  lookback_window: [50, 100, 200]        # Number of past values to use as features
  max_iter: [1000, 2000]                 # Maximum iterations
  random_state: [42]                      # Random seed
```

##### Prophet

```yaml
prophet:
  changepoint_prior_scale: [0.001, 0.01, 0.1]  # Flexibility of trend
  seasonality_prior_scale: [0.01, 0.1, 1.0]    # Flexibility of seasonality
  holidays_prior_scale: [0.01, 0.1, 1.0]       # Flexibility of holidays
  seasonality_mode: ["additive", "multiplicative"] # Seasonality mode
  changepoint_range: [0.8, 0.9]                 # Range of changepoints
```

##### Random Forest

```yaml
random_forest:
  n_estimators: [50, 100, 200]           # Number of trees
  max_depth: [5, 10, None]                # Maximum tree depth
  min_samples_split: [2, 5, 10]           # Minimum samples to split
  min_samples_leaf: [1, 2, 4]             # Minimum samples per leaf
  lookback_window: [50, 100, 200]         # Number of past values to use as features
  random_state: [42]                      # Random seed
  n_jobs: [-1]                            # Number of parallel jobs
```

##### Theta

```yaml
theta:
  theta: [1.0, 2.0, 3.0]                 # Theta parameter
  seasonality: [1, 2, 4, 12]             # Seasonality period
  deseasonalize: [True, False]            # Whether to deseasonalize
  training_loss: ["mae", "mse"]           # Loss function
```

##### DeepAR

```yaml
deepar:
  hidden_size: [32, 64, 128]              # Hidden layer size
  num_layers: [1, 2, 3]                   # Number of layers
  dropout: [0.1, 0.2, 0.3]               # Dropout rate
  learning_rate: [0.001, 0.0001]          # Learning rate
  batch_size: [16, 32, 64]                # Batch size
  epochs: [20, 50, 100]                   # Training epochs
  context_length: [10, 20, 50]            # Context length
```

##### TabPFN

```yaml
tabpfn:
  N_ensemble_configurations: [16, 32, 64] # Number of ensemble configurations
  device: ["cpu", "cuda"]                  # Device to use
  random_state: [42]                       # Random seed
```

### Evaluation Configuration

```yaml
evaluation:
  type: deterministic                      # Evaluation type (deterministic, probabilistic)
  metrics: [mae, rmse, mape, smape]       # Metrics to compute
  validation_strategy: "holdout"           # Validation strategy
  cross_validation_folds: 5                # Number of CV folds (if applicable)
  test_size: 0.2                           # Test set size (if applicable)
```

#### Available Metrics

**Point Forecast Metrics:**
- `mae`: Mean Absolute Error
- `rmse`: Root Mean Square Error
- `mape`: Mean Absolute Percentage Error
- `smape`: Symmetric Mean Absolute Percentage Error
- `mase`: Mean Absolute Scaled Error

**Probabilistic Metrics:**
- `crps`: Continuous Ranked Probability Score
- `interval_score`: Interval score for prediction intervals
- `quantile_loss`: Quantile loss for quantile forecasts

**Distribution Metrics:**
- `log_loss`: Log loss for probabilistic forecasts
- `brier_score`: Brier score for probabilistic forecasts

## Configuration Examples

### Minimal Configuration

```yaml
test_type: deterministic
dataset:
  name: test_dataset
  path: datasets/test
  frequency: D
  forecast_horizon: 10
  split_ratio: [0.8, 0.1, 0.1]
  chunks: 1

model:
  arima:
    p: [1]
    d: [1]
    q: [1]

evaluation:
  type: deterministic
  metrics: [mae, mase, rmse]
```

### Comprehensive Configuration

```yaml
test_type: deterministic
tensorboard: true

dataset:
  name: china_air_quality
  path: benchmarking_pipeline/datasets/china_air_quality
  frequency: H
  forecast_horizon: [10, 25, 50]
  split_ratio: [0.8, 0.1, 0.1]
  normalize: false
  handle_missing: interpolate
  chunks: 3

model:
  # Foundation models
  chronos:
    model_size: ['small', 'base']
    context_length: [8, 16]
    num_samples: [5, 10]
    batch_size: [8, 16]
    learning_rate: [0.0001, 0.00001]
    fine_tune_epochs: [0, 1]
  
  lagllama:
    context_length: [4, 8]
    num_samples: [5, 10]
    batch_size: [4, 8]
  
  # Traditional models
  arima:
    p: [0, 1, 2]
    d: [0, 1]
    q: [0, 1, 2]
    s: [2, 4, 12]
    maxlags: [10, 20]
    training_loss: ["mae", "mse"]
  
  lstm:
    units: [32, 64]
    layers: [1, 2]
    dropout: [0.1, 0.2]
    learning_rate: [0.01, 0.001]
    batch_size: [32, 64]
    epochs: [20, 50]
    sequence_length: [10, 20]
    training_loss: ["mae", "mse"]
  
  xgboost:
    lookback_window: [50, 100]
    n_estimators: [50, 100]
    max_depth: [5, 6, 10]
    learning_rate: [0.1, 0.3]
    subsample: [0.8, 1.0]
    colsample_bytree: [0.8, 1.0]
    random_state: [42]
    n_jobs: [-1]

evaluation:
  type: deterministic
  metrics: [mae, rmse, mape, smape]
  validation_strategy: "holdout"
  test_size: 0.2
```

## Configuration Best Practices

### 1. Model Selection

- **Start simple**: Begin with basic models (ARIMA, Seasonal Naive) to establish baselines
- **Progressive complexity**: Add more sophisticated models (LSTM, XGBoost) incrementally
- **Foundation models**: Use foundation models for comparison, but be aware of computational costs

### 2. Hyperparameter Tuning

- **Grid search**: Use small grids initially to identify promising regions
- **Random search**: For high-dimensional spaces, random search often outperforms grid search
- **Bayesian optimization**: Consider advanced optimization for expensive models

### 3. Data Configuration

- **Forecast horizons**: Test multiple horizons to understand model performance across different time scales
- **Split ratios**: Use appropriate ratios (typically 0.8/0.1/0.1 for train/val/test)
- **Chunks**: Start with few chunks for development, increase for final evaluation

### 4. Evaluation Strategy

- **Multiple metrics**: Use several metrics to get a comprehensive view of performance
- **Validation strategy**: Choose appropriate validation strategy based on data characteristics
- **Statistical significance**: Consider statistical tests for comparing model performance

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch sizes or number of chunks
2. **Long training times**: Start with smaller hyperparameter grids
3. **Poor performance**: Check data preprocessing and model assumptions
4. **Configuration errors**: Validate YAML syntax and parameter names

### Validation

The configuration system includes validation to catch common errors:
- Parameter type checking
- Valid value ranges
- Required parameter validation
- Cross-parameter consistency checks

## Advanced Configuration

### Conditional Configuration

```yaml
model:
  arima:
    p: [0, 1, 2]
    d: [0, 1]
    q: [0, 1, 2]
    # Conditional parameters based on d
    s: [2, 4]  # Only used when d > 0
```

### Environment-Specific Configuration

```yaml
# Development configuration
development:
  dataset:
    chunks: 1
  model:
    arima:
      p: [1]
      d: [1]
      q: [1]

# Production configuration
production:
  dataset:
    chunks: 10
  model:
    arima:
      p: [0, 1, 2]
      d: [0, 1]
      q: [0, 1, 2]
```

### Dynamic Configuration

```yaml
# Use environment variables
model:
  chronos:
    batch_size: ${CHRONOS_BATCH_SIZE:-8}
    learning_rate: ${CHRONOS_LR:-0.0001}
```

## References

- [YAML Specification](https://yaml.org/spec/)
- [Hyperparameter Tuning Best Practices](https://arxiv.org/abs/2103.07545)
- [Time Series Forecasting Evaluation](https://robjhyndman.com/papers/forecasting.pdf)
- [Foundation Models for Time Series](https://arxiv.org/abs/2310.13525) 