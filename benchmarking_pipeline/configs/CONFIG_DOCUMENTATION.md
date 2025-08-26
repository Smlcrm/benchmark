# Configuration File Documentation

This document explains the configuration file format used in the benchmarking pipeline. The configuration files control all aspects of data processing, model training, and evaluation.

## Overview

The benchmarking pipeline now uses a **consolidated configuration approach** with a single configuration file (`all_model_test.yaml`) that contains all model parameters and settings. This eliminates the need for multiple configuration files and ensures consistency across all models.

## Configuration File Structure

### Root Level Parameters

#### `test_type` *(string, required)*
- **Description**: Specifies the type of time series test being conducted
- **Possible values**: `"univariate"`, `"multivariate"`, `"additive"`, `"multiplicative"`, `"deterministic"`, `"probabilistic"`, `"irregular"`, `"regular"`
- **Used in**: Pipeline orchestration, model selection, result logging
- **Example**: `test_type: deterministic`

---

## `dataset` Section

Controls data loading and preprocessing parameters.

#### `name` *(string, required)*
- **Description**: Name/identifier of the dataset
- **Used in**: Data loading, result metadata, output file naming
- **Example**: `name: australian_electricity_demand`

#### `path` *(string, required)*
- **Description**: Path to the dataset files (relative to project root)
- **Used in**: DataLoader module
- **Example**: `path: benchmarking_pipeline/datasets/australian_electricity_demand`

#### `frequency` *(string, required)*
- **Description**: Time series frequency following pandas frequency codes
- **Possible values**: `"H"` (hourly), `"D"` (daily), `"W"` (weekly), `"M"` (monthly), `"30T"` (30 minutes), etc.
- **Used in**: Data validation, feature engineering, model configuration
- **Example**: `frequency: D`

#### `forecast_horizon` *(integer, required)*
- **Description**: Number of time steps to forecast ahead
- **Used in**: Feature engineering, model training, evaluation
- **Example**: `forecast_horizon: 5`

#### `split_ratio` *(list of floats, required)*
- **Description**: Proportions for train/validation/test splits (must sum to 1.0)
- **Format**: `[train_ratio, validation_ratio, test_ratio]`
- **Used in**: DataLoader for splitting time series data
- **Example**: `split_ratio: [0.8, 0.1, 0.1]`

#### `normalize` *(boolean, required)*
- **Description**: Whether to normalize/standardize features
- **Used in**: Preprocessor module for feature scaling
- **Example**: `normalize: false`

#### `handle_missing` *(string, required)*
- **Description**: Strategy for handling missing values
- **Possible values**: 
  - `"interpolate"`: Linear interpolation
  - `"mean"`: Fill with column mean
  - `"median"`: Fill with column median
  - `"drop"`: Remove rows with missing values
  - `"forward_fill"`: Forward fill (use last valid value)
  - `"backward_fill"`: Backward fill (use next valid value)
- **Used in**: Preprocessor module
- **Example**: `handle_missing: interpolate`

#### `chunks` *(integer, optional)*
- **Description**: Number of dataset chunks to process
- **Used in**: DataLoader for processing large datasets in chunks
- **Default**: `1`
- **Example**: `chunks: 2`

---

## `model` Section

Controls model selection and configuration.

#### `name` *(list of strings, required)*
- **Description**: List of models to train and evaluate
- **Current supported models**: `["arima", "theta", "moirai", "moirai_moe", "croston_classic", "seasonal_naive", "toto", "exponential_smoothing", "lstm", "xgboost", "prophet", "random_forest", "svr", "tabpfn", "chronos", "moment", "lagllama", "timesfm", "deepar", "foundation_model", "base_model"]`
- **Used in**: Model factory for instantiating models
- **Example**: `name: ["arima", "theta", "moirai"]`

#### `parameters` *(object, required)*
- **Description**: Model-specific parameter configurations
- **Structure**: Each model has its own parameter section with specific hyperparameters
- **Used in**: Model instantiation and training

##### Model Parameter Examples:

**ARIMA Model:**
```yaml
arima:
  p: [0, 1]                    # AR order
  d: [0, 1]                    # Differencing order
  q: [0, 1]                    # MA order
  s: [2, 4]                    # Seasonal period
  target_col: ["y"]            # Target column name
  loss_functions: ["mae"]      # Loss functions to use
  primary_loss: ["mae"]        # Primary loss function
  forecast_horizon: [2]        # Forecast horizon
```

**LSTM Model:**
```yaml
lstm:
  units: [2]                   # Number of LSTM units
  layers: [1]                  # Number of LSTM layers
  dropout: [0.1]               # Dropout rate
  learning_rate: [0.01]        # Learning rate
  batch_size: [8]              # Batch size
  epochs: [1]                  # Number of training epochs
  sequence_length: [20]        # Input sequence length
  target_col: ["y"]            # Target column name
  loss_functions: ["mae"]      # Loss functions to use
  primary_loss: ["mae"]        # Primary loss function
  forecast_horizon: [10]       # Forecast horizon
```

**XGBoost Model:**
```yaml
xgboost:
  lookback_window: [10]        # Number of past values to use as features
  forecast_horizon: [10]       # Number of future values to predict
  n_estimators: [10]           # Number of boosting rounds
  max_depth: [5]               # Maximum tree depth
  learning_rate: [0.1]         # Learning rate
  random_state: [42]           # Random seed
  n_jobs: [-1]                 # Number of parallel jobs
```

**Foundation Models (MOMENT, Chronos, etc.):**
```yaml
moment:
  model_path: ['AutonLab/MOMENT-1-large']  # Pre-trained model path
  context_length: [512]                     # Input context length
  fine_tune_epochs: [0]                     # Fine-tuning epochs
  batch_size: [8]                           # Batch size
  learning_rate: [0.0001]                   # Learning rate
  prediction_length: [40]                   # Prediction length

chronos:
  model_size: ['small']                     # Model size variant
  context_length: [8]                       # Input context length
  num_samples: [5]                          # Number of prediction samples
  prediction_length: [40]                   # Prediction length
```

---

## `evaluation` Section

Controls model evaluation and metrics.

#### `type` *(string, required)*
- **Description**: Type of evaluation to perform
- **Possible values**: `"deterministic"`, `"probabilistic"`
- **Used in**: Evaluator module for selecting appropriate metrics
- **Example**: `type: deterministic`

#### `metrics` *(list of strings, required)*
- **Description**: Metrics to calculate for model evaluation
- **Possible values**: 
  - For deterministic: `"mae"`, `"rmse"`, `"mape"`, `"smape"`, `"mase"`
  - For probabilistic: `"crps"`, `"quantile_loss"`, `"interval_score"`
- **Used in**: Evaluator module for metric calculation
- **Example**: `metrics: [mae, rmse]`

---

## Configuration Examples

### Current Consolidated Configuration
The main configuration file (`all_model_test.yaml`) contains all models and their parameters:

```yaml
test_type: deterministic
dataset:
  name: australian_electricity_demand
  path: benchmarking_pipeline/datasets/australian_electricity_demand
  frequency: D
  forecast_horizon: 5
  split_ratio: [0.8, 0.1, 0.1]
  normalize: false
  handle_missing: interpolate
  chunks: 2

model:
  name: ["arima", "theta", "moirai", "moirai_moe", "croston_classic", 
          "seasonal_naive", "toto", "exponential_smoothing", "lstm", 
          "xgboost", "prophet", "random_forest", "svr", "tabpfn", 
          "chronos", "moment", "lagllama", "timesfm", "deepar", 
          "foundation_model", "base_model"]
  
  parameters:
    arima:
      p: [0, 1]
      d: [0, 1]
      q: [0, 1]
      s: [2, 4]
      target_col: ["y"]
      loss_functions: ["mae"]
      primary_loss: ["mae"]
      forecast_horizon: [2]
    
    lstm:
      units: [2]
      layers: [1]
      dropout: [0.1]
      learning_rate: [0.01]
      batch_size: [8]
      epochs: [1]
      sequence_length: [20]
      target_col: ["y"]
      loss_functions: ["mae"]
      primary_loss: ["mae"]
      forecast_horizon: [10]

evaluation:
  type: deterministic
  metrics: [mae, rmse]
```

### Minimal Configuration
For testing a subset of models:

```yaml
test_type: deterministic
dataset:
  name: australian_electricity_demand
  path: benchmarking_pipeline/datasets/australian_electricity_demand
  frequency: D
  forecast_horizon: 5
  split_ratio: [0.8, 0.1, 0.1]
  normalize: false
  handle_missing: interpolate
  chunks: 1

model:
  name: ["arima", "lstm"]
  
  parameters:
    arima:
      p: [1]
      d: [1]
      q: [1]
      s: [7]
      target_col: ["y"]
      loss_functions: ["mae"]
      primary_loss: ["mae"]
      forecast_horizon: [5]
    
    lstm:
      units: [10]
      layers: [2]
      dropout: [0.2]
      learning_rate: [0.001]
      batch_size: [16]
      epochs: [10]
      sequence_length: [20]
      target_col: ["y"]
      loss_functions: ["mae"]
      primary_loss: ["mae"]
      forecast_horizon: [5]

evaluation:
  type: deterministic
  metrics: [mae, rmse]
```

---

## Where Each Section is Used

| Configuration Section | Used In Module | Purpose |
|----------------------|----------------|---------|
| `test_type` | Pipeline orchestrator | Test categorization and routing |
| `dataset.*` | DataLoader, Preprocessor | Data loading and preprocessing |
| `model.name` | ModelFactory | Model selection and instantiation |
| `model.parameters.*` | Individual models | Model-specific hyperparameters |
| `evaluation.*` | Evaluator | Performance evaluation |

---

## Benefits of Consolidated Configuration

1. **Single Source of Truth**: All model configurations in one place
2. **Consistency**: Same dataset and evaluation settings for all models
3. **Maintainability**: Easier to update and manage parameters
4. **Reproducibility**: Consistent experimental setup across all models
5. **Scalability**: Easy to add new models or modify existing parameters

---

## Adding New Models

To add a new model to the configuration:

1. Add the model name to the `model.name` list
2. Add a new parameter section under `model.parameters`
3. Define the model-specific hyperparameters
4. Ensure the model implementation exists in the `models/` directory

Example:
```yaml
model:
  name: ["arima", "new_model"]
  
  parameters:
    arima:
      # ... existing parameters ...
    
    new_model:
      param1: [value1, value2]
      param2: [value3]
      target_col: ["y"]
      forecast_horizon: [10]
```

---

## Notes

- **Parameter Lists**: Most parameters use lists to enable hyperparameter tuning and grid search
- **Default Values**: Models have sensible defaults for parameters not specified in config
- **Model Dependencies**: Some models require specific packages (see individual model requirements.txt files)
- **Memory Considerations**: Large models (like MOMENT, Chronos) may require significant memory
- **GPU Support**: Some models (LSTM, DeepAR, foundation models) support GPU acceleration 