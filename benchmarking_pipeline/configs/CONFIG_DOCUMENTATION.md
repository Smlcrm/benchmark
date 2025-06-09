# Configuration File Documentation

This document explains the configuration file format used in the benchmarking pipeline. The configuration files control all aspects of data processing, model training, and evaluation.

## Configuration File Structure

### Root Level Parameters

#### `test_type` *(string, required)*
- **Description**: Specifies the type of time series test being conducted
- **Possible values**: `"univariate"`, `"multivariate"`, `"additive"`, `"multiplicative"`, `"deterministic"`, `"probabilistic"`, `"irregular"`, `"regular"`
- **Used in**: Pipeline orchestration, model selection, result logging
- **Example**: `test_type: univariate`

---

## `dataset` Section

Controls data loading and preprocessing parameters.

#### `name` *(string, required)*
- **Description**: Name/identifier of the dataset
- **Used in**: Data loading, result metadata, output file naming
- **Example**: `name: bdg-2_bear`

#### `path` *(string, required)*
- **Description**: Path to the dataset files (relative to project root)
- **Used in**: DataLoader module
- **Example**: `path: datasets/univariate/bdg-2_bear`

#### `frequency` *(string, required)*
- **Description**: Time series frequency following pandas frequency codes
- **Possible values**: `"H"` (hourly), `"D"` (daily), `"W"` (weekly), `"M"` (monthly), `"30T"` (30 minutes), etc.
- **Used in**: Data validation, feature engineering, model configuration
- **Example**: `frequency: H`

#### `forecast_horizon` *(integer, required)*
- **Description**: Number of time steps to forecast ahead
- **Used in**: Feature engineering, model training, evaluation
- **Example**: `forecast_horizon: 144`  # 144 hours = 6 days

#### `split_ratio` *(list of floats, required)*
- **Description**: Proportions for train/validation/test splits (must sum to 1.0)
- **Format**: `[train_ratio, validation_ratio, test_ratio]`
- **Used in**: DataLoader for splitting time series data
- **Example**: `split_ratio: [0.6, 0.2, 0.2]`

#### `normalize` *(boolean, required)*
- **Description**: Whether to normalize/standardize features
- **Used in**: Preprocessor module for feature scaling
- **Example**: `normalize: true`

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

---

## `features` Section

Controls feature engineering for tree-based and linear models.

#### `lags` *(list of integers, optional)*
- **Description**: Lag values to create as features (e.g., t-1, t-2, t-24)
- **Used in**: FeatureExtractor for creating lagged features
- **Example**: `lags: [1, 2, 3, 24, 168]`  # 1-3 hours, 1 day, 1 week back

#### `rolling_windows` *(list of integers, optional)*
- **Description**: Window sizes for rolling statistics (mean, std, min, max)
- **Used in**: FeatureExtractor for creating rolling window features
- **Example**: `rolling_windows: [7, 30]`  # 7-day and 30-day windows

#### `seasonal_features` *(boolean, optional)*
- **Description**: Whether to create seasonal features (hour, day, month, etc.)
- **Used in**: FeatureExtractor for datetime-based features
- **Default**: `false`
- **Example**: `seasonal_features: true`

#### `trend_features` *(boolean, optional)*
- **Description**: Whether to create trend features (linear/polynomial trends)
- **Used in**: FeatureExtractor for trend-based features
- **Default**: `false`
- **Example**: `trend_features: true`

---

## `training` Section

Controls model training and validation strategies.

### `cross_validation` Subsection

#### `method` *(string, optional)*
- **Description**: Cross-validation method for time series
- **Possible values**: `"time_series_split"`, `"blocked_time_series_split"`
- **Used in**: Trainer module for model validation
- **Example**: `method: "time_series_split"`

#### `n_splits` *(integer, optional)*
- **Description**: Number of cross-validation splits
- **Used in**: Cross-validation implementation
- **Default**: `3`
- **Example**: `n_splits: 3`

#### `test_size` *(float, optional)*
- **Description**: Proportion of data to use for each validation fold
- **Used in**: Cross-validation splitting
- **Default**: `0.2`
- **Example**: `test_size: 0.2`

### `early_stopping` Subsection

#### `patience` *(integer, optional)*
- **Description**: Number of epochs to wait before stopping if no improvement
- **Used in**: Neural network training (LSTM, GRU, Transformer models)
- **Default**: `10`
- **Example**: `patience: 10`

#### `monitor` *(string, optional)*
- **Description**: Metric to monitor for early stopping
- **Possible values**: `"val_loss"`, `"val_mae"`, `"val_rmse"`
- **Used in**: Neural network training callbacks
- **Default**: `"val_loss"`
- **Example**: `monitor: "val_loss"`

---

## `model` Section

Controls model selection and training.

#### `name` *(string or list, required)*
- **Description**: Model(s) to train and evaluate
- **Possible values**: 
  - `"all"`: Run all available models
  - Specific models: `"lstm"`, `"arima"`, `"prophet"`, `"xgboost"`, `"random_forest"`, etc.
  - List: `["lstm", "arima", "prophet"]`
- **Used in**: Model factory for instantiating models
- **Example**: `name: all`

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
- **Example**: `metrics: [mae, rmse, smape]`

---

## `logging` Section

Controls experiment logging and tracking.

#### `group` *(string, optional)*
- **Description**: Experiment group name for organizing results
- **Used in**: Logger module, experiment tracking systems
- **Example**: `group: "univariate-tests"`

#### `seed` *(integer, optional)*
- **Description**: Random seed for reproducibility
- **Used in**: Random number generators across the pipeline
- **Default**: `42`
- **Example**: `seed: 42`

---

## `output` Section

Controls result saving and output generation.

#### `save_predictions` *(boolean, optional)*
- **Description**: Whether to save model predictions to files
- **Used in**: Output handler for saving prediction results
- **Default**: `false`
- **Example**: `save_predictions: true`

#### `save_path` *(string, optional)*
- **Description**: Directory path where results should be saved
- **Used in**: Output handler for determining save location
- **Default**: `"outputs/"`
- **Example**: `save_path: outputs/elecdemand_naive_short/`

---

## Configuration Examples

### Minimal Configuration
```yaml
test_type: univariate
dataset:
  name: my_dataset
  path: datasets/my_dataset
  frequency: H
  forecast_horizon: 24
  split_ratio: [0.7, 0.15, 0.15]
  normalize: true
  handle_missing: interpolate
model:
  name: all
evaluation:
  type: deterministic
  metrics: [mae, rmse]
```

### Full Configuration
```yaml
test_type: multivariate
dataset:
  name: complex_dataset
  path: datasets/multivariate/complex_dataset
  frequency: 30T
  forecast_horizon: 48
  split_ratio: [0.6, 0.2, 0.2]
  normalize: true
  handle_missing: interpolate

features:
  lags: [1, 2, 3, 48, 96, 336]
  rolling_windows: [7, 14, 30]
  seasonal_features: true
  trend_features: true

training:
  cross_validation:
    method: "time_series_split"
    n_splits: 5
    test_size: 0.2
  early_stopping:
    patience: 15
    monitor: "val_mae"

model:
  name: ["lstm", "arima", "prophet", "xgboost"]

evaluation:
  type: probabilistic
  metrics: [mae, rmse, crps, quantile_loss]

logging:
  group: "production-tests"
  seed: 123

output:
  save_predictions: true
  save_path: outputs/production_run_2024/
```

---

## Where Each Section is Used

| Configuration Section | Used In Module | Purpose |
|----------------------|----------------|---------|
| `test_type` | Pipeline orchestrator | Test categorization and routing |
| `dataset.*` | DataLoader, Preprocessor | Data loading and preprocessing |
| `features.*` | FeatureExtractor | Feature engineering |
| `training.*` | Trainer | Model training and validation |
| `model.*` | ModelFactory | Model instantiation |
| `evaluation.*` | Evaluator | Performance evaluation |
| `logging.*` | Logger | Experiment tracking |
| `output.*` | OutputHandler | Result saving | 