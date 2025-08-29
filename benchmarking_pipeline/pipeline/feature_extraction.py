import pandas as pd
import numpy as np

class FeatureExtractor:
    def __init__(self, config=None):
        """
        Initialize feature extractor with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with feature extraction parameters.
                                     Can include keys like 'lags', 'rolling_windows',
                                     'datetime_features', 'exog_cols', 'model_type', etc.
                                     Defaults to None for basic operation.
        """
        self.config = config if config is not None else {}
        self.datetime_col = self.config.get('datetime_col', None) # If None, assumes index is datetime

    def _create_lags(self, df, column_name, lags):
        """
        Creates lagged features for a specified column.
        """
        df_temp = df.copy()
        for lag in lags:
            df_temp[f'{column_name}_lag_{lag}'] = df_temp[column_name].shift(lag)
        return df_temp

    def _create_datetime_features(self, df_input):
        """
        Creates date and time features from a datetime column or the DataFrame index.
        """
        df = df_input.copy()
        if self.datetime_col and self.datetime_col in df.columns:
            dt_series = pd.to_datetime(df[self.datetime_col])
        else:
            dt_series = pd.to_datetime(df.index)

        config_dt_features = self.config.get('datetime_features', [])
        
        if not config_dt_features or 'all' in config_dt_features: # Default to a common set or all
            df['year'] = dt_series.year
            df['month'] = dt_series.month
            df['day'] = dt_series.day
            df['dayofweek'] = dt_series.dayofweek
            df['dayofyear'] = dt_series.dayofyear
            df['weekofyear'] = dt_series.isocalendar().week.astype(int)
            df['quarter'] = dt_series.quarter
            if dt_series.dt.hour.any(): # Check if hour info is non-zero or present
                 df['hour'] = dt_series.hour
            if dt_series.dt.minute.any():
                 df['minute'] = dt_series.minute
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
            df['time_idx'] = np.arange(len(df)) # Simple linear trend
        else:
            if 'year' in config_dt_features: df['year'] = dt_series.year
            if 'month' in config_dt_features: df['month'] = dt_series.month
            if 'day' in config_dt_features: df['day'] = dt_series.day
            if 'dayofweek' in config_dt_features: df['dayofweek'] = dt_series.dayofweek
            if 'dayofyear' in config_dt_features: df['dayofyear'] = dt_series.dayofyear
            if 'weekofyear' in config_dt_features: df['weekofyear'] = dt_series.isocalendar().week.astype(int)
            if 'quarter' in config_dt_features: df['quarter'] = dt_series.quarter
            if 'hour' in config_dt_features and dt_series.dt.hour.any(): df['hour'] = dt_series.hour
            if 'minute' in config_dt_features and dt_series.dt.minute.any(): df['minute'] = dt_series.minute
            if 'is_weekend' in config_dt_features: df['is_weekend'] = (df['dayofweek'] >= 5).astype(int) # Requires dayofweek
            if 'time_idx' in config_dt_features: df['time_idx'] = np.arange(len(df))
        return df

    def _create_rolling_features(self, df, column_name, window_configs):
        """
        Creates rolling window features for a specified column.
        window_configs is a list of dicts, e.g., [{'window': 7, 'aggs': ['mean', 'std']}]
        Shift by 1 to prevent data leakage.
        """
        df_temp = df.copy()
        shifted_series = df_temp[column_name].shift(1) # Use data up to t-1
        
        for config in window_configs:
            window = config.get('window')
            aggs = config.get('aggs', ['mean', 'std'])
            if not window:
                continue
            for agg in aggs:
                col_name = f'{column_name}_rolling_{agg}_{window}'
                if agg == 'mean':
                    df_temp[col_name] = shifted_series.rolling(window=window, min_periods=1).mean()
                elif agg == 'std':
                    df_temp[col_name] = shifted_series.rolling(window=window, min_periods=1).std()
                elif agg == 'min':
                    df_temp[col_name] = shifted_series.rolling(window=window, min_periods=1).min()
                elif agg == 'max':
                    df_temp[col_name] = shifted_series.rolling(window=window, min_periods=1).max()
                # Add more aggregations if needed
        return df_temp

    def extract_features_for_tabular(self, data):
        """
        Extracts features for tree-based, linear models, SVR, Quantile Regression.
        Assumes 'data' is a Pandas DataFrame.
        All data is treated as multivariate where univariate is just num_targets == 1.
        """
        df = data.copy()
        
        # Get forecast_horizon from dataset configuration
        dataset_cfg = self.config.get('dataset', {})
        forecast_horizon = dataset_cfg.get('forecast_horizon', 1)
        # Infer target column as the first column (multivariate approach)
        target_column_name = df.columns[0]
        df['y_target'] = df[target_column_name].shift(-forecast_horizon)
        
        # Get lag and rolling window configurations
        target_lags = self.config.get('lags', [1, 2, 3, 7])
        rolling_configs = self.config.get('rolling_windows', [{'window': 7, 'aggs': ['mean', 'std']}])
        
        # Create lag features
        df = self._create_lags(df, target_column_name, lags=target_lags)
        
        # Create rolling window features
        df = self._create_rolling_features(df, target_column_name, rolling_configs)

        # Datetime features
        df = self._create_datetime_features(df)

        # Drop rows with NaN values (from lag creation)
        df = df.dropna()

        # Extract target variable
        y = df['y_target'].values
        X = df.drop(columns=['y_target']).values

        return X, y

    def extract_features_for_sequence(self, data):
        """
        Extracts features for sequence models like LSTM, GRU, Transformer.
        Assumes 'data' is a Pandas DataFrame.
        All data is treated as multivariate where univariate is just num_targets == 1.
        """
        df = data.copy()
        
        # Get forecast_horizon from dataset configuration
        dataset_cfg = self.config.get('dataset', {})
        forecast_horizon = dataset_cfg.get('forecast_horizon', 1)
        # Infer target column as the first column (multivariate approach)
        target_column_name = df.columns[0]
        df['y_target'] = df[target_column_name].shift(-forecast_horizon)

        # Target lags
        target_lags = self.config.get('lags_target', [1, 7, 14])
        if target_lags:
            df = self._create_lags(df, target_column_name, lags=target_lags)

        # Target rolling features
        target_rolling_configs = self.config.get('rolling_features_target', [{'window': 7, 'aggs': ['mean', 'std']}])
        if target_rolling_configs:
            df = self._create_rolling_features(df, target_column_name, window_configs=target_rolling_configs)
        
        # Datetime features
        if self.config.get('extract_datetime_features', True):
            df = self._create_datetime_features(df)

        # Exogenous variables processing
        exog_cols = self.config.get('exog_cols', [])
        if exog_cols:
            exog_lags_config = self.config.get('lags_exog', {}) # e.g. {'exog1': [1,2]}
            exog_rolling_configs = self.config.get('rolling_features_exog', {}) # e.g. {'exog1': [{'window':7, 'aggs':['mean']}]}
            for exog_col in exog_cols:
                if exog_col not in df.columns:
                    raise ValueError(f"Exogenous column '{exog_col}' not found in data. Please check your configuration.")
                if exog_lags_config and exog_col in exog_lags_config:
                    df = self._create_lags(df, exog_col, lags=exog_lags_config[exog_col])
                if exog_rolling_configs and exog_col in exog_rolling_configs:
                    # For exog vars, rolling features can use current data if known at prediction time
                    # Here, for simplicity, we'll use past data like target rolling features
                    df = self._create_rolling_features(df, exog_col, window_configs=exog_rolling_configs[exog_col])
        
        df = df.dropna(subset=['y_target']) # Ensure target is not NaN
        df = df.fillna(method='bfill').fillna(method='ffill') # Simple imputation for features, consider more sophisticated methods

        y = df['y_target']
        
        # Columns to drop from X: original target, original exogs (if transformed), and y_target
        cols_to_drop_from_X = ['y_target', target_column_name] + exog_cols
        X = df.drop(columns=cols_to_drop_from_X, errors='ignore')
        
        return X, y

    def extract_features_for_prophet(self, data):
        """
        Formats data for Prophet.
        All data is treated as multivariate where univariate is just num_targets == 1.
        """
        df = data.copy()
        df_prophet = pd.DataFrame()
        
        if self.datetime_col and self.datetime_col in df.columns:
            df_prophet['ds'] = pd.to_datetime(df[self.datetime_col])
        else:
            df_prophet['ds'] = pd.to_datetime(df.index)
            
        # Get forecast_horizon from dataset configuration
        dataset_cfg = self.config.get('dataset', {})
        forecast_horizon = dataset_cfg.get('forecast_horizon', 1)
        # Infer target column as the first column (multivariate approach)
        target_column_name = df.columns[0]
        df_prophet['y'] = df[target_column_name].shift(-forecast_horizon).values

        exog_cols_prophet = self.config.get('exog_cols_for_prophet', [])
        if exog_cols_prophet:
            for col in exog_cols_prophet:
                if col in df.columns:
                    df_prophet[col] = df[col].values
                else:
                    raise ValueError(f"Prophet regressor '{col}' not found in input DataFrame. Please check your configuration.")
        
        # holiday_df should be prepared separately and passed to Prophet model
        # self.config.get('holiday_df', None)
        return df_prophet

    def extract_features(self, data):
        """
        Extract features from preprocessed data based on model_type in config.
        All data is treated as multivariate where univariate is just num_targets == 1.

        Args:
            data: Preprocessed input data (Pandas DataFrame)

        Returns:
            Extracted features (format depends on model_type)
        """
        model_type = self.config.get('model_type', 'tabular').lower()

        if model_type in ['naive', 'seasonal_naive', 'drift', 'arima', 'ets', 'theta_method', 'nbeats']:
            # These models often work with the series directly or have library-specific handling
            print(f"For model type '{model_type}', feature extraction often means providing the (scaled) target series.")
            print("Specific preprocessing like differencing for ARIMA or windowing for NBEATS is typically handled by the model/library.")
            # For simplicity, we'll return the relevant columns.
            # Actual feature engineering for these is minimal from this class's perspective.
            target_column_name = data.columns[0]  # Infer target column (multivariate approach)
            cols = [target_column_name] + self.config.get('exog_cols', [])
            return data[cols].copy() # Return a copy to avoid modifying original
        
        elif model_type in ['xgboost', 'random_forest', 'ridge_regression', 'svr', 'quantile_regression']:
            return self.extract_features_for_tabular(data)
            
        elif model_type in ['lstm', 'tcn', 'transformer']:
            return self.extract_features_for_sequence(data)
            
        elif model_type == 'prophet':
            return self.extract_features_for_prophet(data)
            
        elif model_type == 'deepar':
            print("DeepAR feature extraction is highly specific to the library (e.g., GluonTS).")
            print("This class will return relevant columns for further processing by such a library.")
            target_column_name = data.columns[0]  # Infer target column (multivariate approach)
            cols = [target_column_name] + \
                   self.config.get('dynamic_exog_cols', []) + \
                   self.config.get('static_cat_cols', [])
            # Ensure unique columns
            cols = list(dict.fromkeys(c for c in cols if c in data.columns))
            return data[cols].copy()
            
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")