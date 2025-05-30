import pandas as pd
import numpy as np

"""
Feature extraction utilities.
"""

class FeatureExtractor:
    def __init__(self, config):
        """
        Initialize feature extractor with configuration.
        
        Args:
            config: Configuration dictionary with feature extraction parameters
        """
        self.config = config
        
    # --- Helper Functions ---

    def create_lags(df, column_name, lags=[1, 2, 3]):
        """
        Creates lagged features for a specified column.
        """
        df_temp = df.copy()
        for lag in lags:
            df_temp[f'{column_name}_lag_{lag}'] = df_temp[column_name].shift(lag)
        return df_temp

    def create_datetime_features(df_input, datetime_col_or_index=None):
        """
        Creates date and time features from a datetime column or the DataFrame index.
        """
        df = df_input.copy()
        if datetime_col_or_index is None:
            dt_series = pd.to_datetime(df.index)
        elif isinstance(datetime_col_or_index, str):
            dt_series = pd.to_datetime(df[datetime_col_or_index])
        else: # Assumes it's already a DatetimeIndex or Series
            dt_series = pd.to_datetime(datetime_col_or_index)

        df['year'] = dt_series.year
        df['month'] = dt_series.month
        df['day'] = dt_series.day
        df['dayofweek'] = dt_series.dayofweek # Monday=0, Sunday=6
        df['dayofyear'] = dt_series.dayofyear
        df['weekofyear'] = dt_series.isocalendar().week.astype(int)
        df['quarter'] = dt_series.quarter
        df['hour'] = dt_series.hour # If applicable
        df['minute'] = dt_series.minute # If applicable
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        # Simple linear trend
        df['time_idx'] = np.arange(len(df))
        return df

    def create_rolling_features(df, column_name, window_sizes=[7, 14, 28], aggs=['mean', 'std', 'min', 'max']):
        """
        Creates rolling window features for a specified column.
        Shift by 1 to prevent data leakage (use past data only).
        """
        df_temp = df.copy()
        for window in window_sizes:
            shifted_series = df_temp[column_name].shift(1) # Use data up to t-1
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
        return df_temp

    # --- Feature Extraction Functions for Specific Models ---

    def features_for_naive(df, target_col='target'):
        """
        Output: DataFrame with the last observed value.
        For actual forecasting, you'd use df[target_col].iloc[-1].
        This function prepares a 'feature' for a conceptual supervised setup.
        """
        df_out = pd.DataFrame(index=df.index)
        df_out['last_observed_value'] = df[target_col].shift(1)
        df_out['target_for_training'] = df[target_col] # Actual value at time t
        return df_out.dropna()

    def features_for_seasonal_naive(df, target_col='target', seasonal_period=365):
        """
        Output: DataFrame with the value from the same period in the previous season.
        For actual forecasting, you'd use df[target_col].iloc[-seasonal_period].
        This function prepares a 'feature' for a conceptual supervised setup.
        """
        df_out = pd.DataFrame(index=df.index)
        df_out[f'last_seasonal_value_{seasonal_period}'] = df[target_col].shift(seasonal_period)
        df_out['target_for_training'] = df[target_col]
        return df_out.dropna()

    def features_for_drift(df, target_col='target'):
        """
        Output: For actual forecasting, the model fits a line between first and last point.
        Conceptually, the "features" are these points and the time indices.
        This function just returns the series for the model to process.
        """
        print("Drift method uses the first and last points of the historical series to extrapolate.")
        print("Provide the historical series directly to a drift model implementation.")
        return df[[target_col]].copy()

    def features_for_arima(df, target_col='target', exog_cols=None):
        """
        Output: DataFrame with target series and optional exogenous variables.
        """
        cols_to_return = [target_col]
        if exog_cols:
            cols_to_return.extend(exog_cols)
        return df[cols_to_return].copy()

    def features_for_tree_and_linear_models(
        df_input,
        target_col='target',
        lags_target=[1, 2, 3, 7, 14],
        rolling_windows_target=[7, 14],
        rolling_aggs_target=['mean', 'std'],
        datetime_features_to_extract=True,
        exog_cols=None,
        lags_exog=None, # dict like {'exog1': [1,2], 'exog2': [1]}
        rolling_windows_exog=None, # dict like {'exog1': [7], 'exog2': [7]}
        rolling_aggs_exog=None, # list of aggs, e.g. ['mean']
        forecast_horizon=1
    ):
        """
        Prepares features for models like XGBoost, Random Forest, Ridge Regression, SVR, Quantile Regression.
        Outputs a DataFrame suitable for supervised learning.
        """
        df = df_input.copy()

        # Target variable for training (shifted for forecasting horizon)
        df['y'] = df[target_col].shift(-forecast_horizon)

        # Target lags
        if lags_target:
            df = create_lags(df, target_col, lags=lags_target)

        # Target rolling features
        if rolling_windows_target and rolling_aggs_target:
            df = create_rolling_features(df, target_col, window_sizes=rolling_windows_target, aggs=rolling_aggs_target)

        # Datetime features
        if datetime_features_to_extract:
            df = create_datetime_features(df) # Assumes datetime index

        # Exogenous variables processing
        if exog_cols:
            for exog_col in exog_cols:
                # Exogenous variable lags
                if lags_exog and exog_col in lags_exog:
                    df = create_lags(df, exog_col, lags=lags_exog[exog_col])
                # Exogenous variable rolling features (shifted appropriately so they are available at time t)
                if rolling_windows_exog and rolling_aggs_exog and exog_col in rolling_windows_exog:
                    # For exog, we can use current and past values up to t-1 to predict t+h
                    # so shift is not strictly needed for exog variables IF they are known at time t
                    # However, to be safe and use only "past" information for exog vars to predict target at t+h,
                    # we can also shift them, or ensure they are contemporary.
                    # Here, let's assume exog_col at time t is available when predicting target at t+h
                    # Thus, rolling features for exog vars use data up to and including time t.
                    # If exog vars are only known with a lag, shift them before creating rolling features.
                    df_temp_exog = df.copy()
                    for window in rolling_windows_exog[exog_col]:
                        for agg in rolling_aggs_exog:
                            col_name = f'{exog_col}_rolling_{agg}_{window}'
                            if agg == 'mean':
                                df_temp_exog[col_name] = df_temp_exog[exog_col].rolling(window=window, min_periods=1).mean()
                            elif agg == 'std':
                                df_temp_exog[col_name] = df_temp_exog[exog_col].rolling(window=window, min_periods=1).std()
                            # Add other aggs if needed
                    df = df_temp_exog

        df = df.dropna() # Drop rows with NaNs created by lags/rolling windows/target shift
        
        # Select feature columns (X) and target (y)
        y = df['y']
        X = df.drop(columns=['y', target_col] + (exog_cols if exog_cols else [])) # Drop original target and exog, keep engineered features
        
        # Ensure all original exog columns that were not transformed/lagged are also dropped from X if not explicitly kept
        if exog_cols:
            original_exog_to_drop = [col for col in exog_cols if col in X.columns]
            X = X.drop(columns=original_exog_to_drop, errors='ignore')

        return X, y


    def features_for_sequential_models(
        df_input,
        target_col='target',
        sequence_length=14,
        forecast_horizon=1,
        exog_cols=None,
        datetime_features_to_extract=['month', 'dayofweek']
    ):
        """
        Prepares sequences for models like LSTM, TCN, Transformer.
        Outputs X (sequences of features) and y (target values).
        """
        df = df_input.copy()

        # Add datetime features if requested (these will become part of the sequence)
        if datetime_features_to_extract:
            df = create_datetime_features(df) # Assumes datetime index
            # One-hot encode categorical datetime features for neural networks
            for col in ['year', 'month', 'dayofweek', 'quarter', 'hour', 'minute', 'second']:
                if col in df.columns and col in datetime_features_to_extract:
                    df = pd.get_dummies(df, columns=[col], prefix=col)
            
            # Select only the specified datetime features to keep for sequencing
            selected_dt_cols = []
            for dt_feat_base in datetime_features_to_extract:
                selected_dt_cols.extend([col for col in df.columns if col.startswith(dt_feat_base)])
        else:
            selected_dt_cols = []

        feature_cols = [target_col] + (exog_cols if exog_cols else []) + selected_dt_cols
        df_features = df[feature_cols].copy()

        # Create sequences
        X_list, y_list = [], []
        for i in range(len(df_features) - sequence_length - forecast_horizon + 1):
            X_list.append(df_features.iloc[i : i + sequence_length].values)
            y_list.append(df_features[target_col].iloc[i + sequence_length + forecast_horizon - 1])

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"Generated X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def features_for_deepar(df, target_col='target', dynamic_exog_cols=None, static_cat_cols=None):
        """
        DeepAR implementations (e.g., in GluonTS) have specific data format requirements.
        This function describes the conceptual features.
        """
        print("DeepAR conceptually needs:")
        print(f"- Historical target series: df['{target_col}']")
        if dynamic_exog_cols:
            print(f"- Historical and future dynamic exogenous variables: df[{dynamic_exog_cols}]")
        if static_cat_cols:
            print(f"- Static categorical features (e.g., item IDs): df[{static_cat_cols}] (often one per series)")
        print("Data is usually fed via specialized iterators/datasets (e.g., GluonTS ListDataset).")
        print("Scaling/normalization is standard.")
        # Return the relevant parts of the dataframe for further processing by a DeepAR library
        cols_to_return = [target_col]
        if dynamic_exog_cols:
            cols_to_return.extend(dynamic_exog_cols)
        if static_cat_cols:
            cols_to_return.extend(static_cat_cols)
        return df[cols_to_return].copy()

    def features_for_prophet(df_input, target_col='target', holiday_df=None, exog_cols_for_prophet=None):
        """
        Prophet expects a DataFrame with 'ds' (datetimes) and 'y' (target) columns.
        """
        df = df_input.copy()
        df_prophet = pd.DataFrame()
        df_prophet['ds'] = df.index
        df_prophet['y'] = df[target_col].values

        if exog_cols_for_prophet:
            for col in exog_cols_for_prophet:
                if col in df.columns:
                    df_prophet[col] = df[col].values
                else:
                    print(f"Exogenous column '{col}' not found in input DataFrame.")
        
        print("Prophet requires a DataFrame with 'ds' and 'y' columns.")
        if holiday_df is not None:
            print("Provide holiday_df separately to Prophet model during fitting.")
        if exog_cols_for_prophet:
            print(f"Additional regressors {exog_cols_for_prophet} included.")
    
        return df_prophet