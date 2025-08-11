"""
Exponential Smoothing model implementation.
"""

import os
import pickle
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from benchmarking_pipeline.models.base_model import BaseModel


class ExponentialSmoothingModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Exponential Smoothing model with a given configuration.
        
        Args:
            config: Configuration dictionary for model parameters.
                    e.g., {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12, ...}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)

        def _cast_param(key, value):
            if key == 'seasonal_periods':
                return int(value) if value is not None else None
            if key == 'damped_trend':
                if isinstance(value, str):
                    return value.lower() == 'true'
                return bool(value)
            if key == 'forecast_horizon':
                return int(value) if value is not None else 1
            if key in ['trend', 'seasonal']:
                if isinstance(value, str) and value.lower() == 'none':
                    return None
                return value
            return value

        self.trend = _cast_param('trend', self.config.get('trend', None))
        self.seasonal = _cast_param('seasonal', self.config.get('seasonal', None))
        self.seasonal_periods = _cast_param('seasonal_periods', self.config.get('seasonal_periods', None))
        self.damped_trend = _cast_param('damped_trend', self.config.get('damped_trend', False))

        # Single-model holder for univariate
        self.model_ = None
        # Per-series models for multivariate
        self.models_: Dict[str, Any] = {}

        self.is_fitted = False
        self.is_multivariate_ = False
        self.series_columns_ = None

        self.loss_functions = self.config.get('loss_functions', ['mae'])
        self.primary_loss = self.config.get('primary_loss', self.loss_functions[0])
        self.forecast_horizon = _cast_param('forecast_horizon', self.config.get('forecast_horizon', 1))

    def _normalize_endog(self, y_context: Union[pd.Series, pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input into a DataFrame and record column names."""
        if isinstance(y_context, pd.Series):
            df = y_context.to_frame(name=y_context.name or 'y0')
        elif isinstance(y_context, pd.DataFrame):
            df = y_context.copy()
        elif isinstance(y_context, np.ndarray):
            if y_context.ndim == 1:
                df = pd.DataFrame(y_context, columns=['y0'])
            elif y_context.ndim == 2:
                n_cols = y_context.shape[1]
                df = pd.DataFrame(y_context, columns=[f'y{i}' for i in range(n_cols)])
            else:
                raise ValueError("y_context ndarray must be 1D or 2D.")
        else:
            raise TypeError("y_context must be a pandas Series/DataFrame or numpy ndarray.")
        return df

    def _cast_holt_params(self):
        """Cast/clean Holt-Winters parameters for statsmodels."""
        trend = self.trend
        seasonal = self.seasonal
        if isinstance(trend, str) and trend.lower() == 'none':
            trend = None
        if isinstance(seasonal, str) and seasonal.lower() == 'none':
            seasonal = None

        seasonal_periods = int(self.seasonal_periods) if self.seasonal_periods is not None else None

        damped_trend = self.damped_trend
        if isinstance(damped_trend, str):
            damped_trend = damped_trend.lower() == 'true'
        damped_trend = bool(damped_trend)
        if trend is None:
            # statsmodels only allows damped_trend when a trend component is used
            damped_trend = None

        return trend, seasonal, seasonal_periods, damped_trend

    def train(self, y_context: Union[pd.Series, np.ndarray, pd.DataFrame],
              y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None,
              x_context: Union[pd.Series, np.ndarray, pd.DataFrame] = None,
              x_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None,
              **kwargs) -> 'ExponentialSmoothingModel':
        print(f"[ExponentialSmoothing train] y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}")
        df = self._normalize_endog(y_context)
        self.series_columns_ = list(df.columns)
        self.is_multivariate_ = len(self.series_columns_) > 1

        trend, seasonal, seasonal_periods, damped_trend = self._cast_holt_params()

        if not self.is_multivariate_:
            # Univariate: fit a single statsmodels model as before
            endog = df.iloc[:, 0].values
            self.model_ = ExponentialSmoothing(
                endog,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend
            ).fit()
            self.models_.clear()
        else:
            # Multivariate: fit one independent univariate model per column
            self.model_ = None
            self.models_.clear()
            for col in self.series_columns_:
                endog = df[col].values
                fitted = ExponentialSmoothing(
                    endog,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods,
                    damped_trend=damped_trend
                ).fit()
                self.models_[col] = fitted

        self.is_fitted = True
        return self

    def _infer_horizon(self, y_target) -> int:
        """Infer forecast horizon from y_target."""
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")
        if isinstance(y_target, (pd.Series, pd.DataFrame, np.ndarray)):
            return len(y_target)
        # Fallback: try to treat as sized object
        try:
            return len(y_target)  # type: ignore
        except Exception as e:
            raise ValueError("Unable to infer forecast horizon from y_target.") from e

    def predict(self, y_context: Union[pd.Series, np.ndarray, pd.DataFrame] = None,
                y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None,
                x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None,
                x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None,
                **kwargs) -> np.ndarray:
        print(f"[ExponentialSmoothing predict] y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}")
        print(f"[ExponentialSmoothing predict] y_target type: {type(y_target)}, shape: {getattr(y_target, 'shape', 'N/A')}")
        if not self.is_fitted:
            raise ValueError("Model not initialized. Call train first.")

        forecast_steps = self._infer_horizon(y_target)

        if not self.is_multivariate_:
            # Preserve original univariate shape: (1, horizon)
            forecast = self.model_.forecast(steps=forecast_steps)
            return np.asarray(forecast).reshape(1, -1)

        # Multivariate: forecast each column and stack => (1, horizon, n_series)
        if not self.models_ or not self.series_columns_:
            raise ValueError("Multivariate models are not available. Train the model first.")

        per_col_forecasts = []
        for col in self.series_columns_:
            fc = self.models_[col].forecast(steps=forecast_steps)
            per_col_forecasts.append(np.asarray(fc).reshape(-1, 1))  # (h, 1)

        forecast_matrix = np.hstack(per_col_forecasts)  # (h, n_series)
        return forecast_matrix.reshape(1, forecast_steps, len(self.series_columns_))

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the configuration and instance attributes.
        """
        return {
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'damped_trend': self.damped_trend,
            'loss_functions': self.loss_functions,
            'primary_loss': self.primary_loss,
            'forecast_horizon': self.forecast_horizon,
            'is_fitted': self.is_fitted,
            'is_multivariate': self.is_multivariate_
        }

    def set_params(self, **params: Dict[str, Any]) -> 'ExponentialSmoothingModel':
        """
        Set model parameters by updating the configuration and instance attributes.
        The model will be rebuilt with these new parameters on the next .train() call.
        """
        def _cast_param(key, value):
            if key == 'seasonal_periods':
                return int(value) if value is not None else None
            if key == 'damped_trend':
                if isinstance(value, str):
                    return value.lower() == 'true'
                return bool(value)
            if key == 'forecast_horizon':
                return int(value) if value is not None else 1
            if key in ['trend', 'seasonal']:
                if isinstance(value, str) and value.lower() == 'none':
                    return None
                return value
            return value

        for key, value in params.items():
            casted_value = _cast_param(key, value)
            setattr(self, key, casted_value)
            self.config[key] = casted_value

        # Reset fitted state
        self.is_fitted = False
        self.is_multivariate_ = False
        self.model_ = None
        self.models_.clear()
        self.series_columns_ = None
        return self

    def save(self, path: str) -> None:
        """
        Save the trained statsmodels ETS model(s) to disk using pickle.
        
        Args:
            path: Path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")

        payload = {
            'is_multivariate': self.is_multivariate_,
            'series_columns': self.series_columns_,
            'model': self.model_,
            'models': self.models_
        }

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> 'ExponentialSmoothingModel':
        """
        Load trained statsmodels ETS model(s) from disk.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")

        with open(path, 'rb') as f:
            payload = pickle.load(f)

        self.is_multivariate_ = payload.get('is_multivariate', False)
        self.series_columns_ = payload.get('series_columns', None)
        self.model_ = payload.get('model', None)
        self.models_ = payload.get('models', {}) or {}
        self.is_fitted = True
        return self