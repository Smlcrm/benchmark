import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.base import BaseForecaster
from statsmodels.tsa.stattools import adfuller
import warnings
import itertools

# Define the Trainer class
class Trainer: #need to update this - move code to individual models.
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        self.target_col = self.config.get('target_col', 'y')
        self.exog_cols = self.config.get('exog_cols', None)

    def train(self, model, data):
        model_type = self.config.get('model', {}).get('name', 'ARIMA')

        if isinstance(model, (XGBRegressor, RandomForestRegressor, Ridge, SVR)):
            if not (isinstance(data, tuple) and len(data) == 2):
                raise ValueError(f"Data for {model_type} must be a tuple (X_train, y_train).")
            X_train, y_train = data
            model.fit(X_train, y_train)
            return model

        elif isinstance(model, (ARIMA, SARIMAX)):
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Data for {model_type} must be a Pandas DataFrame.")
            
            endog = data[self.target_col]
            exog = data[self.exog_cols] if self.exog_cols and self.exog_cols in data.columns else None
            
            # Use grid search for ARIMA
            best_model, best_hyperparameters = self._arima_grid_search(endog)
            return best_model

        elif isinstance(model, ExponentialSmoothing):
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Data for {model_type} must be a Pandas DataFrame.")
            endog = data[self.target_col]
            
            # Get ETS parameters from config
            ets_params = self.config.get('model', {}).get('parameters', {})
            trend = ets_params.get('trend', 'add')
            seasonal = ets_params.get('seasonal', None)
            seasonal_periods = ets_params.get('seasonal_periods', None)
            
            fitted_model = ExponentialSmoothing(
                endog=endog,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            ).fit()
            return fitted_model

        elif isinstance(model, BaseForecaster):
            if isinstance(data, pd.DataFrame):
                y_train = data[self.target_col]
                X_train = data[self.exog_cols] if self.exog_cols and self.exog_cols in data.columns else None
            elif isinstance(data, pd.Series):
                y_train = data
                X_train = None
            else:
                raise ValueError("Data for Sktime models must be a Pandas Series or DataFrame.")

            model.fit(y=y_train, X=X_train)
            return model

        else:
            raise TypeError(f"Model type '{model_type}' is not supported by this trainer.")