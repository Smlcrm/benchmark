import pandas as pd
import numpy as np

# Scikit-learn for ML models and utilities
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, QuantileRegressor as SklearnQuantileRegressor
from sklearn.svm import SVR
# Note: XGBoost must be installed separately: pip install xgboost
from xgboost import XGBRegressor

# Statsmodels for classical time series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prophet for its specific forecasting procedure
# Note: Prophet must be installed separately: pip install prophet
from prophet import Prophet

# Sktime for specific time series models like Theta
# Note: Sktime must be installed separately: pip install sktime
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.base import BaseForecaster

# Baseline Models
# Maybe move these to the models folder later

class NaiveForecaster(BaseForecaster):
    """
    A simple forecaster that predicts the last observed value.
    """
    def _fit(self, y, X=None, fh=None):
        self.last_value = y.iloc[-1]
        return self
    
    def _predict(self, fh=None, X=None):
        return pd.Series(self.last_value, index=self.fh.to_absolute(self.cutoff))

class SeasonalNaiveForecaster(BaseForecaster):
    """
    A forecaster that predicts the value from the last seasonal period.
    """
    def __init__(self, sp=1):
        super().__init__()
        self.sp = sp

    def _fit(self, y, X=None, fh=None):
        self.y_train = y
        return self

    def _predict(self, fh=None, X=None):
        # Create an index for the forecast horizon
        fh_index = self.fh.to_absolute(self.cutoff)
        # For each future point, find the corresponding past seasonal point
        predictions = []
        for i in range(len(fh)):
            # Find the index that is `sp` periods before the current forecast point
            # This is a simplified approach; more robust logic might be needed for irregular dates
            pred_idx = len(self.y_train) - self.sp + i
            predictions.append(self.y_train.iloc[pred_idx])
        return pd.Series(predictions, index=fh_index)


class DriftForecaster(BaseForecaster):
    """
    A forecaster that predicts based on a line between the first and last points.
    """
    def _fit(self, y, X=None, fh=None):
        self.y_1 = y.iloc[0]
        self.y_t = y.iloc[-1]
        self.t = len(y)
        self.drift = (self.y_t - self.y_1) / (self.t - 1)
        return self

    def _predict(self, fh=None, X=None):
        fh_index = self.fh.to_absolute(self.cutoff)
        predictions = [self.y_t + h * self.drift for h in fh.to_relative(self.cutoff)]
        return pd.Series(predictions, index=fh_index)

# Main Trainer

class Trainer:
    def __init__(self, config=None):
        """
        Initialize trainer with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with training parameters.
                                     e.g., {'target_col': 'y', 'exog_cols': ['feature1']}
        """
        self.config = config if config is not None else {}
        self.target_col = self.config.get('target_col', 'y')
        self.exog_cols = self.config.get('exog_cols', None)

    def train(self, model, data):
        """
        Train the model with given data. Dispatches to the correct training method
        based on the model's type.
        
        Args:
            model: An unfitted model instance.
            data: Training data. Expected to be a tuple (X, y) for sklearn models,
                  or a DataFrame for statsmodels, Prophet, and sktime models.
                  
        Returns:
            Trained/fitted model.
        """
        model_type = model.__class__.__name__

        # Scikit-learn compatible models (XGBoost, RandomForest, Ridge, SVR)
        if isinstance(model, (XGBRegressor, RandomForestRegressor, Ridge, SVR, SklearnQuantileRegressor)):
            if not (isinstance(data, tuple) and len(data) == 2):
                raise ValueError(f"Data for {model_type} must be a tuple (X_train, y_train).")
            X_train, y_train = data
            model.fit(X_train, y_train)
            return model

        # Statsmodels models (ARIMA, SARIMAX, ExponentialSmoothing/ETS)
        elif isinstance(model, (ARIMA, SARIMAX, ExponentialSmoothing)):
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Data for {model_type} must be a Pandas DataFrame.")
            endog = data[self.target_col]
            exog = data[self.exog_cols] if self.exog_cols and self.exog_cols in data.columns else None
            
            # Re-initialize the model with data, as this is the statsmodels pattern
            # The passed 'model' object serves to define the type and params
            model_class = model.__class__
            model_params = model.get_params() # Assumes a get_params method or access to params
            
            # Statsmodels models are initialized with the data
            # We assume the model we are passed is just a template for the parameters we want, ignoring the data the model was initialized with
            if isinstance(model, (ARIMA, SARIMAX)):
                 fitted_model = model_class(endog=endog, exog=exog, **model_params).fit()
            elif isinstance(model, ExponentialSmoothing):
                 fitted_model = model_class(endog=endog, trend=model.trend, seasonal=model.seasonal, seasonal_periods=model.seasonal_periods, damped_trend=model.damped_trend).fit()
            return fitted_model

        # Prophet Model
        elif isinstance(model, Prophet):
            if not isinstance(data, pd.DataFrame) or 'ds' not in data or 'y' not in data:
                 raise ValueError("Data for Prophet must be a DataFrame with 'ds' and 'y' columns.")
            # Add regressors if specified in config and available in data
            if self.exog_cols:
                for regressor in self.exog_cols:
                    if regressor in data.columns:
                        model.add_regressor(regressor)
            model.fit(data)
            return model
            
        # Sktime and custom baseline models
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