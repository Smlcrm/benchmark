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

# Define Baseline Forecasters
class NaiveForecaster(BaseForecaster):
    def _fit(self, y, X=None, fh=None):
        self.last_value = y.iloc[-1]
        return self

    def _predict(self, fh=None, X=None):
        return pd.Series(self.last_value, index=self.fh.to_absolute(self.cutoff))

class SeasonalNaiveForecaster(BaseForecaster):
    def __init__(self, sp=1):
        super().__init__()
        self.sp = sp

    def _fit(self, y, X=None, fh=None):
        self.y_train = y
        return self

    def _predict(self, fh=None, X=None):
        fh_index = self.fh.to_absolute(self.cutoff)
        predictions = []
        for i in range(len(fh)):
            pred_idx = len(self.y_train) - self.sp + i
            predictions.append(self.y_train.iloc[pred_idx])
        return pd.Series(predictions, index=fh_index)

class DriftForecaster(BaseForecaster):
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

# Define the Trainer class
class Trainer:
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

        elif isinstance(model, (ARIMA, SARIMAX, ExponentialSmoothing)):
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Data for {model_type} must be a Pandas DataFrame.")
            endog = data[self.target_col]
            exog = data[self.exog_cols] if self.exog_cols and self.exog_cols in data.columns else None
            model_class = model.__class__

            if isinstance(model, (ARIMA, SARIMAX)):
                fitted_model = model_class(endog=endog, exog=exog).fit()
            elif isinstance(model, ExponentialSmoothing):
                fitted_model = model_class(endog=endog, trend="add", seasonal=None).fit()
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

# Dry run for ARIMA
np.random.seed(42)
n = 50
df = pd.DataFrame({'y': np.random.randn(n).cumsum()})

config = {'target_col': 'y', 'model': {'name': 'ARIMA'}}
model = ARIMA(endog=df['y'])
trainer = Trainer(config)
fitted_model = trainer.train(model, df)

# Display the ARIMA model summary as a string
summary_text = fitted_model.summary().as_text()
print(summary_text)