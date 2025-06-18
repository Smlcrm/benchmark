# Forecasting Model Implementations & Explanations

Below is a summary of the various time series forecasting models, how they work, and the key Python libraries used for their implementation.

## Model Overview

The table below outlines each model, its primary use case, and the main library required for its implementation.

| Model | How It Works | Primary Library |
| :--- | :--- | :--- |
| **ARIMA** | (Autoregressive Integrated Moving Average) A classical statistical model that combines three concepts: **(AR)** it uses past values of the series to predict the future; **(I)** it uses differencing to make the series stationary (remove trend/seasonality); and **(MA)** it uses past forecast errors to improve the current forecast. | `statsmodels` |
| **Exponential Smoothing (ETS)** | A family of models that generate forecasts based on weighted averages of past observations, with the weights decaying exponentially as the observations get older. It explicitly models and combines error, trend, and seasonality components in an additive or multiplicative manner. | `statsmodels` |
| **LSTM** | (Long Short-Term Memory) A type of Recurrent Neural Network (RNN) that uses internal "gates" (forget, input, and output gates) to regulate the flow of information. This allows it to remember patterns over long sequences while forgetting irrelevant data, making it effective for capturing long-term dependencies. | `TensorFlow`/`Keras` |
| **Prophet** | A forecasting procedure developed by Meta that is also based on decomposition. It fits an additive model with separate components for: a piecewise linear or logistic trend, multiple seasonalities (yearly, weekly, daily) using Fourier series, and a user-specified list of holidays. | `prophet` |
| **Random Forest** | An ensemble machine learning model that fits multiple decision trees on various sub-samples of the dataset and uses averaging to improve predictive accuracy and control over-fitting. It can capture complex, non-linear relationships between the engineered features (lags, date parts, etc.) and the target. | `scikit-learn` |
| **Seasonal Naive** | A simple but powerful baseline model that forecasts a future value using the observation from the same period in the previous season (e.g., the value from the same day last week). | `sktime` |
| **SVR** | (Support Vector Regression) A version of Support Vector Machines for regression. It finds a hyperplane in the feature space that best fits the data, but it only penalizes points that fall outside a specified margin ("the tube") around the hyperplane. This makes it robust to some outliers. Requires feature scaling. | `scikit-learn` |
| **Theta Method** | A strong statistical baseline that works by decomposing the time series into two "theta lines." One line represents the long-term trend, and the other captures the short-term behavior. The model extrapolates these components to produce the forecast. | `sktime` |
| **XGBoost** | (Extreme Gradient Boosting) An advanced ensemble model that builds decision trees sequentially, where each new tree corrects the errors of the previous one. It is known for its high performance and speed. Like Random Forest, it requires feature engineering to frame the forecasting problem as a regression task. | `xgboost` |
