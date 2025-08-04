# Forecasting Model Implementations & Explanations

Below is a summary of the various time series forecasting models, how they work, and the key Python libraries used for their implementation.

## Model Overview

The table below outlines each model, its primary use case, and the main library required for its implementation.

| Model | How It Works | Primary Library |
| :--- | :--- | :--- |
| **ARIMA** | (Autoregressive Integrated Moving Average) A classical statistical model that combines three concepts: **(AR)** it uses past values of the series to predict the future; **(I)** it uses differencing to make the series stationary (remove trend/seasonality); and **(MA)** it uses past forecast errors to improve the current forecast. | `statsmodels` |
| **Chronos** | A pretrained time series forecasting model based on language model architectures. It tokenizes time series values into a vocabulary and then predicts the next "token" in the sequence, similar to how an LLM predicts the next word. Creates probabilistic predictions. | `chronos` |
| **Croston Classic** |	A method for intermittent demand (sparse time series with many zeros). It separately forecasts the magnitude of non-zero demand and the time interval between demands using simple exponential smoothing. The final forecast is the ratio of the two smoothed components. | `N/A` |
| **Holt-Winters (Exponential Smoothing)** | A model that generates forecasts based on weighted averages of past observations, with the weights decaying exponentially as the observations get older. It explicitly models and combines error, trend, and seasonality components in an additive or multiplicative manner. | `statsmodels` |
| **LSTM** | (Long Short-Term Memory) A type of Recurrent Neural Network (RNN) that uses internal "gates" (forget, input, and output gates) to regulate the flow of information. This allows it to remember patterns over long sequences while forgetting irrelevant data, making it effective for capturing long-term dependencies. | `TensorFlow`/`Keras` |
| **Moirai** | A large-scale foundation model for universal time series forecasting. It takes patches of a time series as input and uses an encoder-decoder architecture to perform zero-shot forecasting, handling multivariate and univariate scenarios. | `moirai ` |
| **Moment** | A foundation model that represents time series as sequences of statistical moments (e.g., mean, variance). It uses a Transformer architecture to process these representations, enabling and forecast future data. | `momentfm` |
| **Prophet** | A forecasting procedure developed by Meta that is also based on decomposition. It fits an additive model with separate components for: a piecewise linear or logistic trend, multiple seasonalities (yearly, weekly, daily) using Fourier series, and a user-specified list of holidays. | `prophet` |
| **Random Forest** | An ensemble machine learning model that fits multiple decision trees on various sub-samples of the dataset and uses averaging to improve predictive accuracy and control over-fitting. It can capture complex, non-linear relationships between the engineered features (lags, date parts, etc.) and the target. | `scikit-learn` |
| **Seasonal Naive** | A simple but powerful baseline model that forecasts a future value using the observation from the same period in the previous season (e.g., the value from the same day last week). | `sktime` |
| **SVR** | (Support Vector Regression) A version of Support Vector Machines for regression. It finds a hyperplane in the feature space that best fits the data, but it only penalizes points that fall outside a specified margin ("the tube") around the hyperplane. This makes it robust to some outliers. Requires feature scaling. | `scikit-learn` |
| **TabPFN** | A pre-trained Transformer model that performs in-context learning on tabular datasets. It can make predictions on univariate and multivariate data. | `tabpfn` |
| **Theta Method** | A strong statistical baseline that works by decomposing the time series into two "theta lines." One line represents the long-term trend, and the other captures the short-term behavior. The model extrapolates these components to produce the forecast. | `sktime` |
| **TimesFM** | A pre-trained, decoder-only Transformer foundation model developed by Google Research for time-series forecasting. | `timesfm` |
| **TinyTimeMixer** | A lightweight, pre-trained foundation model for forecasting developed by IBM Research. TTMs require less computational power and memory compared to models, and work for multivariate and univariate data. | `tsfm_public ` |
| **XGBoost** | (Extreme Gradient Boosting) An advanced ensemble model that builds decision trees sequentially, where each new tree corrects the errors of the previous one. It is known for its high performance and speed. Like Random Forest, it requires feature engineering to frame the forecasting problem as a regression task. | `xgboost` |
