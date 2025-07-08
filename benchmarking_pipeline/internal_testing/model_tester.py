from benchmarking_pipeline.models.arima_model import ARIMAModel
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.models.seasonal_naive_model import SeasonalNaiveModel
from benchmarking_pipeline.models.exponential_smoothing_model import ExponentialSmoothingModel
from benchmarking_pipeline.models.prophet_model import ProphetModel
from benchmarking_pipeline.models.theta_model import ThetaModel
from benchmarking_pipeline.models.deepAR_model import DeepARModel
from benchmarking_pipeline.models.xgboost_model import XGBoostModel
from benchmarking_pipeline.models.random_forest_model import RandomForestModel
from benchmarking_pipeline.models.lstm_model import LSTMModel
from benchmarking_pipeline.models.croston_classic_model import CrostonClassicModel
from benchmarking_pipeline.models.lstm_model import LSTMModel
from benchmarking_pipeline.models.random_forest_model import RandomForestModel
import pandas as pd
import re
import numpy as np


def _extract_number_before_capital(freq_str):
    match = re.match(r'(\d+)?[A-Z]', freq_str)
    if match:
        return int(match.group(1)) if match.group(1) else 1
    else:
        raise ValueError(f"Invalid frequency string: {freq_str}")


def test_arima(all_australian_chunks):

  # FIRST MODEL: ARIMA
  arima_model = ARIMAModel({
    "p": -1,
    "d": -1,
    "q": -1,
    "target_col": "y",
    "exog_cols": None,
    "loss_functions": ["mae"],
    "primary_loss": "mae",
    "forecast_horizon": 100
  })

  arima_hyperparameter_tuner = HyperparameterTuner(arima_model,{
    "p": [0, 1, 2],
    "d": [0, 1],
    "q": [0, 1, 2]
    }, False)
  
  # Give the ARIMA model the first chunk to hyperparameter tune on
  validation_score_hyperparameter_tuple = arima_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
  
  best_hyperparameters_dict = {
    "p": validation_score_hyperparameter_tuple[1][0], 
    "d": validation_score_hyperparameter_tuple[1][1], 
    "q": validation_score_hyperparameter_tuple[1][2]
    }
  print(f"Final Evaluation ARIMA australia: {arima_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
  print(f"Test Evaluation ARIMA australia: {arima_hyperparameter_tuner.final_evaluation({'p':0,'d':2,'q':0}, all_australian_chunks)}")
  print("ARIMA WORKS!")

def test_theta(all_australian_chunks):
  theta_model = ThetaModel({
    'sp':-1,
    "forecast_horizon": 100
    })

  theta_hyperparameter_tuner = HyperparameterTuner(theta_model, {
    'sp':[1,2,3,4]  
      }, False)
  
  validation_score_hyperparameter_tuple = theta_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
  print(validation_score_hyperparameter_tuple)

  print(f"Testing the theta model's get_params(...) method:\n\n{theta_model.get_params()}")

  best_hyperparameters_dict = {
    "sp": validation_score_hyperparameter_tuple[1][0], 
    }
  print(f"Final Evaluation Theta australia: {theta_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
  print(f"Test Evaluation Theta australia: {theta_hyperparameter_tuner.final_evaluation({'sp':3,}, all_australian_chunks)}")
  print("Theta WORKS!")

def test_deep_ar(all_australian_chunks):
  deep_ar_model = DeepARModel({
    "hidden_size": 2,
    "rnn_layers" : 2,
    "dropout" : 0.1,
    "batch_size" : 64,
    "learning_rate" : 0.001,
    "target_col" : 'y',
    "feature_cols" : None,
    "forecast_horizon" : 100,
    "epochs": 1,
    "num_workers":8
  })

  deep_ar_hyperparameter_tuner = HyperparameterTuner(deep_ar_model, {
    'rnn_layers' : [2,3],
  }, False)

  shorten_australia = False
  if shorten_australia:
    for australian_chunk in all_australian_chunks:
      #print("train", pd.Series(australian_chunk.train.features[6300:].squeeze(),index=list(range(900))).shape)
      australian_chunk.train.features = pd.DataFrame({'y': australian_chunk.train.features[7000:].squeeze()})
      australian_chunk.train.features.reset_index()
      print("train", australian_chunk.train.features)
  

  time_series_dataset = all_australian_chunks[0]
  deep_ar_model.set_params(rnn_layers=2)
  target = time_series_dataset.train.features[deep_ar_model.target_col]
  validation_series = time_series_dataset.validation.features[deep_ar_model.target_col]
  start_date = time_series_dataset.metadata["start"]
  freq_str = time_series_dataset.metadata["freq"]
  first_capital_letter_finder = re.search(r'[A-Z]', freq_str)
  freq = first_capital_letter_finder.group()
  freq_coefficient = _extract_number_before_capital(freq_str)
  freq_offset = pd.tseries.frequencies.to_offset(freq)
  x_start_date = pd.to_datetime(start_date)
  y_start_date = x_start_date + (freq_coefficient * len(target) * freq_offset)

  """print("Starting training")
  trained_model = deep_ar_model.train(y_context=target, y_target=validation_series, y_start_date=y_start_date, x_start_date=x_start_date)
  print("Done training; starting predicting")
  model_predictions = trained_model.predict(y_context=target, y_target=validation_series)
  print("Done predicting; starting computing loss")
  train_loss = trained_model.compute_loss(time_series_dataset.validation.features[deep_ar_model.target_col], model_predictions)
  print(train_loss)
  print("Done computing loss")"""

  
  validation_score_hyperparameter_tuple = deep_ar_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)


  best_hyperparameters_dict = {
    "rnn_layers": validation_score_hyperparameter_tuple[1][0], 
    }
  

  
  print(f"Final Evaluation DeepAR australia: {deep_ar_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
  print(f"Test Evaluation DeepAR australia: {deep_ar_hyperparameter_tuner.final_evaluation({'rnn_layers':2,}, all_australian_chunks)}")
  print("Deep AR WORKS!")
  

def test_xgboost(all_australian_chunks):
  xgb_model = XGBoostModel({
    "lookback_window": 10,
    "forecast_horizon": 10,
    "model_params": {
      "n_estimators": 50,
      "max_depth": 5,
      "learning_rate": 0.1,
      "random_state": 42,
      "n_jobs": -1
    }
  })

  xgb_hyperparameter_tuner = HyperparameterTuner(xgb_model, {
    "lookback_window": [10],
    "model_params__n_estimators": [50],
    "model_params__max_depth": [3],
    "model_params__learning_rate": [0.01],
  }, False)

  validation_score_hyperparameter_tuple = xgb_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)

  best_hyperparameters_dict = {
    "lookback_window": validation_score_hyperparameter_tuple[1][0],
    "model_params__n_estimators": validation_score_hyperparameter_tuple[1][1],
    "model_params__max_depth": validation_score_hyperparameter_tuple[1][2],
    "model_params__learning_rate": validation_score_hyperparameter_tuple[1][3],
  }
  print(f"Final Evaluation XGBoost australia: {xgb_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
  print(f"Test Evaluation XGBoost australia: {xgb_hyperparameter_tuner.final_evaluation({'lookback_window': 10, 'model_params__n_estimators': 50, 'model_params__max_depth': 5, 'model_params__learning_rate': 0.1}, all_australian_chunks)}")
  print("XGBoost WORKS!")

def test_random_forest(all_australian_chunks):
    rf_model = RandomForestModel({
        "lookback_window": 10,
        "forecast_horizon": 5,
        
        "n_estimators": 50,
        "max_depth": 5,
        "random_state": 42,
        "n_jobs": -1

    })

    rf_hyperparameter_tuner = HyperparameterTuner(rf_model, {
        "lookback_window": [10, 20],
        "n_estimators": [10, 50],
        "max_depth": [3,7],
    }, False)

    validation_score_hyperparameter_tuple = rf_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)

    best_hyperparameters_dict = {
        "lookback_window": validation_score_hyperparameter_tuple[1][0],
        "n_estimators": validation_score_hyperparameter_tuple[1][1],
        "max_depth": validation_score_hyperparameter_tuple[1][2],
    }
    print(f"Final Evaluation Random Forest australia: {rf_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
    print(f"Test Evaluation Random Forest australia: {rf_hyperparameter_tuner.final_evaluation({'lookback_window': 10, 'model_params__n_estimators': 50, 'model_params__max_depth': 5}, all_australian_chunks)}")
    print("Random Forest WORKS!")

def test_prophet(all_australian_chunks):

    prophet_model = ProphetModel({
        "model_params": {
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
        }
    })

    prophet_hyperparameter_tuner = HyperparameterTuner(prophet_model, {
        "seasonality_mode": ["additive", "multiplicative"],
        "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
    }, "prophet")

    validation_score_hyperparameter_tuple = prophet_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)

    best_hyperparameters_dict = {
        "seasonality_mode": validation_score_hyperparameter_tuple[1][0],
        "changepoint_prior_scale": validation_score_hyperparameter_tuple[1][1],
        "seasonality_prior_scale": validation_score_hyperparameter_tuple[1][2],
    }
    print(f"Final Evaluation Prophet australia: {prophet_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
    print(f"Test Evaluation Prophet australia: {prophet_hyperparameter_tuner.final_evaluation({'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0}, all_australian_chunks)}")
    print("Prophet WORKS!")


def test_croston_classic(all_australian_chunks):
    y_train = np.array([0, 0, 5, 0, 0, 0, 3, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 6, 0, 0, 1, 4, 0, 0, 5, 19, 0, 0, 0])
    y_target = np.zeros(5)  # Forecast 5 steps ahead

    croston_model = CrostonClassicModel({
        "alpha": 0.1,
        "target_col": "y",
        "loss_functions": ["mae"],
        "primary_loss": "mae",
        "forecast_horizon": 5
    })

    croston_model.train(y_context=y_train)
    preds = croston_model.predict(y_context=y_train, y_target=y_target)
    print("Synthetic Croston predictions:", preds.flatten())
    print("Model summary:", croston_model.get_model_summary())

def test_lstm(all_australian_chunks):
    # LSTM model configuration
    lstm_model = LSTMModel({
        "units": 32,
        "layers": 1,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 20,
        "sequence_length": 20,
        "target_col": "y",
        "loss_functions": ["mae"],
        "primary_loss": "mae",
        "forecast_horizon": 10
    })

    lstm_hyperparameter_tuner = HyperparameterTuner(lstm_model, {
        "units": [32],
        "layers": [1],
    }, False)

    validation_score_hyperparameter_tuple = lstm_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)

    best_hyperparameters_dict = {
        "units": validation_score_hyperparameter_tuple[1][0],
        "layers": validation_score_hyperparameter_tuple[1][1]
    }

    print(f"Final Evaluation LSTM australia: {lstm_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
    print(f"Test Evaluation LSTM australia: {lstm_hyperparameter_tuner.final_evaluation({'units': 32, 'layers': 2}, all_australian_chunks)}")
    print("LSTM WORKS!")

if __name__ == "__main__":
  print("Model testing suite!")
  australian_dataloader = DataLoader({"dataset" : {
    "path": "/Users/alifabdullah/Collaboration/benchmark/benchmarking_pipeline/datasets/australian_electricity_demand",
    "name": "australian_electricity_demand",
    "split_ratio" : [0.8, 0.1, 0.1]
    }})
  single_chunk = australian_dataloader.load_single_chunk(1)
  all_australian_chunks = australian_dataloader.load_several_chunks(2)

  # Preprocess the data with default params
  preprocessor = Preprocessor({"dataset":
                               {"normalize":False}
                               }) # by default interpolate missing values
  single_chunk = preprocessor.preprocess(single_chunk).data
  all_australian_chunks = [preprocessor.preprocess(chunk).data for chunk in all_australian_chunks]


  # test_arima(all_australian_chunks)
  # test_seasonal_naive(all_australian_chunks)
  # test_theta(all_australian_chunks)
  test_deep_ar(all_australian_chunks)
  # test_xgboost(all_australian_chunks)
  # test_random_forest(all_australian_chunks)
  # test_prophet(all_australian_chunks)
  # test_lstm(all_australian_chunks)
  # test_croston_classic(all_australian_chunks)



