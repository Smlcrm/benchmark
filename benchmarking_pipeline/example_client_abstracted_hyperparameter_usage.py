# ARIMA testing
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.models.arima_model import ARIMAModel
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.models.lstm_model import LSTMModel
from benchmarking_pipeline.models.random_forest_model import RandomForestModel
from benchmarking_pipeline.models.prophet_model import ProphetModel
import pandas as pd



if __name__ == "__main__":
  print("Starting the program")
  australian_dataloader = DataLoader({"dataset" : {
    "path": "/Users/aryannair/smlcrm-benchmark/benchmarking_pipeline/datasets/australian_electricity_demand",
    "name": "australian_electricity_demand",
    "split_ratio" : [0.8, 0.1, 0.1]
    }})
  single_chunk = australian_dataloader.load_single_chunk(1)
  all_australian_chunks = australian_dataloader.load_several_chunks(2)

  # Preprocess the data with default params
  preprocessor = Preprocessor({}) # by default interpolate missing values
  single_chunk = preprocessor.preprocess(single_chunk).data
  all_australian_chunks = [preprocessor.preprocess(chunk).data for chunk in all_australian_chunks]


  # arima_model = ARIMAModel({
  #   "p": -1,
  #   "d": -1,
  #   "q": -1,
  #   "target_col": "y",
  #   "exog_cols": None,
  #   "loss_functions": ["mae"],
  #   "primary_loss": "mae",
  #   "forecast_horizon": len(single_chunk.test.features["y"])
  # })

  # arima_hyperparameter_tuner = HyperparameterTuner(arima_model,{
  #   "p": [0, 1, 2],
  #   "d": [0, 1],
  #   "q": [0, 1, 2]
  #   }, "arima")
  
  # # Give the ARIMA model the first chunk to hyperparameter tune on
  # print(f"Single chunk: {single_chunk.validation.features}")
  # validation_score_hyperparameter_tuple = arima_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
  # print(f"Validation score and hyperparamter: {validation_score_hyperparameter_tuple}")
  # best_hyperparameters_dict = {
  #   "p": validation_score_hyperparameter_tuple[1][0], 
  #   "d": validation_score_hyperparameter_tuple[1][1], 
  #   "q": validation_score_hyperparameter_tuple[1][2]
  #   }
  # print(f"Final Evaluation: {arima_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
  
  # print("\nLSTM Model Example:")
  # # Use the first chunk's train and validation sets for demonstration
  # lstm_config = {
  #   "units": 32,
  #   "layers": 2,
  #   "dropout": 0.2,
  #   "learning_rate": 0.001,
  #   "batch_size": 16,
  #   "epochs": 5,
  #   "sequence_length": (len(single_chunk.test.features["y"]) // 10),
  #   "target_col": "y",
  #   "loss_functions": ["mae"],
  #   "primary_loss": "mae",
  #   "forecast_horizon": (len(single_chunk.test.features["y"]) // 10)
  # }
  # lstm_model = LSTMModel(lstm_config)
  # lstm_hyperparameter_tuner = HyperparameterTuner(lstm_model,{
  #   "units": [20, 32],
  #   "layers": [1, 2]
  # }, "lstm")

  # validation_score_hyperparameter_tuple = lstm_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
  # print(f"Validation score and hyperparamter: {validation_score_hyperparameter_tuple}")
  # best_hyperparameters_dict = {
  #   "units": validation_score_hyperparameter_tuple[1][0], 
  #   "layers": validation_score_hyperparameter_tuple[1][1]
  #   }
  # print(f"Final Evaluation: {lstm_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")

  # Random Forest Model Example:
  print("\nRandom Forest Model Example:")
  rf_config = {
    "lookback_window": 20,
    "forecast_horizon": 6,  # Smaller horizon for efficiency, will use iterative prediction for longer forecasts
    "model_params": {
      "n_estimators": 10,  # Reduced from 10 for faster testing
      "max_depth": 10,     # Reduced from 10
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "random_state": 42,
      "n_jobs": -1
    }
  }
  rf_hyperparameter_grid = {
    "lookback_window": [20],  # Only one value for testing
    "model_params__n_estimators": [10],  # Reduced from 50 for faster testing
  }
  print(f"Hyperparameter grid: {rf_hyperparameter_grid}")
  print(f"Number of combinations: {len(rf_hyperparameter_grid['lookback_window']) * len(rf_hyperparameter_grid['model_params__n_estimators'])}")
  print(f"Number of time series chunks: {len(all_australian_chunks)}")
  print(f"Forecast horizon: {rf_config['forecast_horizon']} (will use iterative prediction for longer forecasts)")

  rf_model = RandomForestModel(rf_config)
  rf_hyperparameter_tuner = HyperparameterTuner(rf_model, rf_hyperparameter_grid, "random_forest")

  print("Starting hyperparameter tuning...")
  validation_score_hyperparameter_tuple = rf_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
  print(f"Validation score and hyperparamter: {validation_score_hyperparameter_tuple}")

  # Convert numpy array back to dictionary format
  best_hyperparameters_array = validation_score_hyperparameter_tuple[1]
  best_hyperparameters_dict = {
      "lookback_window": int(best_hyperparameters_array[0]),
      "model_params__n_estimators": int(best_hyperparameters_array[1])
  }
  print(f"Best hyperparameters dict: {best_hyperparameters_dict}")

  print(f"Final Evaluation: {rf_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")

  # Prophet Model Example:
  print("\nProphet Model Example:")

  # Prophet configuration and hyperparameter grid
  prophet_config = {
      "model_params": {
          "seasonality_mode": "additive",  # or "multiplicative"
          "changepoint_prior_scale": 0.05,
          "seasonality_prior_scale": 10.0,
          "yearly_seasonality": True,
          "weekly_seasonality": True,
          "daily_seasonality": False,
      }
  }
  prophet_hyperparameter_grid = {
      "model_params__seasonality_mode": ["additive", "multiplicative"],
      "model_params__changepoint_prior_scale": [0.01, 0.05, 0.1],
      "model_params__seasonality_prior_scale": [1.0, 10.0],
  }

  # Prepare y_context and y_target as Pandas Series with DatetimeIndex
  # Example: use the same chunking logic as for other models

  # Use the first chunk for demonstration
  y_context = single_chunk.train.features["y"]
  y_target = single_chunk.validation.features["y"]

  # Ensure they are Pandas Series with DatetimeIndex
  if not isinstance(y_context, pd.Series):
      y_context = pd.Series(y_context, index=single_chunk.train.features.index)
  if not isinstance(y_target, pd.Series):
      y_target = pd.Series(y_target, index=single_chunk.validation.features.index)

  prophet_model = ProphetModel(prophet_config)
  prophet_hyperparameter_tuner = HyperparameterTuner(prophet_model, prophet_hyperparameter_grid, "prophet")

  print("Starting Prophet hyperparameter tuning...")
  validation_score_hyperparameter_tuple = prophet_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
  print(f"Validation score and hyperparameter: {validation_score_hyperparameter_tuple}")

  best_hyperparameters_dict = {
      "model_params__seasonality_mode": validation_score_hyperparameter_tuple[1][0],
      "model_params__changepoint_prior_scale": validation_score_hyperparameter_tuple[1][1],
      "model_params__seasonality_prior_scale": validation_score_hyperparameter_tuple[1][2],
  }
  print(f"Best hyperparameters dict: {best_hyperparameters_dict}")

  print(f"Final Evaluation: {prophet_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
  