from benchmarking_pipeline.models.arima_model import ARIMAModel
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.models.seasonal_naive_model import SeasonalNaiveModel
from benchmarking_pipeline.models.exponential_smoothing_model import ExponentialSmoothingModel
from benchmarking_pipeline.models.prophet_model import ProphetModel
from benchmarking_pipeline.models.theta_model import ThetaModel

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

def test_seasonal_naive(all_australian_chunks):
  # SECOND MODEL: Seasonal Naive
  seasonal_naive_model = SeasonalNaiveModel({
    "model_params": {
      "sp":"-1"
    },
    "target_col": "y",
    "loss_functions": ["mae"],
    "primary_loss": "mae",
    "forecast_horizon": 100
  })

  seasonal_naive_hyperparameter_tuner = HyperparameterTuner(seasonal_naive_model, {
    "sp": [1,2,3,4]
  }, False)

  # Give the Seasonal Naive model the first chunk to hyperparameter tune on
  validation_score_hyperparameter_tuple = seasonal_naive_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
  
  best_hyperparameters_dict = {
    "sp": validation_score_hyperparameter_tuple[1][0]
    }
  print(f"Final Evaluation Seasonal Naive australia: {seasonal_naive_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
  print(f"Test Evaluation Seasonal Naive australia: {seasonal_naive_hyperparameter_tuner.final_evaluation({'sp':2}, all_australian_chunks)}")
  print("Seasonal Naive WORKS!")

def test_exponential_smooth(all_australian_chunks):
  # THIRD MODEL: Exponential Smoothing
  exponential_smoothing_model = ExponentialSmoothingModel({
    "model_params": {
      "sp":"-1"
    },
    "target_col": "y",
    "loss_functions": ["mae"],
    "primary_loss": "mae",
    "forecast_horizon": 100
  })

  exponential_smoothing_hyperparameter_tuner = HyperparameterTuner(exponential_smoothing_model, {
    "sp": [1,2,3,4]
  }, False)

  # Give the Seasonal Naive model the first chunk to hyperparameter tune on
  validation_score_hyperparameter_tuple = exponential_smoothing_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
  
  best_hyperparameters_dict = {
    "p": validation_score_hyperparameter_tuple[1][0], 
    "d": validation_score_hyperparameter_tuple[1][1], 
    "q": validation_score_hyperparameter_tuple[1][2]
    }
  print(f"Final Evaluation Exponential Smoothing australia: {exponential_smoothing_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
  print(f"Test Evaluation Exponential Smoothing australia: {exponential_smoothing_hyperparameter_tuner.final_evaluation({'p':0,'d':2,'q':0}, all_australian_chunks)}")
  print("Exponential Smoothing WORKS!")

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
  preprocessor = Preprocessor({}) # by default interpolate missing values
  single_chunk = preprocessor.preprocess(single_chunk).data
  all_australian_chunks = [preprocessor.preprocess(chunk).data for chunk in all_australian_chunks]

  test_arima(all_australian_chunks)
  test_seasonal_naive(all_australian_chunks)


  