# %%
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from benchmarking_pipeline.models.arima_model import ARIMAModel
from benchmarking_pipeline.models.theta_model import ThetaModel
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
import numpy as np
import re

# Load first chunk of ARIMA dataset
print("Model testing suite!")
australian_dataloader = DataLoader({"dataset" : {
  "path": "/Users/alifabdullah/Collaboration/Simulacrum-Benchmark/benchmark/benchmarking_pipeline/datasets/australian_electricity_demand",
  "name": "australian_electricity_demand",
  "split_ratio" : [0.8, 0.1, 0.1]
  }})
single_chunk = australian_dataloader.load_single_chunk(1)


# Preprocess the data with default params
preprocessor = Preprocessor({"dataset":
                              {"normalize":False}
                              }) # by default interpolate missing values
all_australian_chunks = [preprocessor.preprocess(single_chunk).data]

def _extract_number_before_capital(freq_str):
    match = re.match(r'(\d+)?[A-Z]', freq_str)
    if match:
        return int(match.group(1)) if match.group(1) else 1
    else:
        raise ValueError(f"Invalid frequency string: {freq_str}")

def hyperparameter_grid_search_several_time_series(model_name, model, hyperparameter_ranges, list_of_time_series_datasets):
    """
    Perform grid search over hyperparameter combinations for multiple time series datasets, 
    then identify the model with the best average validation performance across the datasets.

    Args:
        list_of_time_series_datasets: A list of Dataset objects, each containing 'train', 'validation', and 'test' splits.

    Returns:
        best_arima_model_overall: The model instance that achieved the best average validation loss across datasets.
        best_hyperparameters_overall: The hyperparameter values (as a tuple) that correspond to the best model.
        
    """
    list_of_validation_scores = []
    list_of_hyperparameters_per_validation_score = []

    for time_series_dataset in list_of_time_series_datasets:
      best_hyperparameters = 0
      lowest_train_loss = float("inf")
      print("Starting hyperparameter grid search on single time series!")

      # List of hyperparameter names
      hyperparameter_names = list(hyperparameter_ranges.keys())
      # We iterate over all possible hyperparameter value combinations
      for hyperparameter_setting in itertools.product(*hyperparameter_ranges.values()):
        current_hyperparameter_dict = dict()
        for key_value_index in range(len(hyperparameter_names)):

          # For the current chosen hyperparameter combination,
          # Create a dictionary associating hyperparameter names to the 
          # Appropriate hyperparameter value
          current_hyperparameter_dict[hyperparameter_names[key_value_index]] = hyperparameter_setting[key_value_index]
          print(f"Current hyperparameter dict: {current_hyperparameter_dict}")
          
        # Change the model's hyperparameter values
        model.set_params(**current_hyperparameter_dict)
        # Train a new model
        print(f"DEBUG: train.features columns: {time_series_dataset.train.features.columns}")
        print(f"DEBUG: train.features head:\n{time_series_dataset.train.features.head()}")
        print(f"DEBUG: model.target_col: {model.target_col}")
        target = time_series_dataset.train.features[model.target_col].values
        validation_series = time_series_dataset.validation.features[model.target_col].values
        start_date = time_series_dataset.metadata["start"]
        freq_str = time_series_dataset.metadata["freq"]
        first_capital_letter_finder = re.search(r'[A-Z]', freq_str)
        freq = first_capital_letter_finder.group()
        freq_coefficient = _extract_number_before_capital(freq_str)
        freq_offset = pd.tseries.frequencies.to_offset(freq)
        x_start_date = pd.to_datetime(start_date)
        y_start_date = x_start_date + (freq_coefficient * len(target) * freq_offset)
        y_context_timestamps = time_series_dataset.train.timestamps
        y_target_timestamps = time_series_dataset.validation.timestamps

        # Handle different model types with different train method signatures
        trained_model = model.train(y_context=target, y_target=validation_series, y_context_timestamps=y_context_timestamps, y_target_timestamps=y_target_timestamps, x_start_date=x_start_date)

        # Get validation losses over every chunk
        current_train_loss = 0
        for time_series_dataset_from_all in list_of_time_series_datasets:
          target = time_series_dataset_from_all.train.features[model.target_col].values
          validation_series = time_series_dataset_from_all.validation.features[model.target_col].values
          #print(f"Time series dataset from all datasets\n\n:{time_series_dataset_from_all.metadata}")
          # Handle different model types with different predict method signatures
          model_predictions = trained_model.predict(y_context=target, y_target=validation_series, y_target_timestamps=time_series_dataset_from_all.validation.timestamps)

        true_data = np.concatenate((target, validation_series))
        plotting_start = len(true_data)
        prediction_start = len(target)
        
        plt.plot(range(plotting_start),true_data,color="orange")
        if model_name == "arima":
          plt.plot(range(prediction_start, prediction_start+model_predictions.shape[1]), model_predictions[0],color="blue")
        elif model_name == "theta":
           plt.plot(range(prediction_start, prediction_start+model_predictions.shape[0]), model_predictions,color="blue")
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        

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

  hyperparameter_ranges = {
    "p": [10,20,40,80],
    "d": [1],
    "q": [0]
    }
  
  # Give the ARIMA model the first chunk to hyperparameter tune on
  hyperparameter_grid_search_several_time_series("arima",arima_model, hyperparameter_ranges, all_australian_chunks)

def test_theta(all_australian_chunks):
  theta_model = ThetaModel({
    'sp':-1,
    "forecast_horizon": 100
    })

  hyperparameter_ranges = {
     'sp':[300,350,400,800,1200,1600,2048,2500,3000,3100,3200,3230]  
    #'sp':[256,300,350,400,512,1024,2048,4000]  
      }
  
  hyperparameter_grid_search_several_time_series("theta",theta_model,hyperparameter_ranges,all_australian_chunks)
  
      
#test_arima(all_australian_chunks)
test_theta(all_australian_chunks)
