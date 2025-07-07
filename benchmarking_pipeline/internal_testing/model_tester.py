from benchmarking_pipeline.models.arima_model import ARIMAModel
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.models.seasonal_naive_model import SeasonalNaiveModel
from benchmarking_pipeline.models.exponential_smoothing_model import ExponentialSmoothingModel
from benchmarking_pipeline.models.prophet_model import ProphetModel
from benchmarking_pipeline.models.theta_model import ThetaModel
from benchmarking_pipeline.models.deepAR_model import DeepARModel
from benchmarking_pipeline.models.croston_classic_model import CrostonClassicModel
import pandas as pd
import numpy as np

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
    "hidden_size": 10,
    "rnn_layers" : -1,
    "dropout" : 0.1,
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

  shorten_australia = True
  if shorten_australia:
    for australian_chunk in all_australian_chunks:
      #print("train", pd.Series(australian_chunk.train.features[6300:].squeeze(),index=list(range(900))).shape)
      australian_chunk.train.features = pd.DataFrame({'y': australian_chunk.train.features[7000:].squeeze()})
      australian_chunk.train.features.reset_index()
      print("train", australian_chunk.train.features)

  validation_score_hyperparameter_tuple = deep_ar_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)


  best_hyperparameters_dict = {
    "rnn_layers": validation_score_hyperparameter_tuple[1][0], 
    }
  

  
  print(f"Final Evaluation DeepAR australia: {deep_ar_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
  print(f"Test Evaluation DeepAR australia: {deep_ar_hyperparameter_tuner.final_evaluation({'rnn_layers':2,}, all_australian_chunks)}")
  print("Deep AR WORKS!")

def test_croston_classic(all_australian_chunks):
    croston_model = CrostonClassicModel({
        "alpha": 0.1,
        "target_col": "y",
        "loss_functions": ["mae"],
        "primary_loss": "mae",
        "forecast_horizon": 100
    })

    # Example: No hyperparameter tuning, just fit and evaluate
    for i, chunk in enumerate(all_australian_chunks):
        y = chunk["y"] if isinstance(chunk, dict) else chunk

        print(len(y))
        # Extract a 1D array for y_target
        if isinstance(y, pd.Series):
            y_data = y.values
        else:
            y_data = np.asarray(y)

        # Ensure y_data is at least 1D before slicing
        y_data = np.atleast_1d(y_data)

        print(type(y_data[0]))

        croston_model.train(y_context=y_data)
        preds = croston_model.predict(y_context=y_data, y_target=y_data[-100:])
        print(f"Chunk {i} Croston predictions (first 5): {preds}")
        print(f"Model summary: {croston_model.get_model_summary()}")

    print("Croston Classic WORKS!")

if __name__ == "__main__":
  print("Model testing suite!")
  australian_dataloader = DataLoader({"dataset" : {
    "path": "/Users/chenk/benchmark/benchmarking_pipeline/datasets/australian_electricity_demand",
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

  #test_arima(all_australian_chunks)
  #test_seasonal_naive(all_australian_chunks)
  #test_theta(all_australian_chunks)
  #test_deep_ar(all_australian_chunks)
  test_croston_classic(all_australian_chunks)