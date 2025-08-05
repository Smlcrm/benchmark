# ARIMA testing
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.models.arima.arima_model import ARIMAModel
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.models.lstm_model import LSTMModel
from .testing_utilities import less_than


if __name__ == "__main__":
  print("Hyperparameter testing suite!")
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
  print(f"Final Evaluation australia: {arima_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")
  print(f"Test Evaluation australia: {arima_hyperparameter_tuner.final_evaluation({'p':0,'d':2,'q':0}, all_australian_chunks)}")

  test_cases = []

  # Test Case 1: ARIMA Australia
  test_cases.append((arima_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks),
                     arima_hyperparameter_tuner.final_evaluation({'p':0,'d':2,'q':0}, all_australian_chunks)))
  

  # Test Case 2: ARIMA bdg-2_bear
  bdg_dataloader = DataLoader({"dataset" : {
    "path": "/Users/aryannair/smlcrm-benchmark/benchmarking_pipeline/datasets/bdg-2_bear",
    "name": "bdg-2_bear",
    "split_ratio" : [0.8, 0.1, 0.1]
    }})
  
  all_bdg_chunks = bdg_dataloader.load_several_chunks(2)
  all_bdg_chunks = [preprocessor.preprocess(chunk).data for chunk in all_bdg_chunks]

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

  validation_score_hyperparameter_tuple = arima_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_bdg_chunks)
  
  best_hyperparameters_dict = {
    "p": validation_score_hyperparameter_tuple[1][0], 
    "d": validation_score_hyperparameter_tuple[1][1], 
    "q": validation_score_hyperparameter_tuple[1][2]
    }
  
  print(f"Final Evaluation bdg: {arima_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_bdg_chunks)}")
  print(f"Test Evaluation bdg: {arima_hyperparameter_tuner.final_evaluation({'p':0,'d':2,'q':0}, all_bdg_chunks)}")

  test_cases.append((arima_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_bdg_chunks),
                     arima_hyperparameter_tuner.final_evaluation({'p':0,'d':2,'q':0}, all_bdg_chunks)))
  


  # Test Case 3: ARIMA Loop Seattle
  loop_seattle_dataloader = DataLoader({"dataset" : {
    "path": "/Users/aryannair/smlcrm-benchmark/benchmarking_pipeline/datasets/LOOP_SEATTLE",
    "name": "LOOP_SEATTLE",
    "split_ratio" : [0.8, 0.1, 0.1]
    }})
  
  all_loop_seattle_chunks = loop_seattle_dataloader.load_several_chunks(3)
  all_loop_seattle_chunks = [preprocessor.preprocess(chunk).data for chunk in all_loop_seattle_chunks]

  """
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
  """

  arima_hyperparameter_tuner = HyperparameterTuner(arima_model,{
    "p": [0, 1, 2],
    "d": [0, 1],
    "q": [0, 1, 2]
    }, False)

  validation_score_hyperparameter_tuple = arima_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_loop_seattle_chunks)
  
  best_hyperparameters_dict = {
    "p": validation_score_hyperparameter_tuple[1][0], 
    "d": validation_score_hyperparameter_tuple[1][1], 
    "q": validation_score_hyperparameter_tuple[1][2]
    }
  
  print(f"Final Evaluation loop seattle: {arima_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_loop_seattle_chunks)}")
  print(f"Test Evaluation loop seattle: {arima_hyperparameter_tuner.final_evaluation({'p':0,'d':2,'q':0}, all_loop_seattle_chunks)}")

  test_cases.append((arima_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_loop_seattle_chunks),
                     arima_hyperparameter_tuner.final_evaluation({'p':0,'d':2,'q':0}, all_loop_seattle_chunks)))
  

  test_case_idx = 0
  for test_case in test_cases:
    assert less_than(test_case[0], test_case[1]), f"Test case {test_case_idx} failed.\n\n'{test_case[0]}'\n\nand\n\n'{test_case[1]}'\n\nare not equivalent."
    test_case_idx += 1
  
  
  
  print("Got through without any issues!")
  