# ARIMA testing
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.models.arima_model import ARIMAModel
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.models.lstm_model import LSTMModel


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


  arima_model = ARIMAModel({
    "p": -1,
    "d": -1,
    "q": -1,
    "target_col": "y",
    "exog_cols": None,
    "loss_functions": ["mae"],
    "primary_loss": "mae",
    "forecast_horizon": len(single_chunk.test.features["y"])
  })

  arima_hyperparameter_tuner = HyperparameterTuner(arima_model,{
    "p": [0, 1, 2],
    "d": [0, 1],
    "q": [0, 1, 2]
    }, "arima")
  
  # Give the ARIMA model the first chunk to hyperparameter tune on
  print(f"Single chunk: {single_chunk.validation.features}")
  validation_score_hyperparameter_tuple = arima_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks, False)
  print(f"Validation score and hyperparamter: {validation_score_hyperparameter_tuple}")
  best_hyperparameters_dict = {
    "p": validation_score_hyperparameter_tuple[1][0], 
    "d": validation_score_hyperparameter_tuple[1][1], 
    "q": validation_score_hyperparameter_tuple[1][2]
    }
  print(f"Final Evaluation: {arima_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks, False)}")
  
  print("\nLSTM Model Example:")
  # Use the first chunk's train and validation sets for demonstration
  lstm_config = {
    "units": 32,
    "layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 5,
    "sequence_length": (len(single_chunk.test.features["y"]) // 10),
    "target_col": "y",
    "loss_functions": ["mae"],
    "primary_loss": "mae",
    "forecast_horizon": (len(single_chunk.test.features["y"]) // 10)
  }
  lstm_model = LSTMModel(lstm_config)
  lstm_hyperparameter_tuner = HyperparameterTuner(lstm_model,{
    "units": [20, 32],
    "layers": [1, 2]
  }, "lstm")

  validation_score_hyperparameter_tuple = lstm_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
  print(f"Validation score and hyperparamter: {validation_score_hyperparameter_tuple}")
  best_hyperparameters_dict = {
    "units": validation_score_hyperparameter_tuple[1][0], 
    "layers": validation_score_hyperparameter_tuple[1][1]
    }
  print(f"Final Evaluation: {lstm_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)}")

  