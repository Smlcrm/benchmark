# ARIMA testing
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.models.arima_model import ARIMAModel
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner


if __name__ == "__main__":
  print("Starting the program")
  australian_dataloader = DataLoader({"dataset" : {
    "path": "/Users/alifabdullah/Collaboration/benchmark/benchmarking_pipeline/datasets/australian_electricity_demand",
    "name": "australian_electricity_demand",
    "split_ratio" : [0.8, 0.1, 0.1]
    }})
  single_chunk = australian_dataloader.load_single_chunk(1)
  all_australian_chunks = australian_dataloader.load_several_chunks(5)

  arima_model = ARIMAModel({
    "p": -1,
    "d": -1,
    "q": -1,
    "target_col": "y",
    "exog_cols": None,
    "loss_functions": ["mae"],
    "metrics_to_calculate": ["mae"],
    "primary_loss": "mae",
    "forecast_horizon": 900
  })

  arima_hyperparameter_tuner = HyperparameterTuner(arima_model,{
    "p":range(0,4),
    "d":range(0,2),
    "q":range(0,4)
    })
  
  # Give the ARIMA model the first chunk to hyperparameter tune on
  print(f"Single chunk: {single_chunk.validation.features}")
  model_hyperparameter_tuple = arima_hyperparameter_tuner.hyperparameter_grid_search_single_time_series(all_australian_chunks[0])
  print(f"Model and hyperparamter: {model_hyperparameter_tuple}")

  