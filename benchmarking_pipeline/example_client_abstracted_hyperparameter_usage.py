# ARIMA testing
from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.models.arima_model import ARIMAModel
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner


if __name__ == "__main__":
  print("Starting the program")
  australian_dataloader = DataLoader({"dataset" : {
    "path": "datasets/australian_electricity_demand",
    "name": "australian_electricity_demand",
    "split_ratio" : [0.8, 0.1, 0.1]
    }})