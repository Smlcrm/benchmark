# ARIMA testing
from pipeline.data_loader import DataLoader
from models.arima_model import ARIMAModel
from trainer.hyperparameter_tuning import HyperparameterTuner


if __name__ == "__main__":
  australian_dataloader = DataLoader({"dataset" : {
    "path": "datasets/australian_electricity_demand",
    "name": "australian_electricity_demand",
    "split_ratio" : [0.8, 0.1, 0.1]
    }})