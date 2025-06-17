import itertools
from ..models.base_model import BaseModel

class HyperparameterTuner:
  def __init__(self, model_class: BaseModel, *hyperparameter_ranges):
    self.model_class = model_class
    self.hyperparameter_ranges = hyperparameter_ranges
  
  def hyperparameter_grid_search_single_time_series(self, time_series):
    best_trained_model = None
    best_hyperparameters = None
    lowest_train_loss = float("inf")
    for hyperparameter_setting in itertools.product(*self.hyperparameter_ranges):
      trained_model = self.model_class.train(time_series)
      current_train_loss = self.model_class.evaluate(time_series)

      if current_train_loss < lowest_train_loss:
        best_trained_model = trained_model
        best_hyperparameters = hyperparameter_setting
    
    return best_trained_model, best_hyperparameters
  
  