import itertools
from ..models.base_model import BaseModel
from typing import Dict, List

class HyperparameterTuner:
  def __init__(self, model_class: BaseModel, hyperparameter_ranges: Dict[str, List]):
    self.model_class = model_class
    self.hyperparameter_ranges = hyperparameter_ranges
  
  def hyperparameter_grid_search_single_time_series(self, time_series_dataset):
    best_trained_model = None
    best_hyperparameters = None
    lowest_train_loss = float("inf")
    # List of hyperparameter names
    hyperparameter_names = list(self.hyperparameter_ranges.keys())
    # We iterate over all possible hyperparameter value combinations
    for hyperparameter_setting in itertools.product(*self.hyperparameter_ranges.values()):
      current_hyperparameter_dict = dict()
      for key_value_index in range(len(hyperparameter_names)):
        # For the current chosen hyperparameter combination,
        # Create a dictionary associating hyperparameter names to the 
        # Appropriate hyperparameter value
        current_hyperparameter_dict[hyperparameter_names[key_value_index]] = hyperparameter_setting[key_value_index]
      # Change the model's hyperparameter values
      print(f"Current hyperparameter setting: {hyperparameter_setting}")
      self.model_class.set_params(**current_hyperparameter_dict)
      # Train a new model
      trained_model = self.model_class.train(None,time_series_dataset.train.features[self.model_class.target_col])
      model_predictions = self.model_class.predict(None)
      current_train_loss = self.model_class.compute_loss(time_series_dataset.validation.features[self.model_class.target_col], model_predictions)
      #print(f"Current Train Loss: {current_train_loss}")
      if current_train_loss[self.model_class.primary_loss] < lowest_train_loss:
        print(f"Lowest train loss {lowest_train_loss}")
        lowest_train_loss = current_train_loss[self.model_class.primary_loss]
        best_trained_model = trained_model
        best_hyperparameters = hyperparameter_setting
    
    return best_trained_model, best_hyperparameters

  def hyperparameter_grid_search_several_time_series(self, list_of_time_series_datasets):
    list_of_best_trained_models = []
    list_of_best_hyperparameters_per_model = []
    for time_series_dataset in list_of_time_series_datasets:
      trained_model, hyperparameters = self.hyperparameter_grid_search_single_time_series(time_series_dataset)
      list_of_best_trained_models.append(trained_model)
      list_of_best_hyperparameters_per_model.append(hyperparameters)
    pass
  
  