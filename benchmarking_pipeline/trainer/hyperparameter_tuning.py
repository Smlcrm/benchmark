import itertools
from ..models.base_model import BaseModel
from typing import Dict, List
import numpy as np

class HyperparameterTuner:
  def __init__(self, model_class: BaseModel, hyperparameter_ranges: Dict[str, List]):
    """
    Initialize the hyperparameter tuner with a model class and a hyperparameter search space.

    Args:
        model_class: A model class that inherits from BaseModel
        hyperparameter_ranges: A dictionary where keys are hyperparameter names and values are lists of values to search over.
    """
    self.model_class = model_class
    self.hyperparameter_ranges = hyperparameter_ranges
  
  def hyperparameter_grid_search_single_time_series(self, time_series_dataset):
    """
    Perform grid search over hyperparameter combinations for a single time series dataset.

    Args:
        time_series_dataset: A Dataset object containing 'train', 'validation', and 'test' splits.

    Returns:
        best_trained_model: The model instance trained with the best hyperparameter setting.
        best_hyperparameters: The hyperparameter values (as a tuple) that achieved the lowest validation loss.
    """
    best_hyperparameters = None
    lowest_train_loss = float("inf")
    print("Starting hyperparameter grid search on single time series!")
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
      self.model_class.set_params(**current_hyperparameter_dict)
      # Train a new model
      trained_model = self.model_class.train(None,time_series_dataset.train.features[self.model_class.target_col])
      model_predictions = self.model_class.predict(None)
      current_train_loss = self.model_class.compute_loss(time_series_dataset.validation.features[self.model_class.target_col], model_predictions)
      #print(f"Current Train Loss: {current_train_loss}")
      if current_train_loss[self.model_class.primary_loss] < lowest_train_loss:
        print(f"Lowest train loss {lowest_train_loss}")
        lowest_train_loss = current_train_loss[self.model_class.primary_loss]
        best_hyperparameters = hyperparameter_setting
    
    return lowest_train_loss, best_hyperparameters

  def hyperparameter_grid_search_several_time_series(self, list_of_time_series_datasets):
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
      validation_score, hyperparameters = self.hyperparameter_grid_search_single_time_series(time_series_dataset)
      list_of_validation_scores.append(validation_score)
      list_of_hyperparameters_per_validation_score.append(hyperparameters)

    list_of_validation_scores = np.array(list_of_validation_scores)
    list_of_hyperparameters_per_validation_score = np.array(list_of_hyperparameters_per_validation_score)
    print(f"Best hyperparameters: {list_of_hyperparameters_per_validation_score}")
    print(f"Best validation scores: {list_of_validation_scores}")
    print(f"Validation scores: {list_of_validation_scores}")
    print(f"Chosen index: {list_of_validation_scores.argmin()}")
    best_hyperparameters_overall = list_of_hyperparameters_per_validation_score[list_of_validation_scores.argmin()]
    return list_of_validation_scores.min(), best_hyperparameters_overall

  def final_evaluation(self, best_hyperparamters: Dict[str, int], list_of_time_series_datasets):
    """
    Train a model using the best hyperparameters on the combined train and validation splits, 
    then evaluate its performance on the test split for each dataset.

    Args:
        best_hyperparamters: A dictionary mapping hyperparameter names to their best-tuned values.
        list_of_time_series_datasets: A list of Datasets, each containing 'train', 'validation', and 'test' splits.

    Returns:
        results_dict: A dictionary mapping each loss metric name to its average value across datasets on the test split.
    """
    self.model_class.set_params(**best_hyperparamters)
    results_dict = None
    for time_series_dataset in list_of_time_series_datasets:
      train_val_split = np.concatenate([
            time_series_dataset.train.features[self.model_class.target_col],
            time_series_dataset.validation.features[self.model_class.target_col]
        ])
      self.model_class.train(None, train_val_split)
      predictions = self.model_class.predict(None)
      train_loss_dict = self.model_class.compute_loss(time_series_dataset.test.features[self.model_class.target_col], predictions)
      if results_dict is None:
        results_dict = train_loss_dict
      else:
        # aggregates dict
        results_dict = {key: results_dict[key] + train_loss_dict[key] for key in results_dict}
    results_dict = {key: float(results_dict[key]/len(list_of_time_series_datasets)) for key in results_dict}
    return results_dict
      
    
  