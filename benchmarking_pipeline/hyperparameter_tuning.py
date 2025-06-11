import pandas as pd
import numpy as np
import ast
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools

def intake_dataset(dataset_name, num_chunks):
  """
  Takes in one of the datasets in the datasets/ folder and combines each chunk to create a Pandas 

  Args:
    dataset_name: The name of the dataset that we want to use for hyperparameter tuning.
    num_chunks: The number of time series chunks that exist per dataset.
  
  Returns:
    A list of Pandas Dataframes, one DataFrame for each time series chunk.
  """
  list_of_dataframes = []
  for idx in range(1,num_chunks+1):
    current_df = pd.read_csv(f"datasets/{dataset_name}/chunk{idx:03}.csv")
    list_of_dataframes.append(current_df)
  return list_of_dataframes

def dataset_preprocess(list_of_dataframes, model_type):
  """
  Processes our List of DataFrames based on the type of model we are passing into our function.

  Args:
    list_of_dataframes: A List of DataFrames. One DataFrame per time series chunk. We can get our List of DataFrames from the intake_dataset(...) method.
    model_type: What kind of model do we want to preprocess our dataset for.

  Returns:
    A list of processed dataset chunks. May contain just a time series or a DataFrame with a time series and other information.
  """
  parsed_target = ""
  if model_type == "ARIMA":
    list_of_processed_dataset_chunks = []
    for dataframe in list_of_dataframes:
      chunk = dataframe.iloc[0]
      target = chunk["target"]
      if isinstance(target, str):
        parsed_target = ast.literal_eval(target)
      else:
        parsed_target = target
      list_of_processed_dataset_chunks.append(parsed_target)
  else:
    raise NotImplementedError("This has not been implemented yet, or is an invalid model type.")
  assert parsed_target != "", "Target column of our dataset has no data"
  return list_of_processed_dataset_chunks

def evaluate_model(time_series, model_hyperparameters, model_type):
  """
  Evaluates our model
  """
  pass

def train_val_test_split(time_series):
  """
  Creates train, validation, and test split from our time series.

  Args:
    time_series: A regular Python List containing our Time Series data.
  """
  train_idx_marker = int(len(time_series) * 0.8)
  val_idx_marker = int(len(time_series) * 0.9)
  train_split = time_series[:train_idx_marker]
  val_split = time_series[train_idx_marker:val_idx_marker]
  test_split = time_series[val_idx_marker:]
  return train_split, val_split, test_split

def hyperparameter_grid_search(time_series, p_list, d_list, q_list, model_type):
  """
  Does hyperparameter grid search.
  """
  best_hyperparameters = None
  best_model = None
  if model_type == "ARIMA":
    lowest_score = float("inf")
    print("Start of train loop!")
    for hyperparameter_setting in itertools.product(p_list, d_list, q_list):
      print(f"Current hyperparameter setting: {hyperparameter_setting}")
      model = ARIMA(time_series, order=hyperparameter_setting)
      model_fit = model.fit()
      aic = model_fit.aic
      if aic < lowest_score:
        lowest_score = aic
        best_hyperparameters = hyperparameter_setting
        best_model = model
    print(f"Lowest Score: {lowest_score}")
  else:
    raise NotImplementedError("This has not been implemented yet, or is an invalid model type.")
  assert best_model != None, "This method was unable to train a good model."
  return best_hyperparameters, best_model




if __name__ == "__main__":
  """
  This main function contains a proof of concept procedure where we do hyperparameter grid search using the ARIMA model. 
  """
  list_of_dataframes = intake_dataset('australian_electricity_demand', 5)
  #print(f"Australian Electricity Demand Dataset: {intake_dataset('australian_electricity_demand', 5)}")
  preprocessed_dataset_chunks = dataset_preprocess(list_of_dataframes,"ARIMA")
  list_of_tuples_of_splits_per_chunk = []
  for processed_chunk in preprocessed_dataset_chunks:
    train, val, test = train_val_test_split(processed_chunk)
    list_of_tuples_of_splits_per_chunk.append((train, val, test))
  p_values = range(0, 4)
  d_values = range(0, 2)
  q_values = range(0, 4)
  list_of_best_arima_model_per_chunk_train_split = []
  for tuple_split in list_of_tuples_of_splits_per_chunk:
    train, val, test = tuple_split
    best_model, best_hyperparameters = hyperparameter_grid_search(train,p_values,d_values, q_values,"ARIMA")
    list_of_best_arima_model_per_chunk_train_split.append(best_model)
  print(f"List of best ARIMA models: {list_of_best_arima_model_per_chunk_train_split}")