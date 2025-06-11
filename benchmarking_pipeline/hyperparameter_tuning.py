import statsmodels
import pandas as pd
import numpy as np
import ast
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def intake_dataset(dataset_name, num_chunks):
  """
  Takes in one of the datasets in the datasets/ folder and combines each chunk to create a Pandas 

  Args:
    dataset_name: The name of the dataset that we want to use for hyperparameter tuning.
    num_chunks: The number of chunks that exist per dataset.
  
  Returns:
    A list of Pandas Dataframes, one for each dataset.
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
    list_of_dataframes: A List of DataFrames. One DataFrame per dataset chunk. We can get our List of DataFrames from the intake_dataset(...) method.
    model_type: What kind of model do we want to preprocess our dataset for

  Returns:
    A list of processed dataset chunks.
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
  return parsed_target

def evaluate_model(y, model_hyperparameters, model_type):
  if model_type == "ARIMA":
    model = ARIMA(y, order=model_hyperparameters)
    model_fit = model.fit()
    aic = model_fit.aic
    return aic
  else:
    raise NotImplementedError("This has not been implemented yet, or is an invalid model type.")

def grid_search():
  pass




if __name__ == "__main__":
  list_of_dataframes = intake_dataset('australian_electricity_demand', 5)
  print(f"Australian Electricity Demand Dataset: {intake_dataset('australian_electricity_demand', 5)}")
  preprocessed_dataset_chunks = dataset_preprocess(list_of_dataframes,"ARIMA")
  print(f"Preprocessed chunks: {preprocessed_dataset_chunks}")