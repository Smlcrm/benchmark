from .testing_utilities import value_equality
from ..pipeline.data_loader import DataLoader
from ..pipeline.preprocessor import Preprocessor
import pandas as pd

if __name__ == "__main__":
  print("Preprocessor testing suite!")
  preprocesser_data_loader = DataLoader({"dataset": {
    "path" : "/Users/aryannair/smlcrm-benchmark/benchmarking_pipeline/internal_testing/internal_testing_data/preprocessor_testing",
    "name" : "preprocessor_testing",
    "split_ratio" : [0.8,0.1,0.1]
  }})

  first_chunk = preprocesser_data_loader.load_single_chunk(1)
  #print(f"First chunk {first_chunk}")

  preprocesser = Preprocessor({
    "dataset":{
      "normalize": False,
      "normalization_method": "standard",
      "handle_missing": "median",
      "remove_outliers": False,
      "outlier_threshold": 1
            }
            })
  
  preprocessed_first_chunk = preprocesser.preprocess(first_chunk)
  #print("HUH?",preprocessed_first_chunk)
  
  test_cases = []

  test_cases.append((preprocessed_first_chunk.data.train.targets["y"], pd.Series([1,100000,14,14,14,1789,-5,14])))

  drop_na_outlier_removal_preprocesser = Preprocessor({
    "dataset":{
      "normalize": False,
      "normalization_method": "standard",
      "handle_missing": "drop",
      "remove_outliers": True,
      "outlier_threshold": 1
            }
            })
  secondly_preprocessed_first_chunk = drop_na_outlier_removal_preprocesser.preprocess(first_chunk)
  #print(f"Yibidi {secondly_preprocessed_first_chunk.data}")
  test_cases.append((secondly_preprocessed_first_chunk.data.train.targets["y"], pd.Series([1,14, 1789, -5], index=[0,2,5,6])))

  min_max_preprocessor = Preprocessor({
    "dataset":{
      "normalize": True,
      "normalization_method": "minmax",
      "handle_missing": "interpolate",
      "remove_outliers": False,
      "outlier_threshold": 1
            }
            })
  
  second_chunk = preprocesser_data_loader.load_single_chunk(2)
  #print(f"second chunk {second_chunk}")
  thirdly_preprocessed_second_chunk = min_max_preprocessor.preprocess(second_chunk)
  test_cases.append((thirdly_preprocessed_second_chunk.data.train.targets["y"],pd.Series([0.000000,0.142857,0.285714,0.428571,0.571429,0.714286,0.857143,1.000000])))

  test_case_idx = 0
  for test_case in test_cases:
    assert value_equality(test_case[0], test_case[1]), f"Test case {test_case_idx} failed.\n\n'{test_case[0]}'\n\nand\n\n'{test_case[1]}'\n\nare not equivalent."
    test_case_idx += 1
  
  
  
  print("Got through without any issues!")