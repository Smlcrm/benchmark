from ..pipeline.data_loader import DataLoader
from .testing_utilities import value_equality
import pandas as pd

if __name__ == "__main__":
  print("Data loader testing suite!")
  data_loader_to_be_tested = DataLoader({"dataset": {
    "path" : "/Users/aryannair/smlcrm-benchmark/benchmarking_pipeline/internal_testing/internal_testing_data/data_loader_testing",
    "name" : "data_loader_testing",
    "split_ratio" : [0.8,0.1,0.1]
  }})

  test_cases = []

  # Test Cases: Assert target element equality
  test_cases.append((data_loader_to_be_tested.load_single_chunk(1).train.features["y"], pd.Series([1, 2, 3000, 7001, 1, 2, 3000, 7001, 1, 2, 3000, 7001, 1, 2, 3000, 7001])))
  test_cases.append((data_loader_to_be_tested.load_single_chunk(3).train.features["y"],pd.Series([[8, 15], [7, 4], [8, 15], [7, 4], [8, 15], [7, 4], [8, 15], [7, 4],[8, 15]])))

  two_chunks = data_loader_to_be_tested.load_several_chunks(2)
  chunk_one, chunk_two = two_chunks

  test_cases.append((chunk_one.validation.features["y"], pd.Series([1,2],index=[16,17])))
  test_cases.append((chunk_two.test.features["y"], pd.Series([9,4,8,11,15],index=[37,38,39,40,41])))
  

  test_case_idx = 0
  for test_case in test_cases:
    assert value_equality(test_case[0], test_case[1]), f"Test case {test_case_idx} failed. '{test_case[0]}' and '{test_case[1]}' are not equivalent."
    test_case_idx += 1
  
  
  
  print("Got through without any issues!")
