from .testing_equality import value_equality
from ..pipeline.data_loader import DataLoader
from ..pipeline.preprocessor import Preprocessor

if __name__ == "__main__":
  print("Preprocessor testing suite!")
  preprocesser_data_loader = DataLoader({"dataset": {
    "path" : "/Users/alifabdullah/Collaboration/benchmark/benchmarking_pipeline/internal_testing/internal_testing_data/preprocessor_testing",
    "name" : "preprocessor_testing",
    "split_ratio" : [0.8,0.1,0.1]
  }})

  first_chunk = preprocesser_data_loader.load_single_chunk(1)
  print(f"First chunk {first_chunk}")

  preprocesser = Preprocessor({
      "normalize": True,
            "normalization_method": "standard",
            "handle_missing": "interpolate",
            "remove_outliers": False,
            "outlier_threshold": 3
            })
  
  print(f"Preprocessed first chunk: {preprocesser.preprocess(first_chunk)}")
  print("HUH?",preprocesser.preprocessing_config)
  
  test_cases = []