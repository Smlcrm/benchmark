import argparse
import yaml
import pickle
import importlib

from benchmarking_pipeline.models.base_model import BaseModel
from benchmarking_pipeline.models.foundation_model import FoundationModel

from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.trainer.foundation_model_tuning import FoundationModelTuner

class PipelineRunner:

    def __init__(self, config, chunk_path, model_folder_name, model_file_name, model_class_name):
        self.config = config
        self.chunk_path = chunk_path
        self.model_folder_name = model_folder_name
        self.model_file_name = model_file_name
        self.model_class_name = model_class_name

    def run(self):
        module_path = f"benchmarking_pipeline.models.{self.model_folder_name}.{self.model_file_name}"  # e.g. models.LSTMModel
        module = importlib.import_module(module_path)
        model_class = getattr(module, self.model_class_name)
        print("Yip Yip Yooray!")
        with open(self.chunk_path, 'rb') as f:
          all_dataset_chunks = pickle.load(f)
        
        if issubclass(model_class, BaseModel):
            print(f"{self.model_folder_name} is a Base Model!")
            hyper_grid = config['model']['parameters'][self.model_folder_name]
            print(f"{self.model_folder_name} hyper grid: {hyper_grid}")

            model_params = {k: v[0] if isinstance(v, list) else v for k, v in hyper_grid.items()}
            print(f"{self.model_folder_name} initial model_params: {model_params}")

            base_model = model_class(model_params)

            #hyper_grid = {k: v for k, v in model_params.items() if isinstance(v, list)}
            #print(f"{self.model_folder_name} hyper grid: {hyper_grid}")

            model_hyperparameter_tuner = HyperparameterTuner(base_model, hyper_grid, False)
            validation_score_hyperparameter_tuple = model_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
            print(f"{self.model_folder_name} validation score hyperparameter tuple: {validation_score_hyperparameter_tuple}")
            best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
            print(f"{self.model_folder_name} best hyperparameters dict: {best_hyperparameters_dict}")
            results = model_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)

            print(f"{self.model_folder_name} results: {results}")
            print(f"{self.model_folder_name} WORKS!".capitalize())
        elif issubclass(model_class, FoundationModel):
            print(f"{self.model_folder_name} is a Foundation Model!")
            
            hyper_grid = config['model']['parameters'][self.model_folder_name]
            model_params = {k: v[0] if isinstance(v, list) else v for k, v in hyper_grid.items()}

            foundation_model = model_class(model_params)

            #hyper_grid = {k: v for k, v in model_params.items() if isinstance(v, list)}

            model_hyperparameter_tuner = FoundationModelTuner(foundation_model, hyper_grid, False)
            validation_score_hyperparameter_tuple = model_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
            best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
            print(f"{self.model_folder_name} best hyperparameters dict: {best_hyperparameters_dict}")
            results = model_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)

            print(f"{self.model_folder_name} results: {results}")
            print(f"{self.model_folder_name} WORKS!".capitalize())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal file used to run a full pipeline on a model.")
    parser.add_argument('--config', type=str, help='Path to the config YAML file')
    parser.add_argument('--chunk_path', type=str, help='Path to the temporary pickle file we made to store data.')
    parser.add_argument('--model_folder_name', type=str, help='Name of the model folder we are referencing.')
    parser.add_argument('--model_file_name', type=str, help='Name of the model file we want to use the model class of.')
    parser.add_argument('--model_class_name', type=str, help='Name of our desired model class.')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pipeline_runner = PipelineRunner(config=config, 
                                     chunk_path=args.chunk_path, 
                                     model_folder_name=args.model_folder_name,
                                     model_file_name=args.model_file_name,
                                     model_class_name=args.model_class_name)
    print("[DEBUG] Config",args.config)
    print("[DEBUG] Chunk Path",args.chunk_path)
    print("[DEBUG] Model Folder Name",args.model_folder_name)
    print("[DEBUG] Model File Name",args.model_file_name)
    print("[DEBUG] Model Class Name",args.model_class_name)
    pipeline_runner.run() 