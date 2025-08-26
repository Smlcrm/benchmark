"""
Example usage of the benchmarking pipeline components.

This file demonstrates how to use the data loader, preprocessor, models,
and hyperparameter tuning in a clean, organized way.
"""

from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.models.random_forest.random_forest_model import RandomForestModel
from benchmarking_pipeline.models.prophet.prophet_model import ProphetModel
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner


def run_random_forest_example():
    """Example of using Random Forest model with hyperparameter tuning."""
    print("=== Random Forest Model Example ===")
    
    # Configuration
    config = {
        "dataset": {
            "path": "benchmarking_pipeline/datasets/australian_electricity_demand",
            "name": "australian_electricity_demand",
            "split_ratio": [0.8, 0.1, 0.1]
        }
    }
    
    # Load data
    data_loader = DataLoader(config)
    chunks = data_loader.load_several_chunks(2)
    
    # Preprocess data
    preprocessor = Preprocessor({})
    processed_chunks = [preprocessor.preprocess(chunk).data for chunk in chunks]
    
    # Model configuration
    rf_config = {
        "lookback_window": 20,
        "forecast_horizon": 6,
        "model_params": {
            "n_estimators": 10,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1
        }
    }
    
    # Hyperparameter grid
    rf_hyperparameter_grid = {
        "lookback_window": [20],
        "model_params__n_estimators": [10],
    }
    
    # Create model and tuner
    rf_model = RandomForestModel(rf_config)
    rf_hyperparameter_tuner = HyperparameterTuner(rf_model, rf_hyperparameter_grid, "random_forest")
    
    # Run hyperparameter tuning
    print("Starting hyperparameter tuning...")
    validation_score_hyperparameter_tuple = rf_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(processed_chunks)
    print(f"Validation score and hyperparameter: {validation_score_hyperparameter_tuple}")
    
    # Get best hyperparameters
    best_hyperparameters_array = validation_score_hyperparameter_tuple[1]
    best_hyperparameters_dict = {
        "lookback_window": int(best_hyperparameters_array[0]),
        "model_params__n_estimators": int(best_hyperparameters_array[1])
    }
    print(f"Best hyperparameters: {best_hyperparameters_dict}")
    
    # Final evaluation
    final_results = rf_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, processed_chunks)
    print(f"Final evaluation results: {final_results}")


def run_prophet_example():
    """Example of using Prophet model with hyperparameter tuning."""
    print("\n=== Prophet Model Example ===")
    
    # Configuration
    config = {
        "dataset": {
            "path": "benchmarking_pipeline/datasets/australian_electricity_demand",
            "name": "australian_electricity_demand",
            "split_ratio": [0.8, 0.1, 0.1]
        }
    }
    
    # Load data
    data_loader = DataLoader(config)
    chunks = data_loader.load_several_chunks(2)
    
    # Preprocess data
    preprocessor = Preprocessor({})
    processed_chunks = [preprocessor.preprocess(chunk).data for chunk in chunks]
    
    # Prophet configuration
    prophet_config = {
        "model_params": {
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
        }
    }
    
    # Hyperparameter grid
    prophet_hyperparameter_grid = {
        "model_params__seasonality_mode": ["additive", "multiplicative"],
        "model_params__changepoint_prior_scale": [0.01, 0.05, 0.1],
        "model_params__seasonality_prior_scale": [1.0, 10.0],
    }
    
    # Create model and tuner
    prophet_model = ProphetModel(prophet_config)
    prophet_hyperparameter_tuner = HyperparameterTuner(prophet_model, prophet_hyperparameter_grid, "prophet")
    
    # Run hyperparameter tuning
    print("Starting Prophet hyperparameter tuning...")
    validation_score_hyperparameter_tuple = prophet_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(processed_chunks)
    print(f"Validation score and hyperparameter: {validation_score_hyperparameter_tuple}")
    
    # Get best hyperparameters
    best_hyperparameters_dict = {
        "model_params__seasonality_mode": validation_score_hyperparameter_tuple[1][0],
        "model_params__changepoint_prior_scale": validation_score_hyperparameter_tuple[1][1],
        "model_params__seasonality_prior_scale": validation_score_hyperparameter_tuple[1][2],
    }
    print(f"Best hyperparameters: {best_hyperparameters_dict}")
    
    # Final evaluation
    final_results = prophet_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, processed_chunks)
    print(f"Final evaluation results: {final_results}")


if __name__ == "__main__":
    print("Benchmarking Pipeline Examples")
    print("=" * 40)
    
    try:
        run_random_forest_example()
        run_prophet_example()
        print("\n✅ All examples completed successfully!")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have the required datasets and dependencies installed.")
  