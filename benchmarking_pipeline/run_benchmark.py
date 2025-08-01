"""
Main script for orchestrating the end-to-end benchmarking pipeline.
"""

from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.pipeline.feature_extraction import FeatureExtractor
from benchmarking_pipeline.pipeline.trainer import Trainer
from benchmarking_pipeline.pipeline.evaluator import Evaluator
from benchmarking_pipeline.pipeline.logger import Logger
from torch.utils.tensorboard import SummaryWriter
import time
from benchmarking_pipeline.models.arima_model import ARIMAModel
from benchmarking_pipeline.models.theta_model import ThetaModel
from benchmarking_pipeline.models.deepAR_model import DeepARModel
from benchmarking_pipeline.models.xgboost_model import XGBoostModel
from benchmarking_pipeline.models.random_forest_model import RandomForestModel
from benchmarking_pipeline.models.prophet_model import ProphetModel
from benchmarking_pipeline.models.lstm_model import LSTMModel
from benchmarking_pipeline.models.croston_classic_model import CrostonClassicModel
from benchmarking_pipeline.trainer.hyperparameter_tuning import HyperparameterTuner
from benchmarking_pipeline.trainer.bayesian_hyperparameter_tuner import BayesianHyperparameterTuner
from benchmarking_pipeline.trainer.successive_halving_tuner import SuccessiveHalvingTuner
from benchmarking_pipeline.trainer.population_based_tuner import PopulationBasedTuner
import numpy as np
import pandas as pd
import re
import os
import json
import datetime
import matplotlib.pyplot as plt
import yaml
from benchmarking_pipeline.models.SVR_model import SVRModel
from benchmarking_pipeline.models.seasonal_naive_model import SeasonalNaiveModel
from benchmarking_pipeline.models.exponential_smoothing_model import ExponentialSmoothingModel
import argparse

class BenchmarkRunner:
    def __init__(self, config, config_path=None):
        """
        Initialize benchmark runner with configuration.
        Args:
            config: Configuration dictionary for the pipeline
            config_path: Path to the config file used
        """
        self.config = config
        self.config_path = config_path
        self.logger = Logger(config)
        
    def run(self):
        """Execute the end-to-end benchmarking pipeline."""
        # Determine config file name for logging
        config_file_name = os.path.splitext(os.path.basename(self.config_path))[0] if self.config_path else 'unknown_config'
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=f"runs/benchmark_runner_{config_file_name}_{timestamp}")
        # Load dataset config
        dataset_cfg = self.config['dataset']
        dataset_path = dataset_cfg['path']
        dataset_name = dataset_cfg['name']
        split_ratio = dataset_cfg.get('split_ratio', [0.8, 0.1, 0.1])
        # Use 'chunks' from config, default to 1 if not present
        num_chunks = dataset_cfg.get('chunks', 1)
        data_loader = DataLoader({"dataset": {
            "path": dataset_path,
            "name": dataset_name,
            "split_ratio": split_ratio
        }})
        all_dataset_chunks = data_loader.load_several_chunks(num_chunks)
        preprocessor = Preprocessor({"dataset": {"normalize": dataset_cfg.get('normalize', False)}})
        all_dataset_chunks = [preprocessor.preprocess(chunk).data for chunk in all_dataset_chunks]
        print(f"[DEBUG] Number of chunks: {len(all_dataset_chunks)}")
        if len(all_dataset_chunks) > 0:
            print(f"[DEBUG] First chunk train targets shape: {all_dataset_chunks[0].train.targets.shape}")
            if all_dataset_chunks[0].train.features is not None:
                print(f"[DEBUG] First chunk train features shape: {all_dataset_chunks[0].train.features.shape}")
            else:
                print(f"[DEBUG] First chunk has no exogenous features")
        # Load config
        config_path = self.config_path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_names = config['model']['name']
        for model_name in model_names:
            func_name = f"run_{model_name.lower()}"
            if func_name in globals():
                globals()[func_name](all_dataset_chunks, writer, self.config, self.config_path)
        writer.close()
        print(f"\nTensorBoard logs saved in 'runs/benchmark_runner_{config_file_name}_{timestamp}/'. To view, run: tensorboard --logdir runs/benchmark_runner\n")
        self.logger.log_metrics({"status": "Pipeline completed"}, step=0)

def _extract_number_before_capital(freq_str):
    match = re.match(r'(\d+)?[A-Z]', freq_str)
    if match:
        return int(match.group(1)) if match.group(1) else 1
    else:
        raise ValueError(f"Invalid frequency string: {freq_str}")

def log_preds_vs_true(writer, model_name, y_true, y_pred, step):
    import numpy as np
    import matplotlib.pyplot as plt
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Define a color palette for multiple targets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Handle both univariate and multivariate cases
    if y_true.ndim == 1:
        # Univariate case
        ax.plot(y_true, color=colors[0], label='True', linewidth=2)
        ax.plot(y_pred, color=colors[1], label='Predicted', linewidth=2, linestyle='--')
    else:
        # Multivariate case - plot each target with different colors
        num_targets = y_true.shape[1]
        
        for i in range(num_targets):
            color_true = colors[i % len(colors)]
            
            # Plot true values for all targets
            ax.plot(y_true[:, i], color=color_true, label=f'True Target {i}', 
                   linewidth=2, alpha=0.8)
            
            # Plot predictions if we have them for this target
            if y_pred.ndim == 1 and i == 0:
                # Single prediction for first target only
                ax.plot(y_pred, color=colors[(i + 1) % len(colors)], 
                       label=f'Predicted Target {i}', linewidth=2, linestyle='--', alpha=0.8)
            elif y_pred.ndim == 2 and i < y_pred.shape[1]:
                # Check if we have valid predictions for this target (not all NaN)
                pred_values = y_pred[:, i]
                if not np.all(np.isnan(pred_values)):
                    color_pred = colors[(i + 1) % len(colors)]
                    ax.plot(pred_values, color=color_pred, label=f'Predicted Target {i}', 
                           linewidth=2, linestyle='--', alpha=0.8)
    
    ax.set_title(f'{model_name} Predictions vs True Values', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    writer.add_figure(f'{model_name}/pred_vs_true', fig, global_step=step)
    plt.close(fig)

def _create_hyperparameter_tuner(model_class, model_params, hp_tuning_method="grid"):
    """
    Create appropriate hyperparameter tuner based on config format and method.
    
    Args:
        model_class: The model class instance
        model_params: Model parameters from config
        hp_tuning_method: "grid", "bayesian", "successive_halving", or "pbt"
    
    Returns:
        Appropriate hyperparameter tuner instance
    """
    # Determine if we have range-based or list-based parameters
    has_ranges = any(isinstance(v, dict) and 'min' in v and 'max' in v 
                    for v in model_params.values())
    
    # Convert parameters to list format for all tuners
    hyperparameter_ranges = {}
    for param_name, param_value in model_params.items():
        if isinstance(param_value, dict) and 'min' in param_value and 'max' in param_value:
            # Range-based parameter
            if param_value['type'] == 'int':
                # Create list of integers in range
                min_val, max_val = param_value['min'], param_value['max']
                # Ensure Python integers
                hyperparameter_ranges[param_name] = [int(x) for x in range(min_val, max_val + 1)]
            elif param_value['type'] == 'float':
                # For float ranges, create discrete choices
                min_val, max_val = param_value['min'], param_value['max']
                n_choices = min(15, int((max_val - min_val) * 10 + 1))  # Create reasonable number of choices
                # Convert numpy floats to Python floats
                hyperparameter_ranges[param_name] = [float(x) for x in np.linspace(min_val, max_val, n_choices)]
        elif isinstance(param_value, list):
            # List-based parameter (categorical or discrete)
            # Convert to proper Python types
            converted_list = []
            for item in param_value:
                if isinstance(item, (np.integer, np.floating)):
                    if isinstance(item, np.integer):
                        converted_list.append(int(item))
                    else:
                        converted_list.append(float(item))
                elif isinstance(item, np.str_):
                    converted_list.append(str(item))
                else:
                    converted_list.append(item)
            hyperparameter_ranges[param_name] = converted_list
        else:
            # Single value parameter
            hyperparameter_ranges[param_name] = [param_value]
    
    # Get search parameters from config
    search_iterations = model_params.get('search_iterations', 15)
    
    if hp_tuning_method == "successive_halving":
        print(f"Using Successive Halving hyperparameter tuning for {model_class.__class__.__name__}")
        return SuccessiveHalvingTuner(
            model_class=model_class,
            hyperparameter_ranges=hyperparameter_ranges,
            use_exog=False,
            n_configurations=50,  # Start with many configurations
            min_resources=1,      # Start with minimal resources
            max_resources=10,     # End with full resources
            reduction_factor=3    # Reduce by factor of 3 each round
        )
    
    elif hp_tuning_method == "pbt" and "lstm" in model_class.__class__.__name__.lower():
        print(f"Using Population-Based Training for {model_class.__class__.__name__}")
        return PopulationBasedTuner(
            model_class=model_class,
            hyperparameter_ranges=hyperparameter_ranges,
            use_exog=False,
            population_size=8,
            num_generations=10
        )
    
    elif hp_tuning_method == "bayesian" or has_ranges:
        print(f"Using Bayesian hyperparameter tuning for {model_class.__class__.__name__}")
        return BayesianHyperparameterTuner(
            model_class=model_class,
            hyperparameter_ranges=hyperparameter_ranges,
            use_exog=False,
            n_iterations=search_iterations
        )
    
    else:
        # Use grid search for list-based parameters
        print(f"Using grid search hyperparameter tuning for {model_class.__class__.__name__}")
        return HyperparameterTuner(model_class, model_params, False)

def run_arima(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    arima_params = config['model']['parameters']['ARIMA']
    
    # Auto-detect target column for multivariate datasets
    if len(all_dataset_chunks) > 0:
        available_targets = list(all_dataset_chunks[0].train.targets.columns)
        if 'target_0' in available_targets:
            # Use first target for multivariate datasets
            arima_params['target_col'] = 'target_0'
        elif 'y' not in available_targets and len(available_targets) > 0:
            # Use first available target if 'y' doesn't exist
            arima_params['target_col'] = available_targets[0]
        else:
            # Default to 'y' for univariate datasets
            arima_params['target_col'] = 'y'
    # Create model_params with default values for initial model creation
    model_params = {}
    for k, v in arima_params.items():
        if isinstance(v, list):
            value = v[0]
        elif isinstance(v, dict):
            # For range-based parameters, use the middle value
            if 'min' in v and 'max' in v:
                if v['type'] == 'int':
                    value = int((v['min'] + v['max']) // 2)
                elif v['type'] == 'float':
                    value = float((v['min'] + v['max']) / 2)
            else:
                value = v
        else:
            value = v
            
        if k in ['p', 'd', 'q']:
            model_params[k] = int(value)
            print(f"[DEBUG] ARIMA param {k}: value={value}, type={type(value)} -> casted type={type(model_params[k])}")
        else:
            model_params[k] = value
    print(f"[DEBUG] ARIMA model_params: {model_params}")
    arima_model = ARIMAModel(model_params)
    
    # Create appropriate tuner
    hp_tuning_method = config.get('model', {}).get('search_method', 'grid')
    tuner = _create_hyperparameter_tuner(arima_model, arima_params, hp_tuning_method)
    
    if isinstance(tuner, BayesianHyperparameterTuner):
        validation_score_hyperparameter_tuple = tuner.bayesian_optimization_several_time_series(all_dataset_chunks)
    elif isinstance(tuner, SuccessiveHalvingTuner):
        validation_score_hyperparameter_tuple = tuner.successive_halving_search(all_dataset_chunks)
    elif isinstance(tuner, PopulationBasedTuner):
        validation_score_hyperparameter_tuple = tuner.population_based_search(all_dataset_chunks)
    else:
        validation_score_hyperparameter_tuple = tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(tuner.param_names if hasattr(tuner, 'param_names') else arima_params.keys())}
    # Cast p, d, q to int if present
    for k in ['p', 'd', 'q']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    results = tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation ARIMA {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('ARIMA/' + metric, value, 1)
        y_true, y_pred = arima_model.get_last_eval_true_pred()
        # Get all targets from the dataset for comprehensive plotting
        all_targets_true = all_dataset_chunks[0].test.targets.values
        # For multivariate datasets, we need to handle the case where predictions are only for one target
        if all_targets_true.ndim == 2 and y_pred.ndim == 1:
            # Create a predictions array that matches the shape of all targets
            # Fill with NaN for targets we don't have predictions for
            all_targets_pred = np.full_like(all_targets_true, np.nan)
            all_targets_pred[:, 0] = y_pred  # Put predictions in first column
        else:
            all_targets_pred = y_pred
        log_preds_vs_true(writer, 'ARIMA', all_targets_true, all_targets_pred, 1)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/ARIMA_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("ARIMA WORKS!")

def run_theta(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    theta_params = config['model']['parameters']['Theta']
    
    # Auto-detect target column for multivariate datasets
    if len(all_dataset_chunks) > 0:
        available_targets = list(all_dataset_chunks[0].train.targets.columns)
        if 'target_0' in available_targets:
            # Use first target for multivariate datasets
            theta_params['target_col'] = 'target_0'
        elif 'y' not in available_targets and len(available_targets) > 0:
            # Use first available target if 'y' doesn't exist
            theta_params['target_col'] = available_targets[0]
        else:
            # Default to 'y' for univariate datasets
            theta_params['target_col'] = 'y'
    # Create model_params with default values for initial model creation
    model_params = {}
    for k, v in theta_params.items():
        if isinstance(v, list):
            model_params[k] = v[0]
        elif isinstance(v, dict):
            # For range-based parameters, use the middle value
            if 'min' in v and 'max' in v:
                if v['type'] == 'int':
                    model_params[k] = int((v['min'] + v['max']) // 2)
                elif v['type'] == 'float':
                    model_params[k] = float((v['min'] + v['max']) / 2)
            else:
                model_params[k] = v
        else:
            model_params[k] = v
    # Force additive seasonality for Theta
    model_params['seasonality_mode'] = 'additive'
    theta_model = ThetaModel(model_params)
    
    # Create appropriate tuner
    hp_tuning_method = config.get('model', {}).get('search_method', 'grid')
    tuner = _create_hyperparameter_tuner(theta_model, theta_params, hp_tuning_method)
    
    if isinstance(tuner, BayesianHyperparameterTuner):
        validation_score_hyperparameter_tuple = tuner.bayesian_optimization_several_time_series(all_dataset_chunks)
    elif isinstance(tuner, SuccessiveHalvingTuner):
        validation_score_hyperparameter_tuple = tuner.successive_halving_search(all_dataset_chunks)
    elif isinstance(tuner, PopulationBasedTuner):
        validation_score_hyperparameter_tuple = tuner.population_based_search(all_dataset_chunks)
    else:
        validation_score_hyperparameter_tuple = tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(tuner.param_names if hasattr(tuner, 'param_names') else theta_params.keys())}
    for k in ['sp', 'forecast_horizon']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    # Also force additive seasonality in best hyperparameters
    best_hyperparameters_dict['seasonality_mode'] = 'additive'
    results = tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation Theta {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('Theta/' + metric, value, 7)
        y_true, y_pred = theta_model.get_last_eval_true_pred()
        # Get all targets from the dataset for comprehensive plotting
        all_targets_true = all_dataset_chunks[0].test.targets.values
        # For multivariate datasets, we need to handle the case where predictions are only for one target
        if all_targets_true.ndim == 2 and y_pred.ndim == 1:
            # Create a predictions array that matches the shape of all targets
            # Fill with NaN for targets we don't have predictions for
            all_targets_pred = np.full_like(all_targets_true, np.nan)
            all_targets_pred[:, 0] = y_pred  # Put predictions in first column
        else:
            all_targets_pred = y_pred
        log_preds_vs_true(writer, 'Theta', all_targets_true, all_targets_pred, 7)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f"results/Theta_{dataset_name}_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
    print("Theta WORKS!")

def run_deep_ar(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    deep_ar_params = config['model']['parameters']['DeepAR']
    
    # Auto-detect target column for multivariate datasets
    if len(all_dataset_chunks) > 0:
        available_targets = list(all_dataset_chunks[0].train.targets.columns)
        if 'target_0' in available_targets:
            # Use first target for multivariate datasets
            deep_ar_params['target_col'] = 'target_0'
        elif 'y' not in available_targets and len(available_targets) > 0:
            # Use first available target if 'y' doesn't exist
            deep_ar_params['target_col'] = available_targets[0]
        else:
            # Default to 'y' for univariate datasets
            deep_ar_params['target_col'] = 'y'
    # Create model_params with default values for initial model creation
    model_params = {}
    for k, v in deep_ar_params.items():
        if isinstance(v, list):
            model_params[k] = v[0]
        elif isinstance(v, dict):
            # For range-based parameters, use the middle value
            if 'min' in v and 'max' in v:
                if v['type'] == 'int':
                    model_params[k] = int((v['min'] + v['max']) // 2)
                elif v['type'] == 'float':
                    model_params[k] = float((v['min'] + v['max']) / 2)
            else:
                model_params[k] = v
        else:
            model_params[k] = v
    deep_ar_model = DeepARModel(model_params)
    
    # Create appropriate tuner
    hp_tuning_method = config.get('model', {}).get('search_method', 'grid')
    tuner = _create_hyperparameter_tuner(deep_ar_model, deep_ar_params, hp_tuning_method)
    
    if isinstance(tuner, BayesianHyperparameterTuner):
        validation_score_hyperparameter_tuple = tuner.bayesian_optimization_several_time_series(all_dataset_chunks)
    elif isinstance(tuner, SuccessiveHalvingTuner):
        validation_score_hyperparameter_tuple = tuner.successive_halving_search(all_dataset_chunks)
    elif isinstance(tuner, PopulationBasedTuner):
        validation_score_hyperparameter_tuple = tuner.population_based_search(all_dataset_chunks)
    else:
        validation_score_hyperparameter_tuple = tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(tuner.param_names if hasattr(tuner, 'param_names') else deep_ar_params.keys())}
    for k in ['hidden_size', 'rnn_layers', 'batch_size', 'epochs', 'forecast_horizon', 'num_workers']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    results = tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation DeepAR {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('DeepAR/' + metric, value)
        y_true, y_pred = deep_ar_model.get_last_eval_true_pred()
        # Get all targets from the dataset for comprehensive plotting
        all_targets_true = all_dataset_chunks[0].test.targets.values
        # For multivariate datasets, we need to handle the case where predictions are only for one target
        if all_targets_true.ndim == 2 and y_pred.ndim == 1:
            # Create a predictions array that matches the shape of all targets
            # Fill with NaN for targets we don't have predictions for
            all_targets_pred = np.full_like(all_targets_true, np.nan)
            all_targets_pred[:, 0] = y_pred  # Put predictions in first column
        else:
            all_targets_pred = y_pred
        log_preds_vs_true(writer, 'DeepAR', all_targets_true, all_targets_pred, 8)
    os.makedirs('results', exist_ok=True)
    with open(f'results/DeepAR_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Deep AR WORKS!")

def run_xgboost(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    xgb_params = config['model']['parameters']['XGBoost']
    
    # Auto-detect target column for multivariate datasets
    if len(all_dataset_chunks) > 0:
        available_targets = list(all_dataset_chunks[0].train.targets.columns)
        if 'target_0' in available_targets:
            # Use first target for multivariate datasets
            xgb_params['target_col'] = 'target_0'
        elif 'y' not in available_targets and len(available_targets) > 0:
            # Use first available target if 'y' doesn't exist
            xgb_params['target_col'] = available_targets[0]
        else:
            # Default to 'y' for univariate datasets
            xgb_params['target_col'] = 'y'
    # Create model_params with default values for initial model creation
    model_params = {}
    for k, v in xgb_params.items():
        if isinstance(v, list):
            model_params[k] = v[0]
        elif isinstance(v, dict):
            # For range-based parameters, use the middle value
            if 'min' in v and 'max' in v:
                if v['type'] == 'int':
                    model_params[k] = int((v['min'] + v['max']) // 2)
                elif v['type'] == 'float':
                    model_params[k] = float((v['min'] + v['max']) / 2)
            else:
                model_params[k] = v
        else:
            model_params[k] = v
    xgb_model = XGBoostModel(model_params)
    
    # Create appropriate tuner
    hp_tuning_method = config.get('model', {}).get('search_method', 'grid')
    tuner = _create_hyperparameter_tuner(xgb_model, xgb_params, hp_tuning_method)
    
    if isinstance(tuner, BayesianHyperparameterTuner):
        validation_score_hyperparameter_tuple = tuner.bayesian_optimization_several_time_series(all_dataset_chunks)
    elif isinstance(tuner, SuccessiveHalvingTuner):
        validation_score_hyperparameter_tuple = tuner.successive_halving_search(all_dataset_chunks)
    elif isinstance(tuner, PopulationBasedTuner):
        validation_score_hyperparameter_tuple = tuner.population_based_search(all_dataset_chunks)
    else:
        validation_score_hyperparameter_tuple = tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(tuner.param_names if hasattr(tuner, 'param_names') else xgb_params.keys())}
    for k in ['lookback_window', 'forecast_horizon', 'n_estimators', 'max_depth', 'random_state', 'n_jobs']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    results = tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation XGBoost {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('XGBoost/' + metric, value, 2)
        y_true, y_pred = xgb_model.get_last_eval_true_pred()
        # Get all targets from the dataset for comprehensive plotting
        all_targets_true = all_dataset_chunks[0].test.targets.values
        # For multivariate datasets, we need to handle the case where predictions are only for one target
        if all_targets_true.ndim == 2 and y_pred.ndim == 1:
            # Create a predictions array that matches the shape of all targets
            # Fill with NaN for targets we don't have predictions for
            all_targets_pred = np.full_like(all_targets_true, np.nan)
            all_targets_pred[:, 0] = y_pred  # Put predictions in first column
        else:
            all_targets_pred = y_pred
        log_preds_vs_true(writer, 'XGBoost', all_targets_true, all_targets_pred, 2)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/XGBoost_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("XGBoost WORKS!")

def run_random_forest(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    rf_params = config['model']['parameters']['RandomForest']
    
    # Auto-detect target column for multivariate datasets
    if len(all_dataset_chunks) > 0:
        available_targets = list(all_dataset_chunks[0].train.targets.columns)
        if 'target_0' in available_targets:
            # Use first target for multivariate datasets
            rf_params['target_col'] = 'target_0'
        elif 'y' not in available_targets and len(available_targets) > 0:
            # Use first available target if 'y' doesn't exist
            rf_params['target_col'] = available_targets[0]
    # Create model_params with default values for initial model creation
    model_params = {}
    for k, v in rf_params.items():
        if isinstance(v, list):
            model_params[k] = v[0]
        elif isinstance(v, dict):
            # For range-based parameters, use the middle value
            if 'min' in v and 'max' in v:
                if v['type'] == 'int':
                    model_params[k] = int((v['min'] + v['max']) // 2)
                elif v['type'] == 'float':
                    model_params[k] = float((v['min'] + v['max']) / 2)
            else:
                model_params[k] = v
        else:
            model_params[k] = v
    rf_model = RandomForestModel(model_params)
    
    # Create appropriate tuner
    hp_tuning_method = config.get('model', {}).get('search_method', 'grid')
    tuner = _create_hyperparameter_tuner(rf_model, rf_params, hp_tuning_method)
    
    if isinstance(tuner, BayesianHyperparameterTuner):
        validation_score_hyperparameter_tuple = tuner.bayesian_optimization_several_time_series(all_dataset_chunks)
    elif isinstance(tuner, SuccessiveHalvingTuner):
        validation_score_hyperparameter_tuple = tuner.successive_halving_search(all_dataset_chunks)
    elif isinstance(tuner, PopulationBasedTuner):
        validation_score_hyperparameter_tuple = tuner.population_based_search(all_dataset_chunks)
    else:
        validation_score_hyperparameter_tuple = tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(tuner.param_names if hasattr(tuner, 'param_names') else rf_params.keys())}
    for k in ['lookback_window', 'forecast_horizon', 'n_estimators', 'max_depth', 'random_state', 'n_jobs']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    results = tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation Random Forest {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('RandomForest/' + metric, value, 3)
        # Use rolling_predict for full-length predictions for plotting
        chunk = all_dataset_chunks[0]
        lookback_window = rf_model.lookback_window
        y_context = chunk.train.targets.values.flatten()[-lookback_window:]
        y_target = chunk.test.targets.values.flatten()
        y_context_timestamps = chunk.train.timestamps[-lookback_window:]
        y_target_timestamps = chunk.test.timestamps
        full_preds = rf_model.rolling_predict(
            y_context=y_context,
            y_target=y_target,
            y_context_timestamps=y_context_timestamps,
            y_target_timestamps=y_target_timestamps
        )
        log_preds_vs_true(writer, 'RandomForest', y_target, full_preds, 3)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/RandomForest_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("Random Forest WORKS!")

def run_prophet(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    prophet_params = config['model']['parameters']['Prophet']
    
    # Auto-detect target column for multivariate datasets
    if len(all_dataset_chunks) > 0:
        available_targets = list(all_dataset_chunks[0].train.targets.columns)
        if 'target_0' in available_targets:
            # Use first target for multivariate datasets
            prophet_params['target_col'] = 'target_0'
        elif 'y' not in available_targets and len(available_targets) > 0:
            # Use first available target if 'y' doesn't exist
            prophet_params['target_col'] = available_targets[0]
    # Cast parameters to correct types
    def cast_param(key, value):
        # Handle range-based parameters (dictionaries)
        if isinstance(value, dict):
            # For range-based parameters, use the first value or a default
            if 'min' in value and 'max' in value:
                if value['type'] == 'int':
                    return int((value['min'] + value['max']) // 2)  # Use middle value
                elif value['type'] == 'float':
                    return float((value['min'] + value['max']) / 2)  # Use middle value
            return value  # Return as-is for other dict types
        
        # Handle regular parameters
        if key in ['seasonality_mode']:
            return str(value)
        elif key in ['changepoint_prior_scale', 'seasonality_prior_scale']:
            return float(value)
        elif key in ['yearly_seasonality', 'weekly_seasonality', 'daily_seasonality']:
            if isinstance(value, str):
                return value.lower() == 'true'
            return bool(value)
        return value
    
    # Create model_params with default values for initial model creation
    model_params = {}
    for k, v in prophet_params.items():
        if isinstance(v, list):
            model_params[k] = cast_param(k, v[0])
        elif isinstance(v, dict):
            model_params[k] = cast_param(k, v)
        else:
            model_params[k] = cast_param(k, v)
    prophet_model = ProphetModel(model_params)
    
    # Create appropriate tuner
    hp_tuning_method = config.get('model', {}).get('search_method', 'grid')
    tuner = _create_hyperparameter_tuner(prophet_model, prophet_params, hp_tuning_method)
    
    if isinstance(tuner, BayesianHyperparameterTuner):
        validation_score_hyperparameter_tuple = tuner.bayesian_optimization_several_time_series(all_dataset_chunks)
    elif isinstance(tuner, SuccessiveHalvingTuner):
        validation_score_hyperparameter_tuple = tuner.successive_halving_search(all_dataset_chunks)
    elif isinstance(tuner, PopulationBasedTuner):
        validation_score_hyperparameter_tuple = tuner.population_based_search(all_dataset_chunks)
    else:
        validation_score_hyperparameter_tuple = tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    
    best_hyperparameters_dict = {k: cast_param(k, validation_score_hyperparameter_tuple[1][i]) for i, k in enumerate(tuner.param_names if hasattr(tuner, 'param_names') else prophet_params.keys())}
    results = tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation Prophet {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('Prophet/' + metric, value, 4)
        y_true, y_pred = prophet_model.get_last_eval_true_pred()
        # Get all targets from the dataset for comprehensive plotting
        all_targets_true = all_dataset_chunks[0].test.targets.values
        # For multivariate datasets, we need to handle the case where predictions are only for one target
        if all_targets_true.ndim == 2 and y_pred.ndim == 1:
            # Create a predictions array that matches the shape of all targets
            # Fill with NaN for targets we don't have predictions for
            all_targets_pred = np.full_like(all_targets_true, np.nan)
            all_targets_pred[:, 0] = y_pred  # Put predictions in first column
        else:
            all_targets_pred = y_pred
        log_preds_vs_true(writer, 'Prophet', all_targets_true, all_targets_pred, 4)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/Prophet_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print(f"Test Evaluation Prophet {dataset_name}: {tuner.final_evaluation({'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0}, all_dataset_chunks)}")
    print("Prophet WORKS!")

def run_croston_classic(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    croston_params = config['model']['parameters']['CrostonClassic']
    
    # Auto-detect target column for multivariate datasets
    if len(all_dataset_chunks) > 0:
        available_targets = list(all_dataset_chunks[0].train.targets.columns)
        if 'target_0' in available_targets:
            # Use first target for multivariate datasets
            croston_params['target_col'] = 'target_0'
        elif 'y' not in available_targets and len(available_targets) > 0:
            # Use first available target if 'y' doesn't exist
            croston_params['target_col'] = available_targets[0]
    # Create model_params with default values for initial model creation
    model_params = {}
    for k, v in croston_params.items():
        if isinstance(v, list):
            model_params[k] = v[0]
        elif isinstance(v, dict):
            # For range-based parameters, use the middle value
            if 'min' in v and 'max' in v:
                if v['type'] == 'int':
                    model_params[k] = int((v['min'] + v['max']) // 2)
                elif v['type'] == 'float':
                    model_params[k] = float((v['min'] + v['max']) / 2)
            else:
                model_params[k] = v
        else:
            model_params[k] = v
    croston_model = CrostonClassicModel(model_params)
    
    # Create appropriate tuner
    hp_tuning_method = config.get('model', {}).get('search_method', 'grid')
    tuner = _create_hyperparameter_tuner(croston_model, croston_params, hp_tuning_method)
    
    if isinstance(tuner, BayesianHyperparameterTuner):
        validation_score_hyperparameter_tuple = tuner.bayesian_optimization_several_time_series(all_dataset_chunks)
    elif isinstance(tuner, SuccessiveHalvingTuner):
        validation_score_hyperparameter_tuple = tuner.successive_halving_search(all_dataset_chunks)
    elif isinstance(tuner, PopulationBasedTuner):
        validation_score_hyperparameter_tuple = tuner.population_based_search(all_dataset_chunks)
    else:
        validation_score_hyperparameter_tuple = tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    
    results = tuner.final_evaluation({"alpha": 0.1}, all_dataset_chunks)
    print(f"Final Evaluation CrostonClassic {dataset_name}: {results}")
    if writer is not None:
        y_true, y_pred = croston_model.get_last_eval_true_pred()
        print(f"Croston y_true: {y_true}")
        print(f"Croston y_pred: {y_pred}")
        if y_true is not None and y_pred is not None:
            print(f"Croston y_true shape: {y_true.shape}")
            print(f"Croston y_pred shape: {y_pred.shape}")
        # Get all targets from the dataset for comprehensive plotting
        all_targets_true = all_dataset_chunks[0].test.targets.values
        # For multivariate datasets, we need to handle the case where predictions are only for one target
        if all_targets_true.ndim == 2 and y_pred.ndim == 1:
            # Create a predictions array that matches the shape of all targets
            # Fill with NaN for targets we don't have predictions for
            all_targets_pred = np.full_like(all_targets_true, np.nan)
            all_targets_pred[:, 0] = y_pred  # Put predictions in first column
        else:
            all_targets_pred = y_pred
        log_preds_vs_true(writer, 'CrostonClassic', all_targets_true, all_targets_pred, 6)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/CrostonClassic_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)

def run_lstm(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    lstm_params = config['model']['parameters']['LSTM']
    
    # Check if we have multivariate data
    is_multivariate = False
    if len(all_dataset_chunks) > 0:
        available_targets = list(all_dataset_chunks[0].train.targets.columns)
        print(f"[DEBUG] Available targets: {available_targets}")
        is_multivariate = len(available_targets) > 1 or (len(available_targets) == 1 and available_targets[0] != 'y')
        print(f"[DEBUG] Is multivariate: {is_multivariate}")
        
        if is_multivariate:
            print(f"[DEBUG] Using MultivariateLSTMModel for multivariate data")
        else:
            # Univariate case - use original logic
            if 'target_0' in available_targets:
                lstm_params['target_col'] = 'target_0'
                print(f"[DEBUG] Using target_col: target_0")
            elif 'y' not in available_targets and len(available_targets) > 0:
                lstm_params['target_col'] = available_targets[0]
                print(f"[DEBUG] Using target_col: {available_targets[0]}")
            else:
                print(f"[DEBUG] No target column detection needed, using default")
    
    # Create model_params with default values for initial model creation
    model_params = {}
    for k, v in lstm_params.items():
        if isinstance(v, list):
            model_params[k] = v[0]
        elif isinstance(v, dict):
            # For range-based parameters, use the middle value
            if 'min' in v and 'max' in v:
                if v['type'] == 'int':
                    model_params[k] = int((v['min'] + v['max']) // 2)
                elif v['type'] == 'float':
                    model_params[k] = float((v['min'] + v['max']) / 2)
            else:
                model_params[k] = v
        else:
            model_params[k] = v
    
    if is_multivariate:
        # Use multivariate LSTM model
        from benchmarking_pipeline.models.multivariate.lstm_model import MultivariateLSTMModel
        lstm_model = MultivariateLSTMModel(model_params)
        print(f"[DEBUG] Created MultivariateLSTMModel")
    else:
        # Use regular univariate LSTM model
        lstm_model = LSTMModel(model_params)
        print(f"[DEBUG] Created univariate LSTMModel")
    
    # Create appropriate tuner
    hp_tuning_method = config.get('model', {}).get('search_method', 'grid')
    tuner = _create_hyperparameter_tuner(lstm_model, lstm_params, hp_tuning_method)
    
    if isinstance(tuner, BayesianHyperparameterTuner):
        validation_score_hyperparameter_tuple = tuner.bayesian_optimization_several_time_series(all_dataset_chunks)
    elif isinstance(tuner, SuccessiveHalvingTuner):
        validation_score_hyperparameter_tuple = tuner.successive_halving_search(all_dataset_chunks)
    elif isinstance(tuner, PopulationBasedTuner):
        validation_score_hyperparameter_tuple = tuner.population_based_search(all_dataset_chunks)
    else:
        validation_score_hyperparameter_tuple = tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(tuner.param_names if hasattr(tuner, 'param_names') else lstm_params.keys())}
    for k in ['units', 'layers', 'batch_size', 'epochs', 'sequence_length', 'forecast_horizon']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    results = tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation LSTM {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('LSTM/' + metric, value, 5)
        if is_multivariate:
            # Multivariate case - use all targets
            all_targets_true = all_dataset_chunks[0].test.targets.values
            y_context = all_dataset_chunks[0].train.targets
            y_target = all_dataset_chunks[0].test.targets
            full_preds = lstm_model.predict(y_context=y_context, y_target=y_target)
            log_preds_vs_true(writer, 'LSTM', all_targets_true, full_preds, 5)
        else:
            # Univariate case - use original logic
            y_true, y_pred = lstm_model.get_last_eval_true_pred()
            # Get all targets from the dataset for comprehensive plotting
            all_targets_true = all_dataset_chunks[0].test.targets.values
            # For multivariate datasets, we need to handle the case where predictions are only for one target
            if all_targets_true.ndim == 2 and y_pred.ndim == 1:
                # Create a predictions array that matches the shape of all targets
                # Fill with NaN for targets we don't have predictions for
                all_targets_pred = np.full_like(all_targets_true, np.nan)
                all_targets_pred[:, 0] = y_pred  # Put predictions in first column
            else:
                all_targets_pred = y_pred
            log_preds_vs_true(writer, 'LSTM', all_targets_true, all_targets_pred, 5)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/LSTM_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print(f"Test Evaluation LSTM {dataset_name}: {tuner.final_evaluation({'learning_rate': 0.001}, all_dataset_chunks)}")
    print("LSTM WORKS!") 

def run_svr(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    svr_params = config['model']['parameters']['SVR']
    
    # Check if we have multivariate data
    is_multivariate = False
    if len(all_dataset_chunks) > 0:
        available_targets = list(all_dataset_chunks[0].train.targets.columns)
        print(f"[DEBUG] Available targets: {available_targets}")
        is_multivariate = len(available_targets) > 1 or (len(available_targets) == 1 and available_targets[0] != 'y')
        print(f"[DEBUG] Is multivariate: {is_multivariate}")
        
        if is_multivariate:
            print(f"[DEBUG] Using MultivariateWrapper for SVR")
        else:
            # Univariate case - use original logic
            if 'target_0' in available_targets:
                svr_params['target_col'] = 'target_0'
                print(f"[DEBUG] Using target_col: target_0")
            elif 'y' not in available_targets and len(available_targets) > 0:
                svr_params['target_col'] = available_targets[0]
                print(f"[DEBUG] Using target_col: {available_targets[0]}")
            else:
                print(f"[DEBUG] No target column detection needed, using default")
    def cast_svr_param(key, value):
        # Handle range-based parameters (dictionaries)
        if isinstance(value, dict):
            # For range-based parameters, use the first value or a default
            if 'min' in value and 'max' in value:
                if value['type'] == 'int':
                    return int((value['min'] + value['max']) // 2)  # Use middle value
                elif value['type'] == 'float':
                    return float((value['min'] + value['max']) / 2)  # Use middle value
            return value  # Return as-is for other dict types
        
        # Handle regular parameters
        if key in ['C', 'epsilon']:
            return float(value)
        elif key in ['lookback_window', 'forecast_horizon', 'random_state']:
            return int(value)
        return value
    
    # Create model_params with default values for initial model creation
    model_params = {}
    for k, v in svr_params.items():
        if isinstance(v, list):
            model_params[k] = cast_svr_param(k, v[0])
        elif isinstance(v, dict):
            model_params[k] = cast_svr_param(k, v)
        else:
            model_params[k] = cast_svr_param(k, v)
    if is_multivariate:
        # Use multivariate wrapper
        from benchmarking_pipeline.models.multivariate_wrapper import MultivariateWrapper
        svr_model = MultivariateWrapper(SVRModel, model_params)
        print(f"[DEBUG] Created MultivariateWrapper for SVR")
    else:
        # Use regular model
        svr_model = SVRModel(model_params)
        print(f"[DEBUG] SVR model target_col: {svr_model.target_col}")
    
    # Create appropriate tuner
    hp_tuning_method = config.get('model', {}).get('search_method', 'grid')
    tuner = _create_hyperparameter_tuner(svr_model, svr_params, hp_tuning_method)
    
    if isinstance(tuner, BayesianHyperparameterTuner):
        validation_score_hyperparameter_tuple = tuner.bayesian_optimization_several_time_series(all_dataset_chunks)
    elif isinstance(tuner, SuccessiveHalvingTuner):
        validation_score_hyperparameter_tuple = tuner.successive_halving_search(all_dataset_chunks)
    elif isinstance(tuner, PopulationBasedTuner):
        validation_score_hyperparameter_tuple = tuner.population_based_search(all_dataset_chunks)
    else:
        validation_score_hyperparameter_tuple = tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    
    best_hyperparameters_dict = {k: cast_svr_param(k, validation_score_hyperparameter_tuple[1][i]) for i, k in enumerate(tuner.param_names if hasattr(tuner, 'param_names') else svr_params.keys())}
    results = tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation SVR {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('SVR/' + metric, value, 9)
        chunk = all_dataset_chunks[0]
        lookback_window = svr_model.lookback_window if hasattr(svr_model, 'lookback_window') else 10
        
        if is_multivariate:
            # Multivariate case - use all targets
            all_targets_true = chunk.test.targets.values
            y_context = chunk.train.targets
            y_target = chunk.test.targets
            full_preds = svr_model.predict(y_context=y_context, y_target=y_target)
            log_preds_vs_true(writer, 'SVR', all_targets_true, full_preds, 9)
        else:
            # Univariate case - use original logic
            all_targets_true = chunk.test.targets.values
            y_context = chunk.train.targets.iloc[:, 0].values[-lookback_window:]
            y_target = chunk.test.targets.iloc[:, 0].values
            full_preds = svr_model.predict(y_context=y_context, y_target=y_target)
            # Create predictions array that matches all targets shape
            if all_targets_true.ndim == 2:
                all_targets_pred = np.full_like(all_targets_true, np.nan)
                all_targets_pred[:, 0] = full_preds  # Put predictions in first column
            else:
                all_targets_pred = full_preds
            log_preds_vs_true(writer, 'SVR', all_targets_true, all_targets_pred, 9)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/SVR_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("SVR WORKS!")

def run_seasonalnaive(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    sn_params = config['model']['parameters']['SeasonalNaive']
    
    # Auto-detect target column for multivariate datasets
    if len(all_dataset_chunks) > 0:
        available_targets = list(all_dataset_chunks[0].train.targets.columns)
        if 'target_0' in available_targets:
            # Use first target for multivariate datasets
            sn_params['target_col'] = 'target_0'
        elif 'y' not in available_targets and len(available_targets) > 0:
            # Use first available target if 'y' doesn't exist
            sn_params['target_col'] = available_targets[0]
    # Create model_params with default values for initial model creation
    model_params = {}
    for k, v in sn_params.items():
        if isinstance(v, list):
            model_params[k] = v[0]
        elif isinstance(v, dict):
            # For range-based parameters, use the middle value
            if 'min' in v and 'max' in v:
                if v['type'] == 'int':
                    model_params[k] = int((v['min'] + v['max']) // 2)
                elif v['type'] == 'float':
                    model_params[k] = float((v['min'] + v['max']) / 2)
            else:
                model_params[k] = v
        else:
            model_params[k] = v
    seasonal_naive_model = SeasonalNaiveModel(model_params)
    # No hyperparameter tuning for SeasonalNaive, just use model_params
    results = {}  # You may want to implement evaluation logic here
    print(f"Final Evaluation SeasonalNaive {dataset_name}: {results}")
    if writer is not None:
        chunk = all_dataset_chunks[0]
        lookback_window = model_params['sp'] if 'sp' in model_params else 7
        # Get all targets for plotting
        all_targets_true = chunk.test.targets.values
        # For now, use the first target for predictions (since Seasonal Naive is single-target)
        y_context = chunk.train.targets.iloc[:, 0].values[-lookback_window:]
        y_target = chunk.test.targets.iloc[:, 0].values
        # Train the model before predicting
        seasonal_naive_model.train(y_context=y_context, y_target=y_target)
        full_preds = seasonal_naive_model.predict(y_context=y_context, y_target=y_target)
        # Create predictions array that matches all targets shape
        if all_targets_true.ndim == 2:
            all_targets_pred = np.full_like(all_targets_true, np.nan)
            all_targets_pred[:, 0] = full_preds  # Put predictions in first column
        else:
            all_targets_pred = full_preds
        log_preds_vs_true(writer, 'SeasonalNaive', all_targets_true, all_targets_pred, 10)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/SeasonalNaive_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("SeasonalNaive WORKS!")

def run_exponential_smoothing(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    es_params = config['model']['parameters']['ExponentialSmoothing']
    
    # Auto-detect target column for multivariate datasets
    if len(all_dataset_chunks) > 0:
        available_targets = list(all_dataset_chunks[0].train.targets.columns)
        if 'target_0' in available_targets:
            # Use first target for multivariate datasets
            es_params['target_col'] = 'target_0'
        elif 'y' not in available_targets and len(available_targets) > 0:
            # Use first available target if 'y' doesn't exist
            es_params['target_col'] = available_targets[0]
    # Create model_params with default values for initial model creation
    model_params = {}
    for k, v in es_params.items():
        if isinstance(v, list):
            model_params[k] = v[0]
        elif isinstance(v, dict):
            # For range-based parameters, use the middle value
            if 'min' in v and 'max' in v:
                if v['type'] == 'int':
                    model_params[k] = int((v['min'] + v['max']) // 2)
                elif v['type'] == 'float':
                    model_params[k] = float((v['min'] + v['max']) / 2)
            else:
                model_params[k] = v
        else:
            model_params[k] = v
    es_model = ExponentialSmoothingModel(model_params)
    
    # Create appropriate tuner
    hp_tuning_method = config.get('model', {}).get('search_method', 'grid')
    tuner = _create_hyperparameter_tuner(es_model, es_params, hp_tuning_method)
    
    if isinstance(tuner, BayesianHyperparameterTuner):
        validation_score_hyperparameter_tuple = tuner.bayesian_optimization_several_time_series(all_dataset_chunks)
    elif isinstance(tuner, SuccessiveHalvingTuner):
        validation_score_hyperparameter_tuple = tuner.successive_halving_search(all_dataset_chunks)
    elif isinstance(tuner, PopulationBasedTuner):
        validation_score_hyperparameter_tuple = tuner.population_based_search(all_dataset_chunks)
    else:
        validation_score_hyperparameter_tuple = tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(tuner.param_names if hasattr(tuner, 'param_names') else es_params.keys())}
    results = tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation ExponentialSmoothing {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('ExponentialSmoothing/' + metric, value, 11)
        y_true, y_pred = es_model.get_last_eval_true_pred()
        # Get all targets from the dataset for comprehensive plotting
        all_targets_true = all_dataset_chunks[0].test.targets.values
        # For multivariate datasets, we need to handle the case where predictions are only for one target
        if all_targets_true.ndim == 2 and y_pred.ndim == 1:
            # Create a predictions array that matches the shape of all targets
            # Fill with NaN for targets we don't have predictions for
            all_targets_pred = np.full_like(all_targets_true, np.nan)
            all_targets_pred[:, 0] = y_pred  # Put predictions in first column
        else:
            all_targets_pred = y_pred
        log_preds_vs_true(writer, 'ExponentialSmoothing', all_targets_true, all_targets_pred, 11)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/ExponentialSmoothing_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("ExponentialSmoothing WORKS!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking pipeline with specified config file.")
    parser.add_argument('--config', type=str, default='benchmarking_pipeline/configs/univariate_config.yaml', help='Path to the config YAML file')
    parser.add_argument('--hp_tuning', type=str, default='auto', choices=['grid', 'bayesian', 'successive_halving', 'pbt', 'auto'], help='Hyperparameter tuning method: grid, bayesian, successive_halving, pbt, or auto (detect from config)')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override search method if specified
    if args.hp_tuning != 'auto':
        config['model']['search_method'] = args.hp_tuning
    
    runner = BenchmarkRunner(config=config, config_path=config_path)
    runner.run() 