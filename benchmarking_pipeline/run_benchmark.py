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
import numpy as np
import pandas as pd
import re
import os
import json

class BenchmarkRunner:
    def __init__(self, config):
        """
        Initialize benchmark runner with configuration.
        
        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config = config
        self.logger = Logger(config)
        
    def run(self):
        """Execute the end-to-end benchmarking pipeline."""
        writer = SummaryWriter(log_dir="runs/benchmark_runner/{}".format(time.strftime("%Y%m%d-%H%M%S")))
        data_loader = DataLoader({"dataset": {
            "path": "/Users/aryannair/smlcrm-benchmark/benchmarking_pipeline/datasets/australian_electricity_demand",
            "name": "australian_electricity_demand",
            "split_ratio": [0.8, 0.1, 0.1]
        }})
        all_australian_chunks = data_loader.load_several_chunks(2)
        preprocessor = Preprocessor({"dataset": {"normalize": False}})
        all_australian_chunks = [preprocessor.preprocess(chunk).data for chunk in all_australian_chunks]
        print(f"[DEBUG] Number of chunks: {len(all_australian_chunks)}")
        if len(all_australian_chunks) > 0:
            print(f"[DEBUG] First chunk train shape: {all_australian_chunks[0].train.features.shape}")
        run_arima(all_australian_chunks, writer)
        # run_theta(all_australian_chunks, writer)
        # run_deep_ar(all_australian_chunks, writer)
        run_xgboost(all_australian_chunks, writer)
        run_random_forest(all_australian_chunks, writer)
        run_prophet(all_australian_chunks, writer)
        run_lstm(all_australian_chunks, writer)
        run_croston_classic(all_australian_chunks, writer)
        writer.close()
        print("\nTensorBoard logs saved in 'runs/benchmark_runner/'. To view, run: tensorboard --logdir runs/benchmark_runner\n")
        self.logger.log_metrics({"status": "Pipeline completed"}, step=0)

def _extract_number_before_capital(freq_str):
    match = re.match(r'(\d+)?[A-Z]', freq_str)
    if match:
        return int(match.group(1)) if match.group(1) else 1
    else:
        raise ValueError(f"Invalid frequency string: {freq_str}")

def run_arima(all_australian_chunks, writer=None):
    print("[DEBUG] Starting ARIMA...")
    arima_model = ARIMAModel({
        "p": -1,
        "d": -1,
        "q": -1,
        "target_col": "y",
        "exog_cols": None,
        "loss_functions": ["mae"],
        "primary_loss": "mae",
        "forecast_horizon": 100
    })
    arima_hyperparameter_tuner = HyperparameterTuner(arima_model,{
        "p": [0, 1, 2],
        "d": [0, 1],
        "q": [0, 1, 2]
        }, False)
    print("[DEBUG] Running ARIMA hyperparameter tuning...")
    validation_score_hyperparameter_tuple = arima_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
    best_hyperparameters_dict = {
        "p": validation_score_hyperparameter_tuple[1][0],
        "d": validation_score_hyperparameter_tuple[1][1],
        "q": validation_score_hyperparameter_tuple[1][2]
        }
    print("[DEBUG] Running ARIMA final evaluation...")
    results = arima_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)
    print(f"Final Evaluation ARIMA australia: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('ARIMA/' + metric, value)
    # Save results locally
    os.makedirs('results', exist_ok=True)
    with open('results/ARIMA.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test Evaluation ARIMA australia: {arima_hyperparameter_tuner.final_evaluation({'p':0,'d':2,'q':0}, all_australian_chunks)}")
    print("[DEBUG] Finished ARIMA.")
    print("ARIMA WORKS!")

def run_theta(all_australian_chunks, writer=None):
    theta_model = ThetaModel({
        'sp':-1,
        "forecast_horizon": 100
        })
    theta_hyperparameter_tuner = HyperparameterTuner(theta_model, {
        'sp':[1,2,3,4]  
        }, False)
    validation_score_hyperparameter_tuple = theta_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
    print(validation_score_hyperparameter_tuple)
    print(f"Testing the theta model's get_params(...):\n\n{theta_model.get_params()}")
    best_hyperparameters_dict = {
        "sp": validation_score_hyperparameter_tuple[1][0], 
        }
    results = theta_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)
    print(f"Final Evaluation Theta australia: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('Theta/' + metric, value)
    # Save results locally
    os.makedirs('results', exist_ok=True)
    with open('results/Theta.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test Evaluation Theta australia: {theta_hyperparameter_tuner.final_evaluation({'sp':3,}, all_australian_chunks)}")
    print("Theta WORKS!")

def run_deep_ar(all_australian_chunks, writer=None):
    deep_ar_model = DeepARModel({
        "hidden_size": 2,
        "rnn_layers" : 2,
        "dropout" : 0.1,
        "batch_size" : 1,
        "learning_rate" : 0.001,
        "target_col" : 'y',
        "feature_cols" : None,
        "forecast_horizon" : 100,
        "epochs": 1,
        "num_workers":4
    })
    deep_ar_hyperparameter_tuner = HyperparameterTuner(deep_ar_model, {
        'rnn_layers' : [2,3],
    }, False)
    validation_score_hyperparameter_tuple = deep_ar_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
    best_hyperparameters_dict = {
        "rnn_layers": validation_score_hyperparameter_tuple[1][0], 
        }
    results = deep_ar_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)
    print(f"Final Evaluation DeepAR australia: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('DeepAR/' + metric, value)
    # Save results locally
    os.makedirs('results', exist_ok=True)
    with open('results/DeepAR.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test Evaluation DeepAR australia: {deep_ar_hyperparameter_tuner.final_evaluation({'rnn_layers':2,}, all_australian_chunks)}")
    print("Deep AR WORKS!")

def run_xgboost(all_australian_chunks, writer=None):
    xgb_model = XGBoostModel({
        "lookback_window": 10,
        "forecast_horizon": 10,
        "model_params": {
            "n_estimators": 50,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1
        }
    })
    xgb_hyperparameter_tuner = HyperparameterTuner(xgb_model, {
        "lookback_window": [10],
        "model_params__n_estimators": [50],
        "model_params__max_depth": [3],
        "model_params__learning_rate": [0.01],
    }, False)
    validation_score_hyperparameter_tuple = xgb_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
    best_hyperparameters_dict = {
        "lookback_window": validation_score_hyperparameter_tuple[1][0],
        "model_params__n_estimators": validation_score_hyperparameter_tuple[1][1],
        "model_params__max_depth": validation_score_hyperparameter_tuple[1][2],
        "model_params__learning_rate": validation_score_hyperparameter_tuple[1][3],
    }
    results = xgb_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)
    print(f"Final Evaluation XGBoost australia: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('XGBoost/' + metric, value)
    # Save results locally
    os.makedirs('results', exist_ok=True)
    with open('results/XGBoost.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test Evaluation XGBoost australia: {xgb_hyperparameter_tuner.final_evaluation({'lookback_window': 10, 'model_params__n_estimators': 50, 'model_params__max_depth': 5, 'model_params__learning_rate': 0.1}, all_australian_chunks)}")
    print("XGBoost WORKS!")

def run_random_forest(all_australian_chunks, writer=None):
    rf_model = RandomForestModel({
        "lookback_window": 10,
        "forecast_horizon": 5,
        "n_estimators": 50,
        "max_depth": 5,
        "random_state": 42,
        "n_jobs": -1
    })
    rf_hyperparameter_tuner = HyperparameterTuner(rf_model, {
        "lookback_window": [10, 20],
        "n_estimators": [10, 50],
        "max_depth": [3,7],
    }, False)
    validation_score_hyperparameter_tuple = rf_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
    best_hyperparameters_dict = {
        "lookback_window": validation_score_hyperparameter_tuple[1][0],
        "n_estimators": validation_score_hyperparameter_tuple[1][1],
        "max_depth": validation_score_hyperparameter_tuple[1][2],
    }
    results = rf_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)
    print(f"Final Evaluation Random Forest australia: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('RandomForest/' + metric, value)
    # Save results locally
    os.makedirs('results', exist_ok=True)
    with open('results/RandomForest.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test Evaluation Random Forest australia: {rf_hyperparameter_tuner.final_evaluation({'lookback_window': 10, 'model_params__n_estimators': 50, 'model_params__max_depth': 5}, all_australian_chunks)}")
    print("Random Forest WORKS!")

def run_prophet(all_australian_chunks, writer=None):
    prophet_model = ProphetModel({
        "model_params": {
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
        }
    })
    prophet_hyperparameter_tuner = HyperparameterTuner(prophet_model, {
        "seasonality_mode": ["additive", "multiplicative"],
        "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
    }, "prophet")
    validation_score_hyperparameter_tuple = prophet_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
    best_hyperparameters_dict = {
        "seasonality_mode": validation_score_hyperparameter_tuple[1][0],
        "changepoint_prior_scale": validation_score_hyperparameter_tuple[1][1],
        "seasonality_prior_scale": validation_score_hyperparameter_tuple[1][2],
    }
    results = prophet_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)
    print(f"Final Evaluation Prophet australia: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('Prophet/' + metric, value)
    # Save results locally
    os.makedirs('results', exist_ok=True)
    with open('results/Prophet.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test Evaluation Prophet australia: {prophet_hyperparameter_tuner.final_evaluation({'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0}, all_australian_chunks)}")
    print("Prophet WORKS!")

def run_croston_classic(all_australian_chunks, writer=None):
    y_train = np.array([0, 0, 5, 0, 0, 0, 3, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 6, 0, 0, 1, 4, 0, 0, 5, 19, 0, 0, 0])
    y_target = np.zeros(5)  # Forecast 5 steps ahead
    croston_model = CrostonClassicModel({
        "alpha": 0.1,
        "target_col": "y",
        "loss_functions": ["mae"],
        "primary_loss": "mae",
        "forecast_horizon": 5
    })
    croston_model.train(y_context=y_train)
    preds = croston_model.predict(y_context=y_train, y_target=y_target)
    print("Synthetic Croston predictions:", preds.flatten())
    print("Model summary:", croston_model.get_model_summary())
    # Evaluate on real data
    croston_hyperparameter_tuner = HyperparameterTuner(croston_model, {"alpha": [0.1]}, False)
    results = croston_hyperparameter_tuner.final_evaluation({"alpha": 0.1}, all_australian_chunks)
    print(f"Final Evaluation CrostonClassic australia: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('CrostonClassic/' + metric, value)
    # Save results locally
    os.makedirs('results', exist_ok=True)
    with open('results/CrostonClassic.json', 'w') as f:
        json.dump(results, f, indent=2)

def run_lstm(all_australian_chunks, writer=None):
    lstm_model = LSTMModel({
        "units": 32,
        "layers": 1,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 20,
        "sequence_length": 20,
        "target_col": "y",
        "loss_functions": ["mae"],
        "primary_loss": "mae",
        "forecast_horizon": 10
    })
    lstm_hyperparameter_tuner = HyperparameterTuner(lstm_model, {
        "units": [32],
        "layers": [1],
    }, False)
    validation_score_hyperparameter_tuple = lstm_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_australian_chunks)
    best_hyperparameters_dict = {
        "units": validation_score_hyperparameter_tuple[1][0],
        "layers": validation_score_hyperparameter_tuple[1][1]
    }
    results = lstm_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_australian_chunks)
    print(f"Final Evaluation LSTM australia: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('LSTM/' + metric, value)
    # Save results locally
    os.makedirs('results', exist_ok=True)
    with open('results/LSTM.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test Evaluation LSTM australia: {lstm_hyperparameter_tuner.final_evaluation({'units': 32, 'layers': 2}, all_australian_chunks)}")
    print("LSTM WORKS!") 

if __name__ == "__main__":
    runner = BenchmarkRunner(config={})
    runner.run() 