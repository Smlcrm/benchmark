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
            print(f"[DEBUG] First chunk train shape: {all_dataset_chunks[0].train.features.shape}")
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
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()
    fig, ax = plt.subplots()
    ax.plot(y_true, label='True')
    ax.plot(y_pred, label='Predicted')
    ax.set_title(f'{model_name} Predictions vs True')
    ax.legend()
    writer.add_figure(f'{model_name}/pred_vs_true', fig, global_step=step)
    plt.close(fig)

def run_arima(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    arima_params = config['model']['parameters']['ARIMA']
    model_params = {}
    for k, v in arima_params.items():
        value = v[0] if isinstance(v, list) else v
        if k in ['p', 'd', 'q']:
            model_params[k] = int(value)
            print(f"[DEBUG] ARIMA param {k}: value={value}, type={type(value)} -> casted type={type(model_params[k])}")
        else:
            model_params[k] = value
    print(f"[DEBUG] ARIMA model_params: {model_params}")
    arima_model = ARIMAModel(model_params)
    hyper_grid = {k: v for k, v in arima_params.items() if isinstance(v, list)}
    arima_hyperparameter_tuner = HyperparameterTuner(arima_model, hyper_grid, False)
    validation_score_hyperparameter_tuple = arima_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
    # Cast p, d, q to int if present
    for k in ['p', 'd', 'q']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    results = arima_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation ARIMA {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('ARIMA/' + metric, value, 1)
        y_true, y_pred = arima_model.get_last_eval_true_pred()
        log_preds_vs_true(writer, 'ARIMA', y_true, y_pred, 1)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/ARIMA_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("ARIMA WORKS!")

def run_theta(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    theta_params = config['model']['parameters']['Theta']
    model_params = {k: v[0] if isinstance(v, list) else v for k, v in theta_params.items()}
    # Force additive seasonality for Theta
    model_params['seasonality_mode'] = 'additive'
    theta_model = ThetaModel(model_params)
    hyper_grid = {k: v for k, v in theta_params.items() if isinstance(v, list)}
    theta_hyperparameter_tuner = HyperparameterTuner(theta_model, hyper_grid, False)
    validation_score_hyperparameter_tuple = theta_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
    for k in ['sp', 'forecast_horizon']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    # Also force additive seasonality in best hyperparameters
    best_hyperparameters_dict['seasonality_mode'] = 'additive'
    results = theta_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation Theta {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('Theta/' + metric, value, 7)
        y_true, y_pred = theta_model.get_last_eval_true_pred()
        log_preds_vs_true(writer, 'Theta', y_true, y_pred, 7)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f"results/Theta_{dataset_name}_{time.strftime('%Y%m%d-%H%M%S')}.csv", index=False)
    print("Theta WORKS!")

def run_deep_ar(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    deep_ar_params = config['model']['parameters']['DeepAR']
    model_params = {k: v[0] if isinstance(v, list) else v for k, v in deep_ar_params.items()}
    deep_ar_model = DeepARModel(model_params)
    hyper_grid = {k: v for k, v in deep_ar_params.items() if isinstance(v, list)}
    deep_ar_hyperparameter_tuner = HyperparameterTuner(deep_ar_model, hyper_grid, False)
    validation_score_hyperparameter_tuple = deep_ar_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
    for k in ['hidden_size', 'rnn_layers', 'batch_size', 'epochs', 'forecast_horizon', 'num_workers']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    results = deep_ar_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation DeepAR {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('DeepAR/' + metric, value)
        y_true, y_pred = deep_ar_model.get_last_eval_true_pred()
        log_preds_vs_true(writer, 'DeepAR', y_true, y_pred, 8)
    os.makedirs('results', exist_ok=True)
    with open(f'results/DeepAR_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Deep AR WORKS!")

def run_xgboost(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    xgb_params = config['model']['parameters']['XGBoost']
    model_params = {k: v[0] if isinstance(v, list) else v for k, v in xgb_params.items()}
    xgb_model = XGBoostModel(model_params)
    hyper_grid = {k: v for k, v in xgb_params.items() if isinstance(v, list)}
    xgb_hyperparameter_tuner = HyperparameterTuner(xgb_model, hyper_grid, False)
    validation_score_hyperparameter_tuple = xgb_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
    for k in ['lookback_window', 'forecast_horizon', 'n_estimators', 'max_depth', 'random_state', 'n_jobs']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    results = xgb_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation XGBoost {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('XGBoost/' + metric, value, 2)
        y_true, y_pred = xgb_model.get_last_eval_true_pred()
        log_preds_vs_true(writer, 'XGBoost', y_true, y_pred, 2)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/XGBoost_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("XGBoost WORKS!")

def run_random_forest(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    rf_params = config['model']['parameters']['RandomForest']
    model_params = {k: v[0] if isinstance(v, list) else v for k, v in rf_params.items()}
    rf_model = RandomForestModel(model_params)
    hyper_grid = {k: v for k, v in rf_params.items() if isinstance(v, list)}
    rf_hyperparameter_tuner = HyperparameterTuner(rf_model, hyper_grid, False)
    validation_score_hyperparameter_tuple = rf_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
    for k in ['lookback_window', 'forecast_horizon', 'n_estimators', 'max_depth', 'random_state', 'n_jobs']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    results = rf_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation Random Forest {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('RandomForest/' + metric, value, 3)
        # Use rolling_predict for full-length predictions for plotting
        chunk = all_dataset_chunks[0]
        lookback_window = rf_model.lookback_window
        y_context = chunk.train.features.values.flatten()[-lookback_window:]
        y_target = chunk.test.features.values.flatten()
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
    # Cast parameters to correct types
    def cast_param(key, value):
        if key in ['seasonality_mode']:
            return str(value)
        elif key in ['changepoint_prior_scale', 'seasonality_prior_scale']:
            return float(value)
        elif key in ['yearly_seasonality', 'weekly_seasonality', 'daily_seasonality']:
            if isinstance(value, str):
                return value.lower() == 'true'
            return bool(value)
        return value
    model_params = {k: cast_param(k, v[0] if isinstance(v, list) else v) for k, v in prophet_params.items()}
    prophet_model = ProphetModel(model_params)
    hyper_grid = {k: v for k, v in prophet_params.items() if isinstance(v, list)}
    # Also cast hyper_grid values for grid search
    for k in hyper_grid:
        hyper_grid[k] = [cast_param(k, val) for val in hyper_grid[k]]
    prophet_hyperparameter_tuner = HyperparameterTuner(prophet_model, hyper_grid, "prophet")
    validation_score_hyperparameter_tuple = prophet_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    best_hyperparameters_dict = {k: cast_param(k, validation_score_hyperparameter_tuple[1][i]) for i, k in enumerate(hyper_grid.keys())}
    results = prophet_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation Prophet {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('Prophet/' + metric, value, 4)
        y_true, y_pred = prophet_model.get_last_eval_true_pred()
        log_preds_vs_true(writer, 'Prophet', y_true, y_pred, 4)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/Prophet_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print(f"Test Evaluation Prophet {dataset_name}: {prophet_hyperparameter_tuner.final_evaluation({'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0}, all_dataset_chunks)}")
    print("Prophet WORKS!")

def run_croston_classic(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    croston_params = config['model']['parameters']['CrostonClassic']
    model_params = {k: v[0] if isinstance(v, list) else v for k, v in croston_params.items()}
    croston_model = CrostonClassicModel(model_params)
    hyper_grid = {k: v for k, v in croston_params.items() if isinstance(v, list)}
    croston_hyperparameter_tuner = HyperparameterTuner(croston_model, hyper_grid, False)
    results = croston_hyperparameter_tuner.final_evaluation({"alpha": 0.1}, all_dataset_chunks)
    print(f"Final Evaluation CrostonClassic {dataset_name}: {results}")
    if writer is not None:
        y_true, y_pred = croston_model.get_last_eval_true_pred()
        print(f"Croston y_true: {y_true}")
        print(f"Croston y_pred: {y_pred}")
        if y_true is not None and y_pred is not None:
            print(f"Croston y_true shape: {y_true.shape}")
            print(f"Croston y_pred shape: {y_pred.shape}")
        log_preds_vs_true(writer, 'CrostonClassic', y_true, y_pred, 6)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/CrostonClassic_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)

def run_lstm(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    lstm_params = config['model']['parameters']['LSTM']
    model_params = {k: v[0] if isinstance(v, list) else v for k, v in lstm_params.items()}
    lstm_model = LSTMModel(model_params)
    hyper_grid = {k: v for k, v in lstm_params.items() if isinstance(v, list)}
    lstm_hyperparameter_tuner = HyperparameterTuner(lstm_model, hyper_grid, False)
    validation_score_hyperparameter_tuple = lstm_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
    for k in ['units', 'layers', 'batch_size', 'epochs', 'sequence_length', 'forecast_horizon']:
        if k in best_hyperparameters_dict:
            best_hyperparameters_dict[k] = int(best_hyperparameters_dict[k])
    results = lstm_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation LSTM {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('LSTM/' + metric, value, 5)
        y_true, y_pred = lstm_model.get_last_eval_true_pred()
        log_preds_vs_true(writer, 'LSTM', y_true, y_pred, 5)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/LSTM_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print(f"Test Evaluation LSTM {dataset_name}: {lstm_hyperparameter_tuner.final_evaluation({'learning_rate': 0.001}, all_dataset_chunks)}")
    print("LSTM WORKS!") 

def run_svr(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    svr_params = config['model']['parameters']['SVR']
    def cast_svr_param(key, value):
        if key in ['C', 'epsilon']:
            return float(value)
        elif key in ['lookback_window', 'forecast_horizon', 'random_state']:
            return int(value)
        return value
    model_params = {k: cast_svr_param(k, v[0] if isinstance(v, list) else v) for k, v in svr_params.items()}
    svr_model = SVRModel(model_params)
    hyper_grid = {k: v for k, v in svr_params.items() if isinstance(v, list)}
    for k in hyper_grid:
        hyper_grid[k] = [cast_svr_param(k, val) for val in hyper_grid[k]]
    svr_hyperparameter_tuner = HyperparameterTuner(svr_model, hyper_grid, False)
    validation_score_hyperparameter_tuple = svr_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    best_hyperparameters_dict = {k: cast_svr_param(k, validation_score_hyperparameter_tuple[1][i]) for i, k in enumerate(hyper_grid.keys())}
    results = svr_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation SVR {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('SVR/' + metric, value, 9)
        chunk = all_dataset_chunks[0]
        lookback_window = svr_model.lookback_window if hasattr(svr_model, 'lookback_window') else 10
        y_context = chunk.train.features.values.flatten()[-lookback_window:]
        y_target = chunk.test.features.values.flatten()
        full_preds = svr_model.predict(y_context=y_context, y_target=y_target)
        log_preds_vs_true(writer, 'SVR', y_target, full_preds, 9)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/SVR_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("SVR WORKS!")

def run_seasonalnaive(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    sn_params = config['model']['parameters']['SeasonalNaive']
    model_params = {k: v[0] if isinstance(v, list) else v for k, v in sn_params.items()}
    seasonal_naive_model = SeasonalNaiveModel(model_params)
    # No hyperparameter tuning for SeasonalNaive, just use model_params
    results = {}  # You may want to implement evaluation logic here
    print(f"Final Evaluation SeasonalNaive {dataset_name}: {results}")
    if writer is not None:
        chunk = all_dataset_chunks[0]
        lookback_window = model_params['sp'] if 'sp' in model_params else 7
        y_context = chunk.train.features.values.flatten()[-lookback_window:]
        y_target = chunk.test.features.values.flatten()
        # Train the model before predicting
        seasonal_naive_model.train(y_context=y_context, y_target=y_target)
        full_preds = seasonal_naive_model.predict(y_context=y_context, y_target=y_target)
        log_preds_vs_true(writer, 'SeasonalNaive', y_target, full_preds, 10)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/SeasonalNaive_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("SeasonalNaive WORKS!")

def run_exponential_smoothing(all_dataset_chunks, writer=None, config=None, config_path=None):
    dataset_name = config['dataset']['name']
    es_params = config['model']['parameters']['ExponentialSmoothing']
    model_params = {k: v[0] if isinstance(v, list) else v for k, v in es_params.items()}
    es_model = ExponentialSmoothingModel(model_params)
    hyper_grid = {k: v for k, v in es_params.items() if isinstance(v, list)}
    es_hyperparameter_tuner = HyperparameterTuner(es_model, hyper_grid, False)
    validation_score_hyperparameter_tuple = es_hyperparameter_tuner.hyperparameter_grid_search_several_time_series(all_dataset_chunks)
    best_hyperparameters_dict = {k: validation_score_hyperparameter_tuple[1][i] for i, k in enumerate(hyper_grid.keys())}
    results = es_hyperparameter_tuner.final_evaluation(best_hyperparameters_dict, all_dataset_chunks)
    print(f"Final Evaluation ExponentialSmoothing {dataset_name}: {results}")
    if writer is not None:
        for metric, value in results.items():
            writer.add_scalar('ExponentialSmoothing/' + metric, value, 11)
        y_true, y_pred = es_model.get_last_eval_true_pred()
        log_preds_vs_true(writer, 'ExponentialSmoothing', y_true, y_pred, 11)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame([results]).to_csv(f'results/ExponentialSmoothing_{dataset_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv', index=False)
    print("ExponentialSmoothing WORKS!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarking pipeline with specified config file.")
    parser.add_argument('--config', type=str, default='benchmarking_pipeline/configs/univariate_config.yaml', help='Path to the config YAML file')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    runner = BenchmarkRunner(config=config, config_path=config_path)
    runner.run() 