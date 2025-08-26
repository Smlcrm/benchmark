"""
Command-line interface for the benchmarking pipeline.
"""

import argparse
import yaml
import logging
from datetime import datetime
import os

from pipeline.data_loader import DataLoader
from pipeline.preprocessor import Preprocessor
from pipeline.trainer import Trainer
from pipeline.logger import Logger
from pipeline.visualizer import Visualizer
from pipeline.data_types import Dataset, PreprocessedData, EvaluationMetrics

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmarking Pipeline CLI')
    parser.add_argument('--config', type=str, default='benchmarking_pipeline/configs/all_model_test.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save outputs (plots, metrics, etc.)')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        logging.info(f"Model type: {config.get('model', {}).get('name', 'Not specified')}")
        return config

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def main():
    """Main entry point for the CLI."""
    logger = setup_logging()
    start_time = datetime.now()
    logger.info("Starting benchmarking pipeline")

    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Create output directory
    ensure_dir(args.output_dir)
    plots_dir = os.path.join(args.output_dir, 'plots')
    ensure_dir(plots_dir)

    # Initialize components
    logger.info("Initializing pipeline components")
    data_loader = DataLoader(config)
    preprocessor = Preprocessor(config)
    trainer = Trainer(config)
    metrics_logger = Logger(config)
    visualizer = Visualizer(config)

    # Load data
    logger.info("Loading raw dataset")
    raw_dataset = data_loader.load_data()
    logger.info(f"Loaded dataset with {len(raw_dataset.train.features)} training samples, "
                f"{len(raw_dataset.validation.features)} validation samples, "
                f"{len(raw_dataset.test.features)} test samples")

    # Preprocess data
    logger.info("Preprocessing dataset")
    preprocessed_data = preprocessor.preprocess(raw_dataset)
    logger.info("Preprocessing complete")
    logger.info(f"Preprocessing steps applied: {preprocessed_data.preprocessing_info}")

    # Train model
    logger.info("Training ARIMA model")
    model = ARIMA(preprocessed_data.data.train.features['y'])
    logger.info("Model initialized, starting training")
    fitted_model = trainer.train(model, preprocessed_data.data.train.features)
    logger.info("Model training complete")

    # Generate forecasts
    logger.info("Generating forecasts")
    train_y = preprocessed_data.data.train.features['y']
    val_y = preprocessed_data.data.validation.features['y']
    test_y = preprocessed_data.data.test.features['y']

    logger.info(f"Forecasting {len(val_y)} validation steps")
    val_forecast = fitted_model.forecast(steps=len(val_y))
    
    logger.info(f"Forecasting {len(test_y)} test steps")
    test_forecast = fitted_model.forecast(steps=len(test_y))

    # Evaluate
    logger.info("Computing evaluation metrics")
    metrics = {
        "Validation MAE": mean_absolute_error(val_y, val_forecast),
        "Validation RMSE": mean_squared_error(val_y, val_forecast) ** 0.5,
        "Test MAE": mean_absolute_error(test_y, test_forecast),
        "Test RMSE": mean_squared_error(test_y, test_forecast) ** 0.5
    }

    # Log metrics
    logger.info("Logging evaluation metrics")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    metrics_logger.log_metrics(metrics)

    # Visualize predictions
    logger.info("Generating validation set predictions plot")
    visualizer.plot_predictions(
        y_true=val_y,
        y_pred=val_forecast,
        title="Validation Set: Predictions vs Actual Values",
        save_path=os.path.join(plots_dir, 'validation_predictions.png')
    )
    
    logger.info("Generating test set predictions plot")
    visualizer.plot_predictions(
        y_true=test_y,
        y_pred=test_forecast,
        title="Test Set: Predictions vs Actual Values",
        save_path=os.path.join(plots_dir, 'test_predictions.png')
    )
    
    logger.info("Generating residual analysis plots")
    visualizer.plot_residuals(
        y_true=val_y,
        y_pred=val_forecast,
        title="Validation Set: Residual Analysis",
        save_path=os.path.join(plots_dir, 'validation_residuals.png')
    )
    visualizer.plot_residuals(
        y_true=test_y,
        y_pred=test_forecast,
        title="Test Set: Residual Analysis",
        save_path=os.path.join(plots_dir, 'test_residuals.png')
    )

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Pipeline completed in {duration.total_seconds():.2f} seconds")
    logger.info(f"Plots saved in: {plots_dir}")

if __name__ == '__main__':
    main()
