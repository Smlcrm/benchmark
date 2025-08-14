"""
Command-line interface for the benchmarking pipeline.
"""

import argparse
import yaml
import logging
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.pipeline.trainer import Trainer
from benchmarking_pipeline.pipeline.logger import Logger
from benchmarking_pipeline.pipeline.visualizer import Visualizer
from benchmarking_pipeline.pipeline.data_types import Dataset, PreprocessedData, EvaluationMetrics

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmarking Pipeline CLI')
    parser.add_argument('--config', type=str, default='configs/deterministic_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save outputs (plots, metrics, etc.)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging and detailed output')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable TensorBoard logging for real-time monitoring')
    parser.add_argument('--log-dir', type=str, default='logs/tensorboard',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--run-name', type=str, default='cli_run',
                        help='Name for this experiment run')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from {config_path}")
            logging.info(f"Model type: {config.get('model', {}).get('name', 'Not specified')}")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        sys.exit(1)

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def main():
    """Main entry point for the CLI."""
    # Parse arguments first
    args = parse_args()
    
    # Set up logging based on verbose flag
    logger = setup_logging(args.verbose)
    start_time = datetime.now()
    
    if args.verbose:
        logger.info("Starting benchmarking pipeline with verbose logging")
    else:
        logger.info("Starting benchmarking pipeline")
    
    if args.tensorboard:
        logger.info("TensorBoard logging enabled - training progress will be logged in real-time")

    # Parse arguments and load config
    config = load_config(args.config)
    
    # Add CLI arguments to config for logger
    config.update({
        'verbose': args.verbose,
        'tensorboard': args.tensorboard,
        'log_dir': args.log_dir,
        'run_name': args.run_name
    })
    
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
    
    if args.verbose:
        logger.info(f"Loaded dataset with {len(raw_dataset.train.features)} training samples, "
                    f"{len(raw_dataset.validation.features)} validation samples, "
                    f"{len(raw_dataset.test.features)} test samples")
    else:
        logger.info("Dataset loaded successfully")

    # Preprocess data
    logger.info("Preprocessing dataset")
    preprocessed_data = preprocessor.preprocess(raw_dataset)
    logger.info("Preprocessing complete")
    
    if args.verbose:
        logger.info(f"Preprocessing steps applied: {preprocessed_data.preprocessing_info}")

    # Train model
    logger.info("Training ARIMA model")
    model = ARIMA(preprocessed_data.data.train.targets['y'])
    logger.info("Model initialized, starting training")
    
    if args.tensorboard:
        logger.info("TensorBoard logging active - monitor training progress in real-time")
    
    fitted_model = trainer.train(model, preprocessed_data.data.train.targets)
    logger.info("Model training complete")

    # Generate forecasts
    logger.info("Generating forecasts")
    train_y = preprocessed_data.data.train.targets['y']
    val_y = preprocessed_data.data.validation.targets['y']
    test_y = preprocessed_data.data.test.targets['y']

    if args.verbose:
        logger.info(f"Forecasting {len(val_y)} validation steps")
    val_forecast = fitted_model.forecast(steps=len(val_y))
    
    if args.verbose:
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
        if args.verbose:
            logger.info(f"{metric_name}: {value:.4f}")
        else:
            # Show only key metrics in non-verbose mode
            if 'Test' in metric_name:
                logger.info(f"{metric_name}: {value:.4f}")
    
    metrics_logger.log_metrics(metrics)

    # Visualize predictions
    if args.verbose:
        logger.info("Generating validation set predictions plot")
    visualizer.plot_predictions(
        y_true=val_y,
        y_pred=val_forecast,
        title="Validation Set: Predictions vs Actual Values",
        save_path=os.path.join(plots_dir, 'validation_predictions.png')
    )
    
    if args.verbose:
        logger.info("Generating test set predictions plot")
    visualizer.plot_predictions(
        y_true=test_y,
        y_pred=test_forecast,
        title="Test Set: Predictions vs Actual Values",
        save_path=os.path.join(plots_dir, 'test_predictions.png')
    )
    
    if args.verbose:
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
    
    if args.verbose:
        logger.info(f"Pipeline completed in {duration.total_seconds():.2f} seconds")
        logger.info(f"Plots saved in: {plots_dir}")
        if args.tensorboard:
            logger.info(f"TensorBoard logs saved in: {args.log_dir}")
            logger.info("To view logs, run: tensorboard --logdir logs/tensorboard")
    else:
        logger.info(f"Pipeline completed successfully in {duration.total_seconds():.1f}s")
        logger.info(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
