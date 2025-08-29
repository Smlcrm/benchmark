"""
Command-line interface for the benchmarking pipeline.

This module provides a CLI for running time series forecasting benchmarks with various models.
It supports both traditional statistical models and modern foundation models.
"""

import argparse
import yaml
import logging
from datetime import datetime
import os

from benchmarking_pipeline.pipeline.data_loader import DataLoader
from benchmarking_pipeline.pipeline.preprocessor import Preprocessor
from benchmarking_pipeline.pipeline.trainer import Trainer
from benchmarking_pipeline.pipeline.logger import Logger
from benchmarking_pipeline.pipeline.visualizer import Visualizer
from benchmarking_pipeline.pipeline.data_types import Dataset, PreprocessedData, EvaluationMetrics

# Note: ARIMA import removed as it's not used in the current implementation
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
    parser = argparse.ArgumentParser(
        description='Benchmarking Pipeline CLI for time series forecasting models',
        epilog='Example: python -m benchmarking_pipeline.cli --config configs/all_model_test.yaml'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='benchmarking_pipeline/configs/all_model_test.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Directory to save outputs (plots, metrics, etc.)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/tensorboard',
        help='Base directory for TensorBoard logs'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default='run',
        help='Optional run name used to namespace logs'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging for pipeline and TensorBoard summaries'
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        
        # Log model configuration
        model_config = config.get('model', {})
        if model_config:
            model_names = list(model_config.keys())
            logging.info(f"Configured models: {', '.join(model_names)}")
        else:
            logging.warning("No models configured in the configuration file")
        
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

    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config(args.config)
        # Merge CLI logging options into config (CLI takes precedence)
        config['log_dir'] = args.log_dir
        config['run_name'] = args.run_name
        config['verbose'] = bool(args.verbose)
        
        # Create output directory
        ensure_dir(args.output_dir)
        plots_dir = os.path.join(args.output_dir, 'plots')
        ensure_dir(plots_dir)

        # Merge output_dir into config so downstream components can save plots
        config['output_dir'] = args.output_dir

        # Initialize components
        logger.info("Initializing pipeline components")
        data_loader = DataLoader(config)
        preprocessor = Preprocessor(config)
        trainer = Trainer(config)
        metrics_logger = Logger(config)
        # Emit an initial event so TensorBoard detects this run immediately
        try:
            metrics_logger.log_metrics({"run_started": 1}, step=0, model_name="pipeline")
            # Optionally record configured models as hparams for the HParams dashboard
            metrics_logger.log_hparams(config.get('model', {}), {})
        except Exception as tb_err:
            logger.warning(f"Failed to write initial TensorBoard events: {tb_err}")
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

        # Note: ARIMA-specific code removed as it's not part of the general pipeline
        # The actual model training should be handled by the Trainer class
        logger.info("Starting model training and evaluation")
        
        # TODO: Implement proper model training through the Trainer class
        # This should handle both traditional and foundation models based on config
        
        logger.warning("ARIMA-specific training code removed. Implement proper model training through Trainer class.")

        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Pipeline completed in {duration.total_seconds():.2f} seconds")
        logger.info(f"Plots saved in: {plots_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
