import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Configure TensorFlow threading before import
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

"""
Logging utilities for the pipeline
"""

class Logger:
    def __init__(self, config):
        """
        Initialize logger with configuration.
        
        Args:
            config (dict): Configuration dictionary with logging parameters.
                           Should contain 'log_dir' (base directory for logs)
                           and optionally 'run_name' for this specific experiment.
        """
        self.config = config
        self.base_log_dir = self.config.get('log_dir', 'logs/tensorboard')
        
        # Create a unique directory for each run to keep experiments separate
        run_name = self.config.get('run_name', 'run')
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_log_dir = os.path.join(self.base_log_dir, f"{run_name}_{timestamp}")
        
        # TensorBoard writer for metrics and structured data
        self.writer = tf.summary.create_file_writer(self.run_log_dir)
        
        # Also set up a basic text logger for general status messages
        self._setup_text_logger()
        
    def _setup_text_logger(self):
        """Sets up a basic Python logger for text messages to the console."""
        # Configures the root logger
        logging.basicConfig(
            level=self.config.get('log_level', logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True # Allows re-configuration in interactive environments
        )
        self.text_logger = logging.getLogger(self.__class__.__name__)
        
    def log_info(self, message):
        """Logs an informational text message."""
        self.text_logger.info(message)
        
    def log_warning(self, message):
        """Logs a warning text message."""
        self.text_logger.warning(message)

    def log_error(self, message):
        """Logs an error text message."""
        self.text_logger.error(message)

    def log_metrics(self, metrics, step, model_name=""):
        """
        Log evaluation metrics to TensorBoard.
        
        Args:
            metrics (dict): Dictionary of metrics to log.
            step (int): The current step (e.g., epoch, batch, or experiment ID).
            model_name (str, optional): A prefix for metric names to group them in TensorBoard.
        """
        group_prefix = f"{model_name}/" if model_name else ""
        with self.writer.as_default():
            for metric_name, value in metrics.items():
                if value is None or (isinstance(value, (float, int)) and np.isnan(value)):
                    continue

                if isinstance(value, dict): # For nested results like quantile losses
                    for sub_metric, sub_value in value.items():
                        tag = f"{group_prefix}{metric_name}/{sub_metric}"
                        tf.summary.scalar(tag, sub_value, step=step)
                elif isinstance(value, (np.ndarray, list)):
                    # For arrays log each element and the mean
                    value_arr = np.asarray(value)
                    tf.summary.scalar(f"{group_prefix}{metric_name}/mean", np.nanmean(value_arr), step=step)
                    for i, elem in enumerate(value_arr):
                        tf.summary.scalar(f"{group_prefix}{metric_name}/series_{i}", elem, step=step)
                elif isinstance(value, (float, np.floating, int)):
                    tf.summary.scalar(f"{group_prefix}{metric_name}", value, step=step)
                else:
                    self.log_warning(f"Skipping metric '{metric_name}' with unsupported type: {type(value)}")
        self.writer.flush()


    def log_hparams(self, hparams, metrics):
        """
        Log a set of hyperparameters and the resulting metrics for comparison
        in TensorBoard's HParams dashboard.

        Args:
            hparams (dict): Dictionary of hyperparameters used for the run 
                            (e.g., {'learning_rate': 0.1, 'model': 'XGBoost'}).
            metrics (dict): Dictionary of final metrics for this run (e.g., {'mae': 12.3}).
        """
        with self.writer.as_default():
            # Sanitize hparams for TensorBoard (it has specific type requirements)
            sanitized_hparams = {k: v for k, v in hparams.items() if isinstance(v, (str, bool, int, float))}
            hp.hparams(sanitized_hparams)
            for metric_name, value in metrics.items():
                if isinstance(value, (float, np.floating, int)):
                    # Log final metrics in a way that HParams can read
                    tf.summary.scalar(f"hparams/{metric_name}", value, step=1)
        self.writer.flush()
        
    def close(self):
        """Closes the TensorBoard writer to ensure all data is written to disk."""
        self.writer.close()
