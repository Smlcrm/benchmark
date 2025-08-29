import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np
import io

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
        self.verbose = self.config.get('verbose', False)
        
        # Create a unique directory for each run to keep experiments separate
        run_name = self.config.get('run_name', 'run')
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_log_dir = os.path.join(self.base_log_dir, f"{run_name}_{timestamp}")
        
        # TensorBoard writer for metrics and structured data
        self.writer = tf.summary.create_file_writer(self.run_log_dir)
        
        # Also set up a basic text logger for general status messages
        self._setup_text_logger()
        
        if self.verbose:
            self.log_info(f"Logger initialized with TensorBoard directory: {self.run_log_dir}")
        
    def _setup_text_logger(self):
        """Sets up a basic Python logger for text messages to the console."""
        # Set log level based on verbose setting
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Configures the root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True # Allows re-configuration in interactive environments
        )
        self.text_logger = logging.getLogger(self.__class__.__name__)
        
    def log_info(self, message):
        """Logs an informational text message."""
        if self.verbose:
            self.text_logger.info(message)
        else:
            # For non-verbose mode, only show important messages
            if any(keyword in message.lower() for keyword in ['error', 'warning', 'completed', 'failed', 'success']):
                self.text_logger.info(message)
        
    def log_warning(self, message):
        """Logs a warning text message."""
        self.text_logger.warning(f"⚠️  {message}")
        
    def log_error(self, message):
        """Logs an error text message."""
        self.text_logger.error(f"❌ {message}")
        
    def log_success(self, message):
        """Logs a success message."""
        self.text_logger.info(f"✅ {message}")
        
    def log_progress(self, message):
        """Logs progress information (only in verbose mode)."""
        if self.verbose:
            self.text_logger.info(f"🔄 {message}")

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
        
        # Log key metrics to console in verbose mode
        if self.verbose and metrics:
            self.log_progress(f"Metrics logged to TensorBoard: {list(metrics.keys())}")

    def log_figure(self, figure, tag: str, step: int):
        """
        Log a Matplotlib figure to TensorBoard.
        """
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        with self.writer.as_default():
            tf.summary.image(tag, image, step=step)
        self.writer.flush()

    def log_image_file(self, image_path: str, tag: str, step: int):
        """
        Log an image from disk to TensorBoard.
        """
        try:
            with open(image_path, 'rb') as f:
                data = f.read()
            image = tf.image.decode_image(data, channels=4)
            image = tf.expand_dims(image, 0)
            with self.writer.as_default():
                tf.summary.image(tag, image, step=step)
            self.writer.flush()
        except Exception as e:
            self.log_error(f"Failed to log image file '{image_path}': {e}")

    def log_training_progress(self, model_name, epoch, loss, val_loss=None, step=None):
        """
        Log training progress for real-time monitoring.
        
        Args:
            model_name (str): Name of the model being trained
            epoch (int): Current epoch number
            loss (float): Training loss
            val_loss (float, optional): Validation loss
            step (int, optional): Global step for TensorBoard
        """
        if step is None:
            step = epoch
            
        # Log to TensorBoard
        with self.writer.as_default():
            tf.summary.scalar(f"{model_name}/train_loss", loss, step=step)
            if val_loss is not None:
                tf.summary.scalar(f"{model_name}/val_loss", val_loss, step=step)
        self.writer.flush()
        
        # Log to console
        if self.verbose:
            if val_loss is not None:
                self.log_progress(f"{model_name} - Epoch {epoch}: Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                self.log_progress(f"{model_name} - Epoch {epoch}: Train Loss: {loss:.4f}")

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
        
        if self.verbose:
            self.log_info(f"Hyperparameters logged: {list(sanitized_hparams.keys())}")
            self.log_info(f"Final metrics logged: {list(metrics.keys())}")
        
    def close(self):
        """Closes the TensorBoard writer to ensure all data is written to disk."""
        self.writer.close()
        if self.verbose:
            self.log_info(f"TensorBoard logs saved to: {self.run_log_dir}")
            self.log_info("To view logs, run: tensorboard --logdir logs/tensorboard")
