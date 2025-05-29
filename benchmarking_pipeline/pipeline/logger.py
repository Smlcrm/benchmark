"""
Logging utilities for the pipeline.
"""

import logging

class Logger:
    def __init__(self, config):
        """
        Initialize logger with configuration.
        
        Args:
            config: Configuration dictionary with logging parameters
        """
        self.config = config
        self._setup_logger()
        
    def _setup_logger(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def log_metrics(self, metrics):
        """
        Log evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value}") 