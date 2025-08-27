import os
import ast
import pandas as pd
from typing import Dict, Any, List, Optional
from .data_types import Dataset, DatasetSplit  # assumes these are in data_types.py
import numpy as np

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration dictionary with data loading parameters:
                - dataset.path: path to dataset directory
                - dataset.name: name of the dataset
                - dataset.chunk_index: which chunk to load (default=1)
                - dataset.split_ratio: list of [train, val, test] ratios
                - model.parameters.{model_name}.target_cols: list of target column names to load
        """
        self.config = config
        self.dataset_cfg = config.get('dataset', {})
        self.path = self.dataset_cfg.get('path')
        self.name = self.dataset_cfg.get('name')
        self.split_ratio = self.dataset_cfg.get('split_ratio', [0.6, 0.2, 0.2])
        
        # Extract target_cols from model configuration
        self.target_cols = self._extract_target_cols()
    
    def _extract_target_cols(self) -> List[str]:
        """
        Extract target_cols from model configuration.
        
        Returns:
            List of target column names
            
        Raises:
            ValueError: If target_cols is not defined in any model's parameters or is None
        """
        model_cfg = self.config.get('model', {})
        parameters = model_cfg.get('parameters', {})
        
        # Look for target_cols in any model's parameters
        for model_params in parameters.values():
            if isinstance(model_params, dict) and 'target_cols' in model_params:
                target_cols = model_params['target_cols']
                if target_cols is None:
                    raise ValueError("target_cols cannot be None. Please provide a list of column names.")
                if not isinstance(target_cols, list) or len(target_cols) == 0:
                    raise ValueError("target_cols must be a non-empty list of column names.")
                return target_cols
        
        # target_cols is mandatory - no backward compatibility
        raise ValueError("target_cols must be defined in model parameters. "
                        "Please add target_cols to at least one model's configuration.")
    
    def _generate_timestamp_list(self, start, freq, horizon) -> List[Any]:
        """
        Takes in a starting timestamp and outputs a list of timestamps with 
        element count equal to horizon.

        Please note the returned list will always contain start as the starting
        time step. Thus, horizon can be thought of as follows: 1 (for start) + 
        number of future timesteps you want to generate = horizon

        Args:
            start: A timestamp-like object (e.g., string, datetime).
            freq: A pandas-compatible frequency string (e.g., 'D', 'H', '30T', etc.)
            horizon: Number of timestamps to generate, including the start.

        Returns:
            List of timestamps, starting from `start`, spaced by `freq`, of length `horizon`.
        """

        # Convert start to pandas.Timestamp
        start_timestamp = pd.to_datetime(start)

        # Convert frequency string to offset
        freq_offset = pd.tseries.frequencies.to_offset(freq)

        # Generate timestamps
        timestamps = [start_timestamp + i * freq_offset for i in range(horizon)]

        return timestamps
    
    def load_single_chunk(self, chunk_index) -> Dataset:
        """
        Load a single chunk and return it as a Dataset object.

        Args:
            chunk_index: the index of the chunk we want the data of

        Returns:
            Dataset object containing train, val, test splits.
        """

        # need to change self.chunk_index so that it can take in several chunks
        file = os.path.join(self.path, f"chunk{chunk_index:03}.csv")
        df = pd.read_csv(file)
        row = df.iloc[0]
        start = pd.to_datetime(row['start'])
        freq = row['freq']
        target = ast.literal_eval(row['target'])
        
        # Handle targets based on target_cols configuration
        # target_cols is now mandatory, so this will always be defined
        if isinstance(target, list) and len(target) > 0:
            if isinstance(target[0], list):
                # 2D structure: [[val1, val2], [val3, val4], ...] - Multiple target series
                if len(target[0]) == len(self.target_cols):
                    # Create DataFrame with user-specified column names
                    targets_df = pd.DataFrame(target, columns=self.target_cols)
                else:
                    # Mismatch between data structure and target_cols
                    raise ValueError(f"Data has {len(target[0])} series but target_cols specifies {len(self.target_cols)} columns")
            else:
                # 1D structure: [val1, val2, val3, ...] - Single univariate time series
                if len(self.target_cols) == 1:
                    targets_df = pd.DataFrame({self.target_cols[0]: target})
                else:
                    raise ValueError(f"Data is univariate but target_cols specifies {len(self.target_cols)} columns")
        else:
            # Single target series (univariate)
            if len(self.target_cols) == 1:
                targets_df = pd.DataFrame({self.target_cols[0]: target})
            else:
                raise ValueError(f"Data is univariate but target_cols specifies {len(self.target_cols)} columns")
        
        # Handle exogenous features if present
        exogenous_df = None
        if 'past_feat_dynamic_real' in row and pd.notna(row['past_feat_dynamic_real']):
            try:
                past_features = ast.literal_eval(row['past_feat_dynamic_real'])
                if isinstance(past_features, list) and len(past_features) > 0:
                    # Multiple exogenous series
                    exog_series = []
                    for i, series in enumerate(past_features):
                        if series is not None:  # Skip None series
                            exog_series.append(pd.Series(series, name=f'feature_{i}'))
                    
                    if exog_series:
                        exogenous_df = pd.concat(exog_series, axis=1)
            except (ValueError, SyntaxError):
                # If parsing fails, treat as no exogenous features
                pass
        
        # Create time index
        time_index = pd.date_range(start=start, periods=len(targets_df), freq=freq)
        
        # Split the data
        train_end = int(len(targets_df) * self.split_ratio[0])
        val_end = int(len(targets_df) * (self.split_ratio[0] + self.split_ratio[1]))
        
        # Split targets
        train_targets = targets_df.iloc[:train_end]
        val_targets = targets_df.iloc[train_end:val_end]
        test_targets = targets_df.iloc[val_end:]
        
        # Split exogenous features if present
        train_features = None
        val_features = None
        test_features = None
        if exogenous_df is not None:
            train_features = exogenous_df.iloc[:train_end]
            val_features = exogenous_df.iloc[train_end:val_end]
            test_features = exogenous_df.iloc[val_end:]

        # Generate timestamps
        train_timestamps_plus_val_start = self._generate_timestamp_list(start, freq, len(train_targets) + 1)
        train_timestamps = train_timestamps_plus_val_start[:len(train_targets)]
        val_start = train_timestamps_plus_val_start[-1]

        val_timestamps_plus_test_start = self._generate_timestamp_list(val_start, freq, len(val_targets) + 1)
        val_timestamps = val_timestamps_plus_test_start[:len(val_targets)]
        test_start = val_timestamps_plus_test_start[-1]

        test_timestamps = self._generate_timestamp_list(test_start, freq, len(test_targets))

        return Dataset(
            train=DatasetSplit(targets=train_targets, features=train_features, timestamps=train_timestamps),
            validation=DatasetSplit(targets=val_targets, features=val_features, timestamps=val_timestamps),
            test=DatasetSplit(targets=test_targets, features=test_features, timestamps=test_timestamps),
            name=self.name,
            metadata={'start': start, 'freq': freq}
        )

    def load_several_chunks(self,upper_chunk_index) -> List[Dataset]:
        """
        Loads several chunks, and returns them as a list of Dataset objects.

        Args:
            upper_chunk_index: the last index of the chunk we want the data of (inclusive)

        Returns:
            List of Dataset objects, with each Dataset object containing train, val, test splits.
        """
        list_of_chunks = []
        for chunk_index in range(1, upper_chunk_index+1):
            list_of_chunks.append(self.load_single_chunk(chunk_index))
        return list_of_chunks
