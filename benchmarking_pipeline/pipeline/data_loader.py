import os
import ast
import pandas as pd
from typing import Dict, Any, List
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
        """
        self.config = config
        self.dataset_cfg = config.get('dataset', {})
        self.path = self.dataset_cfg.get('path')
        self.name = self.dataset_cfg.get('name')
        self.split_ratio = self.dataset_cfg.get('split_ratio', [0.6, 0.2, 0.2])
    
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
    
    #doesn't work with exogenous variables for now
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
        time_index = pd.date_range(start=start, periods=len(target), freq=freq)
        ts_df = pd.DataFrame({'ds': time_index, 'y': target})

        # Split the time series
        train_end = int(len(ts_df) * self.split_ratio[0])
        val_end = int(len(ts_df) * (self.split_ratio[0] + self.split_ratio[1]))
        train = ts_df.iloc[:train_end]
        val = ts_df.iloc[train_end:val_end]
        test = ts_df.iloc[val_end:]

        self._generate_timestamp_list(start, freq, 4)
        

        train_timestamps_plus_val_start = self._generate_timestamp_list(start,freq,len(train[['y']])+1)
        
        train_timestamps = train_timestamps_plus_val_start[:len(train[['y']])]
        val_start = train_timestamps_plus_val_start[-1]

        val_timestamps_plus_test_start = self._generate_timestamp_list(val_start,freq,len(val[['y']])+1)

        val_timestamps = val_timestamps_plus_test_start[:len(val[['y']])]
        test_start = val_timestamps_plus_test_start[-1]

        test_timestamps = self._generate_timestamp_list(test_start,freq,len(test[['y']]))
        
        #print("train timestamps", train_timestamps)

        return Dataset(
            train=DatasetSplit(features=train[['y']], labels=None, timestamps=train_timestamps),
            validation=DatasetSplit(features=val[['y']], labels=None, timestamps=val_timestamps),
            test=DatasetSplit(features=test[['y']], labels=None, timestamps=test_timestamps),
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
