import os
import ast
import pandas as pd
from typing import Dict, Any
from .data_types import Dataset, DatasetSplit  # assumes these are in data_types.py

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
    
    def load_data(self, chunk_index) -> Dataset:
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

        return Dataset(
            train=DatasetSplit(features=train[['y']], labels=None),
            validation=DatasetSplit(features=val[['y']], labels=None),
            test=DatasetSplit(features=test[['y']], labels=None),
            name=self.name,
            metadata={'start': start, 'freq': freq}
        )
