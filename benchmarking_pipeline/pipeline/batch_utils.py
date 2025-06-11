"""
Utilities for batching Arrow data into pandas DataFrames.
"""
from typing import Iterator, Optional, Union, Tuple
import pyarrow as pa
import pandas as pd
import numpy as np

def batch_arrow_data(
    features: Union[pa.Table, pa.RecordBatch], 
    labels: Optional[pa.Array] = None,
    batch_size: int = 1000,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Iterator[Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]]:
    """
    Batch Arrow data into pandas DataFrames.
    
    Args:
        features: Arrow Table or RecordBatch containing features
        labels: Optional Arrow Array containing labels
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data before batching
        seed: Random seed for shuffling
        
    Yields:
        If labels is None:
            pd.DataFrame: Batch of features as pandas DataFrame
        If labels is provided:
            Tuple[pd.DataFrame, np.ndarray]: Tuple of (features batch, labels batch)
    """
    # Get total number of rows
    n_samples = len(features)
    
    # Create index array
    indices = np.arange(n_samples)
    
    # Shuffle if requested
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
    
    # Generate batches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Take batch from features
        if isinstance(features, pa.Table):
            features_batch = features.take(batch_indices).to_pandas()
        else:  # RecordBatch
            features_batch = features.take(batch_indices).to_pandas()
        
        if labels is not None:
            # Take batch from labels
            labels_batch = labels.take(batch_indices).to_numpy()
            yield features_batch, labels_batch
        else:
            yield features_batch

def batch_dataset_split(
    split: 'DatasetSplit',  # Forward reference to avoid circular import
    batch_size: int = 1000,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Iterator[Tuple[pd.DataFrame, np.ndarray]]:
    """
    Batch a DatasetSplit into pandas DataFrames.
    
    Args:
        split: DatasetSplit containing features and labels
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data before batching
        seed: Random seed for shuffling
        
    Yields:
        Tuple[pd.DataFrame, np.ndarray]: Tuple of (features batch, labels batch)
    """
    return batch_arrow_data(
        features=split.features,
        labels=split.labels,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed
    )

def batch_dataset(
    dataset: 'Dataset',  # Forward reference to avoid circular import
    batch_size: int = 1000,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Tuple[Iterator[Tuple[pd.DataFrame, np.ndarray]], ...]:
    """
    Batch a Dataset into pandas DataFrames.
    
    Args:
        dataset: Dataset containing train/val/test splits
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data before batching
        seed: Random seed for shuffling
        
    Returns:
        Tuple of iterators for train, validation, and test batches
        Each iterator yields Tuple[pd.DataFrame, np.ndarray]
    """
    train_batches = batch_dataset_split(dataset.train, batch_size, shuffle, seed)
    val_batches = batch_dataset_split(dataset.validation, batch_size, False, None)  # Don't shuffle val
    test_batches = batch_dataset_split(dataset.test, batch_size, False, None)  # Don't shuffle test
    
    return train_batches, val_batches, test_batches 