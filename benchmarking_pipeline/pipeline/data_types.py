"""
Core data types used throughout the benchmarking pipeline.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

@dataclass
class DatasetSplit:
    """Represents a dataset split (train/val/test)."""
    targets: Union[np.ndarray, pd.DataFrame]  # Target variables (univariate or multivariate)
    timestamps: np.ndarray
    features: Optional[Union[np.ndarray, pd.DataFrame]] = None  # Exogenous variables (optional)
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Dataset:
    """Container for all dataset splits."""
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    name: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PreprocessedData:
    """Container for preprocessed data."""
    data: Dataset
    preprocessing_info: Dict[str, Any]

@dataclass
class ModelArtifacts:
    """Container for model artifacts and metadata."""
    model: Any  # The actual model object
    parameters: Dict[str, Any]
    training_history: Optional[Dict[str, List[float]]] = None
    metadata: Optional[Dict[str, Any]] = None 