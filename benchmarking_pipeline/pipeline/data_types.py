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
    features: Union[np.ndarray, pd.DataFrame]
    labels: Union[np.ndarray, pd.Series]
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
class FeatureSet:
    """Container for extracted features."""
    features: Dataset
    feature_info: Dict[str, Any]

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    metrics: Dict[str, float]
    additional_info: Optional[Dict[str, Any]] = None

@dataclass
class ModelArtifacts:
    """Container for model artifacts and metadata."""
    model: Any  # The actual model object
    parameters: Dict[str, Any]
    training_history: Optional[Dict[str, List[float]]] = None
    metadata: Optional[Dict[str, Any]] = None 