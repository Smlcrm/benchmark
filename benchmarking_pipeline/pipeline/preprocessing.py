"""
Data preprocessing utilities.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .data_types import Dataset, DatasetSplit, PreprocessedData

class Preprocessor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Full configuration dictionary that may contain:
                - dataset: Dataset-specific config (univariate_config.yaml style)
                - preprocessing: Preprocessing-specific config (default_config.yaml style)
        """
        self.config = config
        self.scalers = {}
        
        # Extract preprocessing config from either location
        self.preprocessing_config = self._get_preprocessing_config()
        
    def _get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Extract preprocessing configuration from either dataset or preprocessing section.
        Handles both config styles:
        - univariate_config.yaml: dataset.normalize, dataset.handle_missing
        - default_config.yaml: preprocessing.normalize, preprocessing.remove_outliers
        """
        # Start with preprocessing section if it exists
        preprocessing_config = self.config.get('preprocessing', {})
        
        # Override/supplement with dataset-level config if it exists
        dataset_config = self.config.get('dataset', {})
        if dataset_config:
            # Map dataset-level configs to preprocessing configs
            if 'normalize' in dataset_config:
                preprocessing_config['normalize'] = dataset_config['normalize']
            if 'handle_missing' in dataset_config:
                preprocessing_config['handle_missing'] = dataset_config['handle_missing']
                
        return preprocessing_config
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        """
        if not self.preprocessing_config.get('handle_missing', True):
            return df
            
        strategy = self.preprocessing_config.get('handle_missing', 'mean')
        
        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'mean':
            return df.fillna(df.mean(numeric_only=True))
        elif strategy == 'median':
            return df.fillna(df.median(numeric_only=True))
        elif strategy == 'interpolate':
            return df.interpolate()
        elif strategy == 'forward_fill':
            return df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            return df.fillna(method='bfill')
        else:
            # Default to forward fill then backward fill
            return df.fillna(method='ffill').fillna(method='bfill')
            
    def _normalize_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features.
        """
        if not self.preprocessing_config.get('normalize', True):
            return df
            
        method = self.preprocessing_config.get('normalization_method', 'standard')
        df_normalized = df.copy()
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if is_training:
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()  # Default
                    
                df_normalized[col] = scaler.fit_transform(df[[col]]).flatten()
                self.scalers[col] = scaler
            else:
                if col in self.scalers:
                    scaler = self.scalers[col]
                    df_normalized[col] = scaler.transform(df[[col]]).flatten()
                    
        return df_normalized
        
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using z-score method.
        """
        if not self.preprocessing_config.get('remove_outliers', False):
            return df
            
        threshold = self.preprocessing_config.get('outlier_threshold', 3)
        df_clean = df.copy()
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df_clean = df_clean[z_scores < threshold]
            
        return df_clean
        
    def preprocess(self, data: Dataset) -> PreprocessedData:
        """
        Main preprocessing method.
        
        Args:
            data: Dataset object containing train/validation/test splits
            
        Returns:
            PreprocessedData object with processed data and metadata
        """
        # Process training data first (to fit scalers)
        train_features = data.train.features
        if isinstance(train_features, pd.DataFrame):
            train_df = train_features.copy()
        else:
            train_df = pd.DataFrame(train_features)
            
        # Apply preprocessing steps
        train_df = self._handle_missing_values(train_df)
        train_df = self._remove_outliers(train_df)
        train_df = self._normalize_features(train_df, is_training=True)
        
        # Create processed training split
        train_split = DatasetSplit(
            features=train_df,
            labels=data.train.labels,
            metadata=data.train.metadata
        )
        
        # Process validation data
        val_features = data.validation.features
        if isinstance(val_features, pd.DataFrame):
            val_df = val_features.copy()
        else:
            val_df = pd.DataFrame(val_features)
            
        val_df = self._handle_missing_values(val_df)
        val_df = self._normalize_features(val_df, is_training=False)
        
        val_split = DatasetSplit(
            features=val_df,
            labels=data.validation.labels,
            metadata=data.validation.metadata
        )
        
        # Process test data
        test_features = data.test.features
        if isinstance(test_features, pd.DataFrame):
            test_df = test_features.copy()
        else:
            test_df = pd.DataFrame(test_features)
            
        test_df = self._handle_missing_values(test_df)
        test_df = self._normalize_features(test_df, is_training=False)
        
        test_split = DatasetSplit(
            features=test_df,
            labels=data.test.labels,
            metadata=data.test.metadata
        )
        
        # Create preprocessed dataset
        preprocessed_dataset = Dataset(
            train=train_split,
            validation=val_split,
            test=test_split,
            name=data.name,
            metadata=data.metadata
        )
        
        # Create preprocessing info
        preprocessing_info = {
            'normalize': self.preprocessing_config.get('normalize', True),
            'normalization_method': self.preprocessing_config.get('normalization_method', 'standard'),
            'remove_outliers': self.preprocessing_config.get('remove_outliers', False),
            'outlier_threshold': self.preprocessing_config.get('outlier_threshold', 3),
            'handle_missing': self.preprocessing_config.get('handle_missing', True),
            'scalers': {col: {
                'type': type(scaler).__name__,
                'mean': getattr(scaler, 'mean_', None),
                'scale': getattr(scaler, 'scale_', None)
            } for col, scaler in self.scalers.items()}
        }
        
        return PreprocessedData(
            data=preprocessed_dataset,
            preprocessing_info=preprocessing_info
        ) 