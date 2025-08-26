#!/usr/bin/env python3
"""
Test script to check how many targets are being loaded from the Beijing dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmarking_pipeline.pipeline.data_loader import DataLoader

def test_beijing_targets():
    """Test how many targets are loaded from Beijing dataset."""
    
    # Create data loader config for Beijing dataset
    config = {
        "dataset": {
            "path": "benchmarking_pipeline/datasets/BEIJING_SUBWAY_30MIN",
            "name": "BEIJING_SUBWAY_30MIN",
            "split_ratio": [0.8, 0.1, 0.1]
        }
    }
    
    data_loader = DataLoader(config)
    
    # Load the first chunk
    dataset = data_loader.load_single_chunk(1)
    
    print("=== Beijing Dataset Analysis ===")
    print(f"Train features shape: {dataset.train.features.shape}")
    print(f"Train features columns: {list(dataset.train.features.columns)}")
    print(f"Number of feature columns: {len(dataset.train.features.columns)}")
    
    print(f"\nValidation features shape: {dataset.validation.features.shape}")
    print(f"Test features shape: {dataset.test.features.shape}")
    
    # Check if exogenous features are present
    if dataset.train.features is not None:
        print(f"\nFeatures shape: {dataset.train.features.shape}")
        print(f"Features columns: {list(dataset.train.features.columns)}")
    else:
        print("\nNo features found")
    
    # Show first few values of each feature
    print("\n=== First 5 values of each feature ===")
    for col in dataset.train.features.columns:
        print(f"{col}: {dataset.train.features[col].head().tolist()}")
    
    # Test passes if we get here without errors
    assert dataset is not None
    assert hasattr(dataset.train, 'features')
    assert hasattr(dataset.validation, 'features')
    assert hasattr(dataset.test, 'features')

if __name__ == "__main__":
    dataset = test_beijing_targets() 