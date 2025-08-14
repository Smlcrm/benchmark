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
    
    print("=== Beijing Dataset Target Analysis ===")
    print(f"Train targets shape: {dataset.train.targets.shape}")
    print(f"Train targets columns: {list(dataset.train.targets.columns)}")
    print(f"Number of target columns: {len(dataset.train.targets.columns)}")
    
    print(f"\nValidation targets shape: {dataset.validation.targets.shape}")
    print(f"Test targets shape: {dataset.test.targets.shape}")
    
    # Check if exogenous features are present
    if dataset.train.features is not None:
        print(f"\nExogenous features shape: {dataset.train.features.shape}")
        print(f"Exogenous features columns: {list(dataset.train.features.columns)}")
    else:
        print("\nNo exogenous features found")
    
    # Show first few values of each target
    print("\n=== First 5 values of each target ===")
    for col in dataset.train.targets.columns:
        print(f"{col}: {dataset.train.targets[col].head().tolist()}")
    
    return dataset

if __name__ == "__main__":
    dataset = test_beijing_targets() 