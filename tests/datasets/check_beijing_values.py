#!/usr/bin/env python3
"""
Check for actual non-None values in the Beijing dataset targets.
"""

import ast
import pandas as pd
import numpy as np

def check_beijing_values():
    """Check for actual non-None values in Beijing dataset."""
    
    # Read the raw CSV file
    file_path = "benchmarking_pipeline/datasets/BEIJING_SUBWAY_30MIN/chunk001.csv"
    df = pd.read_csv(file_path)
    row = df.iloc[0]
    
    # Parse the target
    target = ast.literal_eval(row['target'])
    
    print("=== Beijing Dataset Value Analysis ===")
    
    for i, series in enumerate(target):
        if series is not None:
            # Convert to numpy array for easier analysis
            series_array = np.array(series)
            
            # Count non-None values
            non_none_mask = series_array != None
            non_none_count = np.sum(non_none_mask)
            
            print(f"\nSeries {i} (target_{i}):")
            print(f"  Total length: {len(series)}")
            print(f"  Non-None values: {non_none_count}")
            print(f"  None values: {len(series) - non_none_count}")
            print(f"  Percentage non-None: {non_none_count/len(series)*100:.1f}%")
            
            if non_none_count > 0:
                # Get first few non-None values
                non_none_values = series_array[non_none_mask]
                print(f"  First 10 non-None values: {non_none_values[:10].tolist()}")
                
                # Get some statistics
                print(f"  Min value: {np.min(non_none_values)}")
                print(f"  Max value: {np.max(non_none_values)}")
                print(f"  Mean value: {np.mean(non_none_values):.2f}")
                print(f"  Std value: {np.std(non_none_values):.2f}")
            else:
                print(f"  All values are None!")

if __name__ == "__main__":
    check_beijing_values() 