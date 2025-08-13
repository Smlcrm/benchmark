#!/usr/bin/env python3
"""
Debug script to examine the raw Beijing data structure.
"""

import ast
import pandas as pd

def debug_beijing_raw():
    """Debug the raw Beijing data structure."""
    
    # Read the raw CSV file
    file_path = "benchmarking_pipeline/datasets/BEIJING_SUBWAY_30MIN/chunk001.csv"
    df = pd.read_csv(file_path)
    row = df.iloc[0]
    
    print("=== Raw Beijing Data Analysis ===")
    print(f"Columns in CSV: {list(df.columns)}")
    print(f"Number of rows: {len(df)}")
    
    # Parse the target column
    target_raw = row['target']
    print(f"\nRaw target string length: {len(target_raw)}")
    print(f"First 200 characters: {target_raw[:200]}")
    print(f"Last 200 characters: {target_raw[-200:]}")
    
    # Parse the target
    target = ast.literal_eval(target_raw)
    print(f"\nParsed target type: {type(target)}")
    print(f"Parsed target length: {len(target)}")
    
    # Count non-None series
    non_none_count = 0
    for i, series in enumerate(target):
        if series is not None:
            non_none_count += 1
            print(f"Series {i}: length {len(series)}, first 5 values: {series[:5]}")
        else:
            print(f"Series {i}: None")
    
    print(f"\nTotal series in target: {len(target)}")
    print(f"Non-None series: {non_none_count}")
    
    # Check if there are nested lists
    print("\n=== Checking for nested structure ===")
    for i, series in enumerate(target):
        if series is not None and isinstance(series, list):
            print(f"Series {i}: list with {len(series)} elements")
            if len(series) > 0:
                print(f"  First element type: {type(series[0])}")
                if isinstance(series[0], list):
                    print(f"  First element length: {len(series[0])}")
                    print(f"  First element first 5 values: {series[0][:5]}")
        elif series is not None:
            print(f"Series {i}: not a list, type: {type(series)}")

if __name__ == "__main__":
    debug_beijing_raw() 