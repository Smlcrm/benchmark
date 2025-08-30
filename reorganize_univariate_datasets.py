#!/usr/bin/env python3
"""
Script to reorganize CSV files in datasets/univariate into individual folders.
Each CSV file will be moved to a folder named after the CSV file (without .csv extension).
"""

import os
import shutil
from pathlib import Path

def reorganize_univariate_datasets():
    # Define the source directory
    source_dir = Path("benchmarking_pipeline/datasets/univariate")
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return
    
    # Get all CSV files in the source directory
    csv_files = list(source_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in the source directory")
        return
    
    print(f"Found {len(csv_files)} CSV files to reorganize")
    
    # Process each CSV file
    for csv_file in csv_files:
        # Get the filename without extension
        folder_name = csv_file.stem
        
        # Create the target folder path
        target_folder = source_dir / folder_name
        
        try:
            # Create the target folder if it doesn't exist
            target_folder.mkdir(exist_ok=True)
            
            # Move the CSV file to the target folder
            target_path = target_folder / csv_file.name
            shutil.move(str(csv_file), str(target_path))
            
            print(f"✓ Moved {csv_file.name} to {target_folder.name}/")
            
        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {e}")
    
    print(f"\nReorganization complete! {len(csv_files)} CSV files moved to individual folders.")

if __name__ == "__main__":
    reorganize_univariate_datasets()
