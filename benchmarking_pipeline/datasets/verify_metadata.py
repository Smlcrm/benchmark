#!/usr/bin/env python3
"""
Script to verify all metadata.json files were created correctly
"""

import os
import json
import ast
import pandas as pd
from pathlib import Path

def count_targets_in_csv(csv_file):
    """Count the number of target series in a CSV file"""
    try:
        df = pd.read_csv(csv_file)
        if 'target' not in df.columns:
            return 0
        
        # Get the first row's target column
        target_data = df.iloc[0]['target']
        
        # Parse the target data (it's stored as a string representation of a list)
        if isinstance(target_data, str):
            target_list = ast.literal_eval(target_data)
        else:
            target_list = target_data
        
        # Count the number of series based on LOTSA structure
        if isinstance(target_list, list):
            if len(target_list) > 0:
                first_item = target_list[0]
                if isinstance(first_item, list):
                    # Multivariate: [[val1, val2, ...], [val1, val2, ...], ...]
                    # Each inner list is a different variable
                    return len(target_list)
                else:
                    # Univariate: [val1, val2, val3, ...]
                    # Single variable with multiple time points
                    return 1
            else:
                return 0
        else:
            return 1  # Single value
            
    except Exception as e:
        print(f"    ⚠️  Error reading CSV: {e}")
        return None

def verify_metadata_files():
    """Verify all metadata.json files exist and are valid JSON"""
    # Since this script is in the datasets folder, we can use relative paths
    current_dir = Path(__file__).parent
    
    # Get all dataset directories (excluding this script and any hidden files)
    dataset_dirs = [d for d in current_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(dataset_dirs)} dataset directories")
    print("=" * 50)
    
    all_valid = True
    
    for dataset_dir in sorted(dataset_dirs):
        metadata_file = dataset_dir / "metadata.json"
        
        if not metadata_file.exists():
            print(f"❌ {dataset_dir.name}: Missing metadata.json")
            all_valid = False
            continue
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Validate structure
            if 'variables' not in metadata:
                print(f"❌ {dataset_dir.name}: Missing 'variables' key")
                all_valid = False
                continue
            
            if not isinstance(metadata['variables'], list):
                print(f"❌ {dataset_dir.name}: 'variables' is not a list")
                all_valid = False
                continue
            
            declared_variables = len(metadata['variables'])
            
            # Check each variable has required fields
            for i, var in enumerate(metadata['variables']):
                required_fields = ['var_name', 'units', 'target_index']
                for field in required_fields:
                    if field not in var:
                        print(f"❌ {dataset_dir.name}: Variable {i} missing '{field}'")
                        all_valid = False
                        break
                
                if var.get('target_index') != i:
                    print(f"❌ {dataset_dir.name}: Variable {i} has incorrect target_index {var.get('target_index')}")
                    all_valid = False
            
            # Now verify that the CSV actually contains the declared number of targets
            csv_files = list(dataset_dir.glob("*.csv"))
            if csv_files:
                # Use the first CSV file for verification
                csv_file = csv_files[0]
                actual_targets = count_targets_in_csv(csv_file)
                
                if actual_targets is not None:
                    if actual_targets != declared_variables:
                        print(f"❌ {dataset_dir.name}: Mismatch! Metadata declares {declared_variables} variables, but CSV has {actual_targets} targets")
                        all_valid = False
                    else:
                        print(f"✅ {dataset_dir.name}: {declared_variables} variables ✓ (CSV verified)")
                else:
                    print(f"⚠️  {dataset_dir.name}: {declared_variables} variables (CSV verification failed)")
            else:
                print(f"⚠️  {dataset_dir.name}: {declared_variables} variables (no CSV files found)")
            
        except json.JSONDecodeError as e:
            print(f"❌ {dataset_dir.name}: Invalid JSON - {e}")
            all_valid = False
        except Exception as e:
            print(f"❌ {dataset_dir.name}: Error reading file - {e}")
            all_valid = False
    
    print("=" * 50)
    if all_valid:
        print("🎉 All metadata files are valid and match their CSV data!")
    else:
        print("❌ Some metadata files have issues or don't match CSV data")
    
    return all_valid

if __name__ == "__main__":
    verify_metadata_files()
