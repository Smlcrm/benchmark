import os
import shutil
from pathlib import Path

def organize_multivariate_csvs():
    """Organize CSV files in datasets/multivariate by moving them into named folders"""
    
    # Base directory
    base_dir = "benchmarking_pipeline/datasets/multivariate"
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return
    
    # Find all CSV files in the multivariate directory
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {base_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to organize")
    
    for csv_file in csv_files:
        csv_path = os.path.join(base_dir, csv_file)
        
        # Get folder name (remove .csv extension)
        folder_name = csv_file.replace('.csv', '')
        folder_path = os.path.join(base_dir, folder_name)
        
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_name}")
        
        # Move CSV file into the folder
        new_csv_path = os.path.join(folder_path, csv_file)
        
        if os.path.exists(new_csv_path):
            print(f"File already exists in {folder_name}, skipping: {csv_file}")
            continue
        
        try:
            shutil.move(csv_path, new_csv_path)
            print(f"Moved {csv_file} -> {folder_name}/")
        except Exception as e:
            print(f"Error moving {csv_file}: {e}")
    
    print("\nOrganization complete!")
    
    # List the final structure
    print("\nFinal directory structure:")
    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            csv_count = len([f for f in os.listdir(item_path) if f.endswith('.csv')])
            print(f"  {item}/ ({csv_count} CSV files)")

if __name__ == "__main__":
    organize_multivariate_csvs()
