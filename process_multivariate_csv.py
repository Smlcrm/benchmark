import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

def get_file_size_mb(file_path):
    """Get file size in MB"""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)

def detect_frequency(datetime_series):
    """
    Detect the frequency of a datetime series using pandas infer_freq.
    Returns (is_regular, frequency_string, min_timedelta)
    """
    if len(datetime_series) < 2:
        raise ValueError("Need at least 2 datetime points for frequency analysis")
    
    # Sort and remove duplicates
    sorted_dates = sorted(datetime_series.dropna().unique())
    if len(sorted_dates) < 2:
        raise ValueError("No valid datetime points after removing NaNs")
    
    # Convert to pandas DatetimeIndex
    dt_index = pd.DatetimeIndex(sorted_dates)
    
    # Try to infer frequency using pandas
    inferred_freq = pd.infer_freq(dt_index)
    if inferred_freq is not None:
        # Pandas successfully inferred a regular frequency
        return True, inferred_freq, None
    
    # If pandas couldn't infer frequency, calculate manually
    time_diffs = []
    for i in range(1, len(sorted_dates)):
        diff = sorted_dates[i] - sorted_dates[i-1]
        time_diffs.append(diff)
    
    if not time_diffs:
        raise ValueError("No time differences found for frequency analysis")
    
    min_diff = min(time_diffs)
    max_diff = max(time_diffs)
    
    # Check if regular (all differences are approximately the same)
    tolerance = 0.1  # 10% tolerance
    is_regular = all(abs(diff - min_diff) <= tolerance * min_diff for diff in time_diffs)
    
    # Convert timedelta to frequency string
    freq_str = timedelta_to_freq(min_diff)
    
    return is_regular, freq_str, min_diff

def timedelta_to_freq(td):
    """Convert timedelta to pandas frequency string"""
    # Handle both Python timedelta and numpy timedelta64
    if hasattr(td, 'total_seconds'):
        total_seconds = td.total_seconds()
    else:
        # numpy timedelta64
        total_seconds = td / np.timedelta64(1, 's')
    
    if total_seconds < 60:  # Less than 1 minute
        return f"{int(total_seconds)}S"
    elif total_seconds < 3600:  # Less than 1 hour
        minutes = total_seconds / 60
        if minutes.is_integer():
            return f"{int(minutes)}T"
        else:
            return f"{int(total_seconds)}S"
    elif total_seconds < 86400:  # Less than 1 day
        hours = total_seconds / 3600
        if hours.is_integer():
            return f"{int(hours)}H"
        else:
            return f"{int(total_seconds)}S"
    elif total_seconds < 2592000:  # Less than 30 days
        days = total_seconds / 86400
        if days.is_integer():
            return f"{int(days)}D"
        else:
            return "D"
    else:
        return "D"

def regularize_timeseries_efficient(df, datetime_col, min_freq):
    """
    Efficiently regularize irregular timeseries using proper linear interpolation.
    Creates a regular grid with the minimum frequency and interpolates values.
    """
    if len(df) < 2:
        raise ValueError("Need at least 2 data points for interpolation")
    
    # Get start and end times
    start_time = df[datetime_col].min()
    end_time = df[datetime_col].max()
    
    # Find all data columns (exclude datetime column)
    data_cols = [col for col in df.columns if col != datetime_col]
    if not data_cols:
        raise ValueError("No data columns found for interpolation")
    
    # Calculate number of steps needed
    if hasattr(min_freq, 'total_seconds'):
        min_freq_seconds = min_freq.total_seconds()
    else:
        # numpy timedelta64
        min_freq_seconds = min_freq / np.timedelta64(1, 's')
    
    # Create regular time grid
    time_grid = pd.date_range(start=start_time, end=end_time, freq=f"{min_freq_seconds}S")
    
    # Limit number of steps for performance
    max_steps = 100000
    if len(time_grid) > max_steps:
        # Adjust frequency to stay within max_steps
        total_seconds = (end_time - start_time).total_seconds()
        new_freq_seconds = total_seconds / max_steps
        
        # Round to nearest reasonable frequency
        if new_freq_seconds < 60:
            new_freq_seconds = 60  # 1 minute minimum
        elif new_freq_seconds < 3600:
            new_freq_seconds = round(new_freq_seconds / 60) * 60  # Round to nearest minute
        elif new_freq_seconds < 86400:
            new_freq_seconds = round(new_freq_seconds / 3600) * 3600  # Round to nearest hour
        else:
            new_freq_seconds = 86400  # 1 day
        
        time_grid = pd.date_range(start=start_time, end=end_time, freq=f"{new_freq_seconds}S")
    
    # Create new dataframe with regular time grid
    new_df = pd.DataFrame({datetime_col: time_grid})
    
    # Interpolate each data column
    for col in data_cols:
        # Remove NaN values for interpolation
        valid_data = df[[datetime_col, col]].dropna()
        if len(valid_data) >= 2:
            # Use scipy interpolation for efficiency
            f = interp1d(valid_data[datetime_col].astype(np.int64), 
                         valid_data[col], 
                         kind='linear', 
                         bounds_error=False, 
                         fill_value='extrapolate')
            
            # Interpolate values
            new_df[col] = f(new_df[datetime_col].astype(np.int64))
        else:
            # If not enough valid data, fill with NaN
            new_df[col] = np.nan
    
    return new_df

def process_csv_file(csv_path):
    """Process a single CSV file and return chunk data"""
    try:
        print(f"Processing: {csv_path}")
        
        # Check file size
        file_size_mb = get_file_size_mb(csv_path)
        if file_size_mb > 150:
            print(f"Skipping {csv_path} - file size {file_size_mb:.1f}MB exceeds 150MB limit")
            return None
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            print(f"Empty CSV file: {csv_path}")
            return None
        
        # Get item_id from filename
        item_id = os.path.splitext(os.path.basename(csv_path))[0]
        
        # Assume first column is always the date column
        datetime_col = df.columns[0]
        
        # Convert first column to datetime
        try:
            # Check if it's numeric (timesteps) or datetime
            if df[datetime_col].dtype in [np.number]:
                # Convert numeric timesteps to datetime starting from 2020-01-01
                base_date = pd.Timestamp('2020-01-01')
                df[datetime_col] = base_date + pd.to_timedelta(df[datetime_col], unit='s')
            else:
                # Convert to datetime
                df[datetime_col] = pd.to_datetime(df[datetime_col])
        except Exception as e:
            print(f"Error converting datetime column {datetime_col}: {e}")
            return None
        
        # Get start time
        start_time = df[datetime_col].min()
        start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Detect frequency and regularity using pandas
        is_regular, freq_str, min_freq = detect_frequency(df[datetime_col])
        
        # Handle synthetic datasets
        if 'synthetic_' in item_id.lower():
            if is_regular:
                freq_str = 'D'
            else:
                # Regularize with interpolation
                df = regularize_timeseries_efficient(df, datetime_col, min_freq)
                freq_str = 'D'
        else:
            # Non-synthetic datasets
            if not is_regular:
                # Regularize with interpolation using minimum frequency
                df = regularize_timeseries_efficient(df, datetime_col, min_freq)
                freq_str = timedelta_to_freq(min_freq) if min_freq else 'D'
            # If regular and non-synthetic, freq_str is already inferred by pandas
        
        # Prepare target data
        # Assume first column is always date, so data columns start from index 1
        data_cols = df.columns[1:].tolist()
        
        # Filter to only numeric data columns
        numeric_data_cols = [col for col in data_cols if df[col].dtype in [np.number]]
        
        if len(numeric_data_cols) == 1:
            # Univariate
            target = df[numeric_data_cols[0]].tolist()
        else:
            # Multivariate - create list of lists
            target = df[numeric_data_cols].values.tolist()
        
        # Ensure target data is not empty
        if not target or (isinstance(target, list) and len(target) == 0):
            print(f"Warning: Empty target data for {csv_path}")
            return None
        
        return {
            'item_id': item_id,
            'start': start_str,
            'freq': freq_str,
            'target': target
        }
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to process all CSV files in multivariate datasets"""
    # Base directory
    base_dir = "benchmarking_pipeline/datasets/multivariate"
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist")
        return
    
    # Find all CSV files recursively
    csv_files = glob.glob(os.path.join(base_dir, "**/*.csv"), recursive=True)
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each subfolder
    processed_folders = set()
    
    for csv_file in csv_files:
        folder_path = os.path.dirname(csv_file)
        
        if folder_path not in processed_folders:
            processed_folders.add(folder_path)
            print(f"\nProcessing folder: {folder_path}")
            
            # Get all CSV files in this folder, excluding existing chunk001.csv
            folder_csv_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) 
                              if not f.endswith('chunk001.csv')]
            
            if not folder_csv_files:
                print(f"No source CSV files found in {folder_path} (only chunk001.csv exists)")
                continue
            
            # Process only the first CSV file in the folder (main dataset)
            main_csv = folder_csv_files[0]  # Take the first CSV file
            chunk_data = process_csv_file(main_csv)
            
            if chunk_data and chunk_data['target']:  # Ensure target data is not empty
                # Create chunk001.csv for this folder with single row
                chunk_df = pd.DataFrame([chunk_data])  # Single row
                chunk_output_path = os.path.join(folder_path, "chunk001.csv")
                
                # Save with proper formatting
                chunk_df.to_csv(chunk_output_path, index=False)
                print(f"Created {chunk_output_path} with 1 row")
            else:
                print(f"No valid data found in {folder_path}")

if __name__ == "__main__":
    main()
