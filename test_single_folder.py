import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta

def detect_frequency(datetime_series):
    """
    Detect the frequency of a datetime series using pandas infer_freq.
    Returns (is_regular, frequency_string, min_timedelta)
    """
    if len(datetime_series) < 2:
        return False, "D", timedelta(days=1)
    
    # Sort and remove duplicates
    sorted_dates = sorted(datetime_series.dropna().unique())
    if len(sorted_dates) < 2:
        return False, "D", timedelta(days=1)
    
    # Convert to pandas DatetimeIndex
    dt_index = pd.DatetimeIndex(sorted_dates)
    
    # Try to infer frequency using pandas
    try:
        inferred_freq = pd.infer_freq(dt_index)
        if inferred_freq is not None:
            # Pandas successfully inferred a regular frequency
            return True, inferred_freq, None
    except Exception as e:
        pass
    
    # If pandas couldn't infer frequency, calculate manually
    time_diffs = []
    for i in range(1, len(sorted_dates)):
        diff = sorted_dates[i] - sorted_dates[i-1]
        time_diffs.append(diff)
    
    if not time_diffs:
        return False, "D", timedelta(days=1)
    
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

def process_csv_file(csv_path):
    """Process a single CSV file and return the chunk data"""
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return None
        
        # Get item_id from filename
        item_id = Path(csv_path).stem
        
        # Find datetime column
        datetime_col = None
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].iloc[0])
                    datetime_col = col
                    break
                except:
                    continue
        
        if datetime_col is None:
            # Check for numeric time columns (timesteps)
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # This might be a timestep column
                    if col.lower() in ['time', 'timestep', 'step', 'index']:
                        datetime_col = col
                        break
        
        if datetime_col is None:
            # No datetime column found, treat as synthetic with timesteps
            df['datetime'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(df.iloc[:, 0], unit='D')
            datetime_col = 'datetime'
        elif datetime_col == 'time' and df[datetime_col].dtype in ['int64', 'float64']:
            # This is a timestep column, convert to proper datetime
            df['datetime'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(df[datetime_col], unit='D')
            datetime_col = 'datetime'
        
        # Convert datetime column
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
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
                freq_str = 'D'
        else:
            # Non-synthetic datasets
            if not is_regular:
                freq_str = timedelta_to_freq(min_freq) if min_freq else 'D'
        
        # Prepare target data
        # Remove datetime column and any other non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove the datetime column we created if it exists
        if 'datetime' in numeric_cols:
            numeric_cols.remove('datetime')
        # Remove the original time column if it exists (since we converted it to datetime)
        if 'time' in numeric_cols:
            numeric_cols.remove('time')
        
        if len(numeric_cols) == 1:
            # Univariate
            target = df[numeric_cols[0]].tolist()
        else:
            # Multivariate - create list of lists
            target = df[numeric_cols].values.tolist()
        
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
        return None

def main():
    """Test with a single folder"""
    folder_path = "benchmarking_pipeline/datasets/univariate/synthetic_additive2_univariate"
    
    print(f"Testing folder: {folder_path}")
    
    # Get all CSV files in this folder, excluding existing chunk001.csv
    folder_csv_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) 
                      if not f.endswith('chunk001.csv')]
    
    print(f"Found {len(folder_csv_files)} source CSV files: {[os.path.basename(f) for f in folder_csv_files]}")
    
    if not folder_csv_files:
        print("No source CSV files found")
        return
    
    # Process only the first CSV file in the folder (main dataset)
    main_csv = folder_csv_files[0]
    print(f"Processing: {os.path.basename(main_csv)}")
    
    chunk_data = process_csv_file(main_csv)
    
    if chunk_data and chunk_data['target']:
        # Create chunk001.csv for this folder with single row
        chunk_df = pd.DataFrame([chunk_data])  # Single row
        chunk_output_path = os.path.join(folder_path, "chunk001.csv")
        
        # Save with proper formatting
        chunk_df.to_csv(chunk_output_path, index=False)
        print(f"Created {chunk_output_path} with 1 row")
        
        # Verify the output
        print(f"\nVerifying output:")
        print(f"  item_id: {chunk_data['item_id']}")
        print(f"  start: {chunk_data['start']}")
        print(f"  freq: {chunk_data['freq']}")
        print(f"  target: {len(chunk_data['target'])} values (first 5: {chunk_data['target'][:5]})")
    else:
        print("No valid data found")

if __name__ == "__main__":
    main()
