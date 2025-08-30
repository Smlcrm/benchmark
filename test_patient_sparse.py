import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

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

def regularize_timeseries_efficient(df, datetime_col, min_freq):
    """
    Efficiently regularize irregular timeseries using scipy interpolation.
    Creates a regular grid with the minimum frequency and interpolates values.
    """
    if len(df) < 2:
        return df
    
    # Get start and end times
    start_time = df[datetime_col].min()
    end_time = df[datetime_col].max()
    
    print(f"  Start time: {start_time}")
    print(f"  End time: {end_time}")
    print(f"  Time span: {end_time - start_time}")
    print(f"  Min frequency: {min_freq}")
    
    # Convert to numeric timestamps for interpolation
    timestamps = pd.to_numeric(df[datetime_col])
    values = df.iloc[:, 1].values  # Assuming second column is the value
    
    # Create regular time grid efficiently
    # Limit the number of points to prevent memory issues
    max_points = 10000  # Reasonable limit
    
    # Calculate number of steps needed
    total_seconds = (end_time - start_time).total_seconds()
    
    # Handle both Python timedelta and numpy timedelta64
    if hasattr(min_freq, 'total_seconds'):
        min_freq_seconds = min_freq.total_seconds()
    else:
        # numpy timedelta64
        min_freq_seconds = min_freq / np.timedelta64(1, 's')
    
    if min_freq_seconds == 0:
        min_freq_seconds = 3600  # Default to 1 hour if min_freq is 0
    
    num_steps = int(total_seconds / min_freq_seconds)
    print(f"  Would need {num_steps} points with min frequency")
    
    # If too many points, adjust frequency to stay within limits
    if num_steps > max_points:
        adjusted_freq_seconds = total_seconds / max_points
        # Round to nearest reasonable frequency
        if adjusted_freq_seconds < 60:  # Less than 1 minute
            adjusted_freq_seconds = 60
        elif adjusted_freq_seconds < 3600:  # Less than 1 hour
            adjusted_freq_seconds = 3600
        else:
            adjusted_freq_seconds = 86400  # 1 day
        
        min_freq = pd.Timedelta(seconds=adjusted_freq_seconds)
        print(f"  Adjusted frequency from {timedelta_to_freq(pd.Timedelta(seconds=min_freq_seconds))} to {timedelta_to_freq(min_freq)} to stay within {max_points} points")
    
    # Create regular time grid
    regular_times = pd.date_range(start=start_time, end=end_time, freq=min_freq)
    print(f"  Created regular grid with {len(regular_times)} points")
    
    # Use scipy interpolation for efficiency
    try:
        # Linear interpolation
        interpolator = interp1d(timestamps, values, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
        
        # Interpolate at regular grid points
        regular_timestamps = pd.to_numeric(regular_times)
        interpolated_values = interpolator(regular_timestamps)
        
        # Create new dataframe
        regular_df = pd.DataFrame({
            datetime_col: regular_times,
            'value': interpolated_values
        })
        
        print(f"  Successfully interpolated to {len(regular_df)} points")
        return regular_df
        
    except Exception as e:
        print(f"  Interpolation failed: {e}, using forward fill method")
        # Fallback: use pandas merge and forward fill
        regular_df = pd.DataFrame({datetime_col: regular_times})
        merged_df = pd.merge(regular_df, df, on=datetime_col, how='left')
        merged_df = merged_df.sort_values(datetime_col)
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
        return merged_df

def process_csv_file(csv_path):
    """Process a single CSV file and return the chunk data"""
    try:
        print(f"Processing: {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        
        if df.empty:
            return None
        
        # Get item_id from filename
        item_id = Path(csv_path).stem
        print(f"  Item ID: {item_id}")
        
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
        
        print(f"  Using datetime column: {datetime_col}")
        
        # Convert datetime column
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Get start time
        start_time = df[datetime_col].min()
        start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  Start time: {start_str}")
        
        # Detect frequency and regularity using pandas
        print("  Detecting frequency...")
        is_regular, freq_str, min_freq = detect_frequency(df[datetime_col])
        print(f"  Frequency detection result: is_regular={is_regular}, freq={freq_str}")
        
        # Handle synthetic datasets
        if 'synthetic_' in item_id.lower():
            print("  This is a synthetic dataset")
            if is_regular:
                freq_str = 'D'
                print(f"  Regular timesteps detected, using freq='D'")
            else:
                print(f"  Irregular timesteps detected, regularizing and using freq='D'")
                df = regularize_timeseries_efficient(df, datetime_col, min_freq)
                freq_str = 'D'
        else:
            print("  This is not a synthetic dataset")
            if not is_regular:
                print(f"  Irregular timesteps detected, regularizing using min_freq")
                df = regularize_timeseries_efficient(df, datetime_col, min_freq)
                freq_str = timedelta_to_freq(min_freq) if min_freq else 'D'
            else:
                print(f"  Regular timesteps detected, using inferred frequency")
        
        # Prepare target data
        # Remove datetime column and any other non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove the datetime column we created if it exists
        if 'datetime' in numeric_cols:
            numeric_cols.remove('datetime')
        # Remove the original time column if it exists (since we converted it to datetime)
        if 'time' in numeric_cols:
            numeric_cols.remove('time')
        
        print(f"  Numeric columns for target: {numeric_cols}")
        
        if len(numeric_cols) == 1:
            # Univariate
            target = df[numeric_cols[0]].tolist()
            print(f"  Univariate target with {len(target)} values")
        else:
            # Multivariate - create list of lists
            target = df[numeric_cols].values.tolist()
            print(f"  Multivariate target with {len(target)} rows and {len(numeric_cols)} columns")
        
        # Ensure target data is not empty
        if not target or (isinstance(target, list) and len(target) == 0):
            print(f"  Warning: Empty target data")
            return None
        
        result = {
            'item_id': item_id,
            'start': start_str,
            'freq': freq_str,
            'target': target
        }
        
        print(f"  Final result:")
        print(f"    item_id: {result['item_id']}")
        print(f"    start: {result['start']}")
        print(f"    freq: {result['freq']}")
        print(f"    target: {len(result['target'])} values")
        
        return result
        
    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Test with the patient_sparse_univariate dataset"""
    folder_path = "benchmarking_pipeline/datasets/univariate/patient_sparse_univariate"
    
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
