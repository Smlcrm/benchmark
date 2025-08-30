import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from pathlib import Path

def load_original_data():
    """Load the original patient_sparse_univariate.csv data"""
    csv_path = "benchmarking_pipeline/datasets/univariate/patient_sparse_univariate/patient_sparse_univariate.csv"
    df = pd.read_csv(csv_path)
    
    # Convert charttime to datetime
    df['charttime'] = pd.to_datetime(df['charttime'])
    
    # Sort by time
    df = df.sort_values('charttime')
    
    return df

def load_interpolated_data():
    """Load the interpolated chunk001.csv data"""
    csv_path = "benchmarking_pipeline/datasets/univariate/patient_sparse_univariate/chunk001.csv"
    df = pd.read_csv(csv_path)
    
    # Parse the target column (it's stored as a string representation of a list)
    target_str = df.iloc[0]['target']
    target_values = ast.literal_eval(target_str)
    
    # Get start time and frequency
    start_time = pd.to_datetime(df.iloc[0]['start'])
    freq = df.iloc[0]['freq']
    
    # Create regular time grid
    regular_times = pd.date_range(start=start_time, periods=len(target_values), freq=freq)
    
    # Create dataframe with interpolated data
    interpolated_df = pd.DataFrame({
        'charttime': regular_times,
        'valuenum': target_values
    })
    
    return interpolated_df

def plot_comparison(original_df, interpolated_df):
    """Plot both datasets for comparison"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Full timeline comparison
    plt.subplot(2, 1, 1)
    plt.plot(original_df['charttime'], original_df['valuenum'], 'o-', 
             label='Original Data', markersize=3, alpha=0.7, linewidth=1)
    plt.plot(interpolated_df['charttime'], interpolated_df['valuenum'], '-', 
             label='Interpolated Data', linewidth=1, alpha=0.8)
    plt.title('Patient Sparse Univariate: Original vs Interpolated Data (Full Timeline)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view of a specific time period
    plt.subplot(2, 1, 2)
    
    # Find a period with dense data for better visualization
    # Look for periods with frequent measurements
    time_diffs = original_df['charttime'].diff().dt.total_seconds()
    dense_periods = time_diffs < 3600  # Less than 1 hour between measurements
    
    if dense_periods.any():
        # Find the start of a dense period
        dense_start_idx = dense_periods.idxmax()
        dense_start_time = original_df.loc[dense_start_idx, 'charttime']
        
        # Show 24 hours around this period
        start_zoom = dense_start_time - pd.Timedelta(hours=12)
        end_zoom = dense_start_time + pd.Timedelta(hours=12)
        
        # Filter data for zoomed view
        original_zoom = original_df[(original_df['charttime'] >= start_zoom) & 
                                   (original_df['charttime'] <= end_zoom)]
        interpolated_zoom = interpolated_df[(interpolated_df['charttime'] >= start_zoom) & 
                                           (interpolated_df['charttime'] <= end_zoom)]
        
        plt.plot(original_zoom['charttime'], original_zoom['valuenum'], 'o-', 
                 label='Original Data', markersize=4, alpha=0.8, linewidth=1)
        plt.plot(interpolated_zoom['charttime'], interpolated_zoom['valuenum'], '-', 
                 label='Interpolated Data', linewidth=1.5, alpha=0.9)
        plt.title(f'Zoomed View: {start_zoom.strftime("%Y-%m-%d %H:%M")} to {end_zoom.strftime("%Y-%m-%d %H:%M")}')
    else:
        # If no dense periods, show first 24 hours
        start_zoom = original_df['charttime'].min()
        end_zoom = start_zoom + pd.Timedelta(hours=24)
        
        original_zoom = original_df[(original_df['charttime'] >= start_zoom) & 
                                   (original_df['charttime'] <= end_zoom)]
        interpolated_zoom = interpolated_df[(interpolated_df['charttime'] >= start_zoom) & 
                                           (interpolated_df['charttime'] <= end_zoom)]
        
        plt.plot(original_zoom['charttime'], original_zoom['valuenum'], 'o-', 
                 label='Original Data', markersize=4, alpha=0.8, linewidth=1)
        plt.plot(interpolated_zoom['charttime'], interpolated_zoom['valuenum'], '-', 
                 label='Interpolated Data', linewidth=1.5, alpha=0.9)
        plt.title(f'First 24 Hours: {start_zoom.strftime("%Y-%m-%d %H:%M")} to {end_zoom.strftime("%Y-%m-%d %H:%M")}')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def analyze_differences(original_df, interpolated_df):
    """Analyze the differences between original and interpolated data"""
    print("=== DATA ANALYSIS ===")
    print(f"Original data points: {len(original_df)}")
    print(f"Interpolated data points: {len(interpolated_df)}")
    print(f"Time span: {original_df['charttime'].min()} to {original_df['charttime'].max()}")
    print(f"Duration: {original_df['charttime'].max() - original_df['charttime'].min()}")
    
    # Find overlapping time points
    original_times = set(original_df['charttime'])
    interpolated_times = set(interpolated_df['charttime'])
    overlapping_times = original_times.intersection(interpolated_times)
    
    print(f"Overlapping time points: {len(overlapping_times)}")
    
    if overlapping_times:
        # Compare values at overlapping times
        overlapping_df = original_df[original_df['charttime'].isin(overlapping_times)].copy()
        overlapping_df = overlapping_df.merge(interpolated_df, on='charttime', suffixes=('_orig', '_interp'))
        
        # Calculate differences
        differences = overlapping_df['valuenum_orig'] - overlapping_df['valuenum_interp']
        
        print(f"\n=== COMPARISON AT OVERLAPPING TIMES ===")
        print(f"Mean absolute difference: {differences.abs().mean():.6f}")
        print(f"Max absolute difference: {differences.abs().max():.6f}")
        print(f"Standard deviation of differences: {differences.std():.6f}")
        print(f"Root mean square error: {np.sqrt((differences**2).mean()):.6f}")
        
        # Show some examples
        print(f"\n=== SAMPLE COMPARISONS ===")
        sample_comparisons = overlapping_df.head(10)[['charttime', 'valuenum_orig', 'valuenum_interp']]
        for _, row in sample_comparisons.iterrows():
            diff = row['valuenum_orig'] - row['valuenum_interp']
            print(f"{row['charttime']}: Original={row['valuenum_orig']:.2f}, "
                  f"Interpolated={row['valuenum_interp']:.2f}, Diff={diff:.6f}")
    
    # Analyze frequency
    print(f"\n=== FREQUENCY ANALYSIS ===")
    time_diffs_orig = original_df['charttime'].diff().dropna()
    time_diffs_interp = interpolated_df['charttime'].diff().dropna()
    
    print(f"Original data - Min interval: {time_diffs_orig.min()}")
    print(f"Original data - Max interval: {time_diffs_orig.max()}")
    print(f"Original data - Mean interval: {time_diffs_orig.mean()}")
    print(f"Interpolated data - Fixed interval: {time_diffs_interp.iloc[0]}")

def main():
    """Main function to run the validation"""
    print("Loading original data...")
    original_df = load_original_data()
    
    print("Loading interpolated data...")
    interpolated_df = load_interpolated_data()
    
    print("Analyzing differences...")
    analyze_differences(original_df, interpolated_df)
    
    print("Creating comparison plots...")
    plot_comparison(original_df, interpolated_df)
    
    print("Validation complete!")

if __name__ == "__main__":
    main()
