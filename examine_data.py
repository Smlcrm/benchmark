import ast
import pandas as pd
import numpy as np

def examine_dataset(dataset_name):
    """Examine a specific dataset to understand its structure"""
    print(f"\n{'='*60}")
    print(f"Examining dataset: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Load the data
        df = pd.read_csv(f'benchmarking_pipeline/datasets/{dataset_name}/chunk001.csv')
        target = ast.literal_eval(df.iloc[0]['target'])
        
        print(f"Target data type: {type(target)}")
        
        if isinstance(target, list):
            print(f"Number of series: {len(target)}")
            
            if len(target) > 0:
                first_item = target[0]
                print(f"First item type: {type(first_item)}")
                
                if isinstance(first_item, list):
                    print(f"Length of first series: {len(first_item)}")
                    
                    # For multivariate datasets, let's look at the actual data to understand what each series represents
                    if len(target) > 1:
                        print("\nExamining first few values of each series:")
                        for i in range(min(3, len(target))):  # Look at first 3 series
                            series = target[i]
                            clean_series = [x for x in series[:10] if x is not None]  # First 10 non-None values
                            if clean_series:
                                mean_val = sum(clean_series) / len(clean_series)
                                range_val = max(clean_series) - min(clean_series)
                                print(f"  Series {i}: mean={mean_val:.2f}, range={range_val:.2f}, sample_values={clean_series[:3]}")
                            else:
                                print(f"  Series {i}: all None values in first 10")
                        
                        # Try to infer what the series might represent based on data characteristics
                        if dataset_name == "BEIJING_SUBWAY_30MIN":
                            print("\n→ Based on name and data: Likely passenger counts from different subway stations")
                        elif dataset_name == "china_air_quality":
                            print("\n→ Based on name and data: Air quality metrics (PM2.5, PM10, O3, NO2, CO, SO2)")
                        else:
                            print("\n→ Need to examine data characteristics to determine variable types")
                    else:
                        print("→ Single series (univariate)")
                else:
                    print(f"→ Single value series (univariate)")
            else:
                print("→ Empty target list")
        else:
            print(f"→ Target is not a list: {target}")
            
    except Exception as e:
        print(f"Error examining {dataset_name}: {e}")

# Examine datasets that are likely to have meaningful variable names
datasets_to_examine = [
    'china_air_quality',
    'BEIJING_SUBWAY_30MIN', 
    'temperature_rain_with_missing',
    'bitcoin_with_missing',
    'traffic_hourly',
    'solar_power',
    'wind_power'
]

for dataset in datasets_to_examine:
    examine_dataset(dataset)
