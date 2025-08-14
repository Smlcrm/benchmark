#!/usr/bin/env python3
"""
Test script for testing multiple config files with verbose options and TensorBoard integration.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add the benchmarking_pipeline to path
sys.path.append('benchmarking_pipeline')

def test_config(config_path, verbose=True, tensorboard=True):
    """
    Test a specific config file.
    
    Args:
        config_path (str): Path to the config file
        verbose (bool): Enable verbose logging
        tensorboard (bool): Enable TensorBoard logging
    """
    print(f"\n{'='*60}")
    print(f"🧪 Testing config: {config_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Build command
    cmd = [
        'python', 'benchmarking_pipeline/run_benchmark.py',
        '--config', config_path
    ]
    
    if verbose:
        cmd.append('--verbose')
    
    if tensorboard:
        cmd.append('--tensorboard')
        cmd.extend(['--log-dir', 'logs/tensorboard'])
        cmd.extend(['--run-name', f'test_{Path(config_path).stem}'])
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    print(f"⏰ Starting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run the benchmark
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True)
        end_time = time.time()
        
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Config {config_path} completed successfully in {duration:.1f}s")
            return True
        else:
            print(f"❌ Config {config_path} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running config {config_path}: {str(e)}")
        return False

def main():
    """Main function to test multiple config files."""
    
    # Config files to test
    configs_to_test = [
        'benchmarking_pipeline/configs/croston_test.yaml',
        'benchmarking_pipeline/configs/default_config.yaml',
        'benchmarking_pipeline/configs/deterministic_config.yaml',
        'benchmarking_pipeline/configs/multivariate_forecast_config_xgboost.yaml',
        'benchmarking_pipeline/configs/multivariate_forecast_horizon_config.yaml',
        'benchmarking_pipeline/configs/multivariate_forecast_horizon_config_svr.yaml',
        'benchmarking_pipeline/configs/multivariate_forecast_horizon_config_theta.yaml'
    ]
    
    print("🚀 Starting config file testing")
    print(f"📊 Testing {len(configs_to_test)} config files")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Test results
    results = {}
    successful = 0
    failed = 0
    
    for config_path in configs_to_test:
        success = test_config(config_path, verbose=True, tensorboard=True)
        results[config_path] = success
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Small delay between tests
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TESTING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(configs_to_test)}")
    
    print(f"\n📋 DETAILED RESULTS:")
    for config_path, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {config_path}")
    
    if failed == 0:
        print(f"\n🎉 All config files passed successfully!")
    else:
        print(f"\n⚠️  {failed} config file(s) failed. Check the logs above for details.")
    
    print(f"\n📈 TensorBoard logs are available in 'logs/tensorboard/'")
    print(f"🌐 To view logs, run: tensorboard --logdir logs/tensorboard")

if __name__ == "__main__":
    main() 