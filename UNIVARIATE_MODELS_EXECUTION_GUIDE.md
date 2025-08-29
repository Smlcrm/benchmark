# 🚀 Univariate Models Execution Guide

## 📋 Overview

This guide covers how to execute all 11 non-foundation univariate models in separate terminals. Each model has been configured with focused hyperparameters for testing.

## 🎯 Models Overview

### **Models with Training Loss** (ML Models)
1. **ARIMA** - Statistical time series model
2. **LSTM** - Neural network for sequence prediction  
3. **DeepAR** - Neural network with autoregressive structure

### **Models without Training Loss** (Statistical/Foundation Models)
4. **XGBoost** - Gradient boosting ensemble
5. **Theta** - Theta method for time series
6. **SVR** - Support Vector Regression
7. **Prophet** - Facebook's forecasting tool
8. **Random Forest** - Ensemble method
9. **Exponential Smoothing** - Statistical smoothing
10. **Seasonal Naive** - Simple baseline
11. **TabPFN** - Foundation model style

## 📁 Configuration Files

All config files are located in `benchmarking_pipeline/configs/`:

- `test_arima_univariate.yaml`
- `test_lstm_univariate.yaml`
- `test_xgboost_univariate.yaml`
- `test_theta_univariate.yaml`
- `test_svr_univariate.yaml`
- `test_prophet_univariate.yaml`
- `test_random_forest_univariate.yaml`
- `test_exponential_smoothing_univariate.yaml`
- `test_seasonal_naive_univariate.yaml`
- `test_deepar_univariate.yaml`
- `test_tabpfn_univariate.yaml`

## 🚀 Execution Methods

### **Method 1: Cursor Terminal (Recommended for Cursor Users)**

Since Cursor's integrated terminal doesn't support opening new terminal windows, use these Cursor-specific scripts:

#### **A. Run All Models in Background**
```bash
# Run all models simultaneously in the background
./run_all_univariate_models_cursor.sh
```

This will:
- Start all 11 models in the background
- Save output to individual log files in `logs/` directory
- Allow you to monitor progress using monitoring commands

#### **B. Monitor All Models**
```bash
# Check status of all running models
./monitor_models.sh
```

#### **C. Run Single Model Interactively**
```bash
# Run a specific model in the foreground
./run_single_model.sh arima
./run_single_model.sh lstm
./run_single_model.sh xgboost
# etc.
```

### **Method 2: Automated Script (External Terminals)**

```bash
# Run the automated script (opens external terminal windows)
./run_all_univariate_models.sh
```

This script will:
- Open separate terminal windows/tabs for each model
- Execute each model with its specific config
- Show progress and completion status
- Work on macOS and Linux (but not in Cursor)

### **Method 3: Manual Execution**

Open separate terminal windows/tabs and run each model manually:

#### **Terminal 1 - ARIMA**
```bash
cd /Users/gandresr/Documents/GitHub/simulacrum/benchmark
python -m benchmarking_pipeline.run_benchmark benchmarking_pipeline/configs/test_arima_univariate.yaml
```

#### **Terminal 2 - LSTM**
```bash
cd /Users/gandresr/Documents/GitHub/simulacrum/benchmark
python -m benchmarking_pipeline.run_benchmark benchmarking_pipeline/configs/test_lstm_univariate.yaml
```

#### **Terminal 3 - XGBoost**
```bash
cd /Users/gandresr/Documents/GitHub/simulacrum/benchmark
python -m benchmarking_pipeline.run_benchmark benchmarking_pipeline/configs/test_xgboost_univariate.yaml
```

#### **Terminal 4 - Theta**
```bash
cd /Users/gandresr/Documents/GitHub/simulacrum/benchmark
python -m benchmarking_pipeline.run_benchmark benchmarking_pipeline/configs/test_theta_univariate.yaml
```

#### **Terminal 5 - SVR**
```bash
cd /Users/gandresr/Documents/GitHub/simulacrum/benchmark
python -m benchmarking_pipeline.run_benchmark benchmarking_pipeline/configs/test_svr_univariate.yaml
```

#### **Terminal 6 - Prophet**
```bash
cd /Users/gandresr/Documents/GitHub/simulacrum/benchmark
python -m benchmarking_pipeline.run_benchmark benchmarking_pipeline/configs/test_prophet_univariate.yaml
```

#### **Terminal 7 - Random Forest**
```bash
cd /Users/gandresr/Documents/GitHub/simulacrum/benchmark
python -m benchmarking_pipeline.run_benchmark benchmarking_pipeline/configs/test_random_forest_univariate.yaml
```

#### **Terminal 8 - Exponential Smoothing**
```bash
cd /Users/gandresr/Documents/GitHub/simulacrum/benchmark
python -m benchmarking_pipeline.run_benchmark benchmarking_pipeline/configs/test_exponential_smoothing_univariate.yaml
```

#### **Terminal 9 - Seasonal Naive**
```bash
cd /Users/gandresr/Documents/GitHub/simulacrum/benchmark
python -m benchmarking_pipeline.run_benchmark benchmarking_pipeline/configs/test_seasonal_naive_univariate.yaml
```

#### **Terminal 10 - DeepAR**
```bash
cd /Users/gandresr/Documents/GitHub/simulacrum/benchmark
python -m benchmarking_pipeline.run_benchmark benchmarking_pipeline/configs/test_deepar_univariate.yaml
```

#### **Terminal 11 - TabPFN**
```bash
cd /Users/gandresr/Documents/GitHub/simulacrum/benchmark
python -m benchmarking_pipeline.run_benchmark benchmarking_pipeline/configs/test_tabpfn_univariate.yaml
```

## 📊 Dataset Configuration

All models use the same dataset configuration:
- **Dataset**: `australian_electricity_demand`
- **Frequency**: Daily (`D`)
- **Forecast Horizon**: 10 steps
- **Split Ratio**: 80% train, 10% validation, 10% test
- **Chunks**: 1 (for faster testing)

## ⚙️ Model-Specific Configurations

### **ARIMA**
- **Parameters**: p=[1,2], d=[0,1], q=[1,2]
- **Training Loss**: MAE
- **Normalization**: false

### **LSTM**
- **Parameters**: units=[32,64], layers=[1,2], dropout=[0.1,0.2]
- **Training Loss**: MAE
- **Normalization**: true

### **XGBoost**
- **Parameters**: lookback_window=[10,20], n_estimators=[50,100]
- **Training Loss**: None (not applicable)
- **Normalization**: false

### **Theta**
- **Parameters**: sp=[1,7]
- **Training Loss**: None (not applicable)
- **Normalization**: false

### **SVR**
- **Parameters**: kernel=[rbf,linear], C=[0.1,1.0,10.0]
- **Training Loss**: None (not applicable)
- **Normalization**: true

### **Prophet**
- **Parameters**: seasonality_mode=[additive,multiplicative]
- **Training Loss**: None (not applicable)
- **Normalization**: false

### **Random Forest**
- **Parameters**: lookback_window=[10,20], n_estimators=[50,100]
- **Training Loss**: None (not applicable)
- **Normalization**: false

### **Exponential Smoothing**
- **Parameters**: trend=[none,add], seasonal=[none,add,mul]
- **Training Loss**: None (not applicable)
- **Normalization**: false

### **Seasonal Naive**
- **Parameters**: sp=[7,14]
- **Training Loss**: None (not applicable)
- **Normalization**: false

### **DeepAR**
- **Parameters**: context_length=[20,30], epochs=[5,10]
- **Training Loss**: MAE
- **Normalization**: true

### **TabPFN**
- **Parameters**: allow_large_cpu_dataset=[true], max_sequence_length=[32,64]
- **Training Loss**: None (not applicable)
- **Normalization**: false

## 🎯 Expected Results

Each model will:
1. **Load** the australian_electricity_demand dataset
2. **Train** using the specified hyperparameters
3. **Evaluate** using MAE and RMSE metrics
4. **Save** results and plots
5. **Display** completion status

## ⚠️ Important Notes

- **Training Loss Models**: ARIMA, LSTM, DeepAR use training loss for optimization
- **Statistical Models**: Theta, Prophet, Exponential Smoothing, Seasonal Naive don't use training loss
- **ML Models**: XGBoost, SVR, Random Forest, TabPFN don't use training loss
- **Normalization**: Some models require normalized data, others work with raw data
- **Execution Time**: Models will run at different speeds (LSTM/DeepAR slower, statistical models faster)

## 🛠️ Troubleshooting

### **If a model fails:**
1. Check the terminal output for error messages
2. Verify the conda environment is activated: `conda activate sim.benchmarks`
3. Check if all dependencies are installed
4. Ensure the dataset path is correct

### **If terminals don't open (Cursor users):**
1. Use the Cursor-specific scripts: `./run_all_univariate_models_cursor.sh`
2. Run models in background and monitor with `./monitor_models.sh`
3. Use `./run_single_model.sh MODEL_NAME` for individual models
4. Check background jobs with `jobs` command

## 🎉 Success Indicators

- Each model completes without errors
- Results are saved to the expected directories
- Plots and metrics are generated
- All terminals show completion messages

## 🔍 Cursor-Specific Monitoring

### **Background Job Management:**
```bash
jobs                    # Show all running background jobs
fg %N                  # Bring job N to foreground (e.g., 'fg %1')
bg %N                  # Send job N to background
kill %N                # Stop job N
```

### **Real-time Monitoring:**
```bash
tail -f logs/MODEL_output.log    # Watch specific model in real-time
./monitor_models.sh              # Check status of all models
```

### **Log Files:**
All model outputs are saved in the `logs/` directory:
- `ARIMA_output.log`
- `LSTM_output.log`
- `XGBoost_output.log`
- etc.

---

**For Cursor Users: Execute: `./run_all_univariate_models_cursor.sh`** 🚀

**For External Terminals: Execute: `./run_all_univariate_models.sh`** 🖥️
