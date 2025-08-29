#!/bin/bash

# Script to run all non-foundation univariate models in Cursor terminal
# This script runs models in the background and provides monitoring

echo "🚀 Starting all non-foundation univariate models in Cursor terminal..."
echo "📊 Total models to run: 11"
echo "💡 Models will run in the background. Use 'jobs' to see running models."
echo ""

# Function to run a model in the background
run_model() {
    local model_name=$1
    local config_file=$2
    local description=$3
    
    echo "🔄 Starting $model_name in background..."
    
    # Run the model in the background
    python benchmarking_pipeline/run_benchmark.py --config "$config_file" > "logs/${model_name}_output.log" 2>&1 &
    
    # Store the job ID
    local job_id=$!
    echo "✅ $model_name started with job ID: $job_id"
    echo "📝 Output will be saved to: logs/${model_name}_output.log"
    echo ""
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Clear any existing background jobs
echo "🧹 Clearing any existing background jobs..."
jobs

echo ""
echo "📋 Starting models in background..."
echo "=" * 60

# Run each model in the background
echo "1️⃣  Starting ARIMA..."
run_model "ARIMA" "benchmarking_pipeline/configs/test_arima_univariate.yaml" "Statistical model with training loss"

echo "2️⃣  Starting LSTM..."
run_model "LSTM" "benchmarking_pipeline/configs/test_lstm_univariate.yaml" "Neural network with training loss"

echo "3️⃣  Starting XGBoost..."
run_model "XGBoost" "benchmarking_pipeline/configs/test_xgboost_univariate.yaml" "Gradient boosting (no training loss)"

echo "4️⃣  Starting Theta..."
run_model "Theta" "benchmarking_pipeline/configs/test_theta_univariate.yaml" "Statistical model (no training loss)"

echo "5️⃣  Starting SVR..."
run_model "SVR" "benchmarking_pipeline/configs/test_svr_univariate.yaml" "Support Vector Regression (no training loss)"

echo "6️⃣  Starting Prophet..."
run_model "Prophet" "benchmarking_pipeline/configs/test_prophet_univariate.yaml" "Facebook's forecasting (no training loss)"

echo "7️⃣  Starting Random Forest..."
run_model "RandomForest" "benchmarking_pipeline/configs/test_random_forest_univariate.yaml" "Ensemble method (no training loss)"

echo "8️⃣  Starting Exponential Smoothing..."
run_model "ExponentialSmoothing" "benchmarking_pipeline/configs/test_exponential_smoothing_univariate.yaml" "Statistical smoothing (no training loss)"

echo "9️⃣  Starting Seasonal Naive..."
run_model "SeasonalNaive" "benchmarking_pipeline/configs/test_seasonal_naive_univariate.yaml" "Simple baseline (no training loss)"

echo "🔟 Starting DeepAR..."
run_model "DeepAR" "benchmarking_pipeline/configs/test_deepar_univariate.yaml" "Neural network with training loss"

echo "1️⃣1️⃣ Starting TabPFN..."
run_model "TabPFN" "benchmarking_pipeline/configs/test_tabpfn_univariate.yaml" "Foundation model style (no training loss)"

echo ""
echo "🎉 All models started in background!"
echo "=" * 60
echo ""
echo "📊 MONITORING COMMANDS:"
echo "   • 'jobs'                    - Show all running background jobs"
echo "   • 'fg %N'                   - Bring job N to foreground (e.g., 'fg %1')"
echo "   • 'bg %N'                   - Send job N to background"
echo "   • 'kill %N'                 - Stop job N"
echo "   • 'tail -f logs/MODEL_output.log' - Monitor specific model output"
echo ""
echo "📁 LOG FILES:"
echo "   All model outputs are saved in the 'logs/' directory"
echo "   Each model has its own log file: MODEL_output.log"
echo ""
echo "🔍 QUICK STATUS CHECK:"
jobs
echo ""
echo "💡 TIPS:"
echo "   • Use 'jobs' to see which models are still running"
echo "   • Models will complete automatically in the background"
echo "   • Check logs/ directory for detailed output from each model"
echo "   • Use 'fg %N' to monitor a specific model interactively"
echo ""
echo "🚀 Happy benchmarking!"
