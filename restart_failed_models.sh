#!/bin/bash

# Script to restart failed models after fixing the variant bug

echo "🔄 Restarting failed models after fixing the variant bug..."
echo "=" * 50

# Check if logs directory exists
if [ ! -d "logs" ]; then
    echo "❌ No logs directory found. Run the models first!"
    exit 1
fi

# Function to restart a failed model
restart_model() {
    local model_name=$1
    local config_file=$2
    
    echo "🔄 Restarting $model_name..."
    
    # Kill any existing process for this model
    pkill -f "test_${model_name}_univariate.yaml" 2>/dev/null || true
    
    # Clear the old log file
    rm -f "logs/${model_name}_output.log"
    
    # Start the model in the background
    python benchmarking_pipeline/run_benchmark.py --config "$config_file" > "logs/${model_name}_output.log" 2>&1 &
    
    local job_id=$!
    echo "✅ $model_name restarted with job ID: $job_id"
    echo "📝 Output will be saved to: logs/${model_name}_output.log"
    echo ""
}

# Models that failed due to the variant bug
echo "📋 Restarting models that failed due to the variant bug..."
echo ""

restart_model "Prophet" "benchmarking_pipeline/configs/test_prophet_univariate.yaml"
restart_model "RandomForest" "benchmarking_pipeline/configs/test_random_forest_univariate.yaml"
restart_model "SeasonalNaive" "benchmarking_pipeline/configs/test_seasonal_naive_univariate.yaml"
restart_model "DeepAR" "benchmarking_pipeline/configs/test_deepar_univariate.yaml"
restart_model "TabPFN" "benchmarking_pipeline/configs/test_tabpfn_univariate.yaml"

echo ""
echo "🎉 Failed models restarted!"
echo "=" * 50
echo ""
echo "📊 MONITORING COMMANDS:"
echo "   • './monitor_models.sh'              - Check status of all models"
echo "   • 'tail -f logs/MODEL_output.log'    - Watch specific model in real-time"
echo "   • 'jobs'                             - Show background jobs"
echo ""
echo "💡 The models should now run without the variant error."
echo "🔄 Run './monitor_models.sh' in a few minutes to check progress."
