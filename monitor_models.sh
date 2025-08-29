#!/bin/bash

# Script to monitor all running univariate models

echo "🔍 Monitoring all running univariate models..."
echo "=" * 50

# Check if logs directory exists
if [ ! -d "logs" ]; then
    echo "❌ No logs directory found. Run the models first!"
    exit 1
fi

# Function to show model status
show_model_status() {
    local model_name=$1
    local log_file="logs/${model_name}_output.log"
    
    if [ -f "$log_file" ]; then
        local last_line=$(tail -n 1 "$log_file" 2>/dev/null)
        local file_size=$(wc -c < "$log_file" 2>/dev/null)
        
        if [ "$file_size" -gt 0 ]; then
            if echo "$last_line" | grep -q "completed\|finished\|done\|success"; then
                echo "✅ $model_name - COMPLETED"
            elif echo "$last_line" | grep -q "error\|failed\|exception"; then
                echo "❌ $model_name - FAILED"
            else
                echo "🔄 $model_name - RUNNING"
                echo "   📝 Last output: $last_line"
            fi
        else
            echo "⏳ $model_name - STARTING"
        fi
    else
        echo "❓ $model_name - NO LOG FILE"
    fi
}

# Show all models status
echo "📊 MODEL STATUS:"
echo ""

show_model_status "ARIMA"
show_model_status "LSTM"
show_model_status "XGBoost"
show_model_status "Theta"
show_model_status "SVR"
show_model_status "Prophet"
show_model_status "RandomForest"
show_model_status "ExponentialSmoothing"
show_model_status "SeasonalNaive"
show_model_status "DeepAR"
show_model_status "TabPFN"

echo ""
echo "🔍 BACKGROUND JOBS:"
jobs

echo ""
echo "📁 LOG FILES:"
ls -la logs/ 2>/dev/null | grep "_output.log" || echo "No log files found"

echo ""
echo "💡 MONITORING COMMANDS:"
echo "   • 'tail -f logs/MODEL_output.log' - Watch specific model in real-time"
echo "   • 'jobs'                         - Show background jobs"
echo "   • 'fg %N'                        - Bring job to foreground"
echo "   • './monitor_models.sh'          - Run this status check again"
echo ""
echo "🔄 Run './monitor_models.sh' again to refresh status"
