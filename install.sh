#!/bin/bash
# One-command installation script for SMLCRM Benchmark Pipeline

echo "🚀 Installing SMLCRM Benchmark Pipeline..."

# Set PYTHONPATH to include current directory
echo "🔧 Setting PYTHONPATH..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Install the package in development mode (includes requirements.txt)
echo "📦 Installing package and dependencies..."
pip install -e .
pip install -r requirements.txt

echo "✅ Installation complete!"
echo ""
echo "🎯 You can now run benchmarks with:"
echo "   python benchmarking_pipeline/run_benchmark.py --config benchmarking_pipeline/configs/all_model_univariate.yaml"
