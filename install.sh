#!/bin/bash
# One-command installation script for SMLCRM Benchmark Pipeline

echo "🚀 Installing SMLCRM Benchmark Pipeline..."

# Install the package in development mode (includes requirements.txt)
echo "📦 Installing package and dependencies..."
pip install -e .

echo "✅ Installation complete!"
echo ""
echo "🎯 You can now run benchmarks with:"
echo "   python benchmarking_pipeline/run_benchmark.py --config your_config.yaml"
echo ""
echo "💡 Threading conflicts are automatically handled in the code."
echo "   No additional environment variables needed!"
