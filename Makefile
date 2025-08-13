.PHONY: install test clean run-benchmark

# One-command installation
install:
	@echo "🚀 Installing SMLCRM Benchmark Pipeline..."
	pip install -e .
	@echo "✅ Installation complete!"

# Run a test benchmark
test:
	@echo "🧪 Running test benchmark..."
	python benchmarking_pipeline/run_benchmark.py --config benchmarking_pipeline/configs/random_forest_test.yaml

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run multivariate benchmark
run-multivariate:
	@echo "🎯 Running multivariate benchmark..."
	python benchmarking_pipeline/run_benchmark.py --config benchmarking_pipeline/configs/multivariate_forecast_horizon_config.yaml

# Help
help:
	@echo "Available commands:"
	@echo "  make install      - Install package and dependencies"
	@echo "  make test         - Run test benchmark"
	@echo "  make run-multivariate - Run multivariate benchmark"  
	@echo "  make clean        - Clean build artifacts"
	@echo "  make help         - Show this help"
