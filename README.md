# Benchmarking Pipeline

A flexible and modular benchmarking pipeline for machine learning models.

## Project Structure

```
benchmarking_pipeline/
├── configs/                # User-defined YAML/JSON files for test configuration
├── datasets/              # User-extendable dataset chunks 
├── models/               # User-pluggable models as Python classes
│   └── my_model/
│       └── model.py
├── pipeline/             # Core pipeline components
│   ├── data_loader.py    # Data loading functionality (HuggingFace datasets)
│   ├── preprocessing.py   # Data preprocessing
│   ├── feature_extraction.py  # Feature extraction
│   ├── trainer.py        # Model training
│   ├── evaluator.py      # Model evaluation
│   └── logger.py         # Logging utilities
├── cli.py               # Command-line interface
└── run_benchmark.py     # End-to-end orchestration
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Smlcrm/benchmark.git
cd benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Create or modify a configuration file in the `configs/` directory.

2. Run the benchmark using the CLI:
```bash
python benchmarking_pipeline/cli.py --config configs/default_config.yaml
```

## How to Run the Benchmarking Pipeline

To run the benchmarking pipeline with a specific configuration file:

```bash
python benchmarking_pipeline/run_benchmark.py --config benchmarking_pipeline/configs/your_config.yaml
```

- Replace `your_config.yaml` with the path to your desired YAML config file.
- The config file controls which models are run, their hyperparameters, and the dataset used.
- Results and TensorBoard logs will be saved in the `runs/` and `results/` directories.

## How to Plot a Dataset

To quickly visualize a dataset (after preprocessing) using the provided script:

```bash
python test_plot.py --dataset_path benchmarking_pipeline/datasets/your_dataset_folder
```

- Replace `your_dataset_folder` with the path to your dataset directory.
- Optionally, you can pass a custom config for preprocessing:

```bash
python test_plot.py --dataset_path benchmarking_pipeline/datasets/your_dataset_folder --config benchmarking_pipeline/configs/your_config.yaml
```

This will display a plot of the full time series after preprocessing.

---

For more details on configuration options and model-specific settings, see the example YAML files in `benchmarking_pipeline/configs/`.

## Configuration

The pipeline is configured using YAML files in the `configs/` directory. See `configs/default_config.yaml` for an example configuration.

## Features

- Flexible model benchmarking pipeline
- Support for HuggingFace datasets
- Modular preprocessing and feature extraction
- Comprehensive logging and metrics tracking
- Configurable model training and evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
