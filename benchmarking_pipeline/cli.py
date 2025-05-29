"""
Command-line interface for the benchmarking pipeline.
"""

import argparse
import yaml

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmarking Pipeline CLI')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    config = load_config(args.config)
    # TODO: Implement pipeline execution logic

if __name__ == '__main__':
    main() 