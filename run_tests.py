#!/usr/bin/env python3
"""
Test runner script for the benchmarking pipeline.

This script provides convenient ways to run different types of tests:
- All tests
- Unit tests only
- Integration tests only
- End-to-end tests only
- Specific test categories
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        print("Please ensure pytest is installed: pip install pytest pytest-cov")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run benchmarking pipeline tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "e2e", "smoke"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests with verbose output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Skip slow tests"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add options based on arguments
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=benchmarking_pipeline", "--cov-report=term-missing"])
    
    if args.fast:
        cmd.append("-m")
        cmd.append("not slow")
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Add test type specific options
    if args.type == "unit":
        cmd.extend(["-m", "unit"])
        description = "Unit Tests"
    elif args.type == "integration":
        cmd.extend(["-m", "integration"])
        description = "Integration Tests"
    elif args.type == "e2e":
        cmd.extend(["-m", "e2e"])
        description = "End-to-End Tests"
    elif args.type == "smoke":
        cmd.extend(["-m", "smoke"])
        description = "Smoke Tests"
    else:  # all
        description = "All Tests"
    
    # Add test directory
    cmd.append("tests/")
    
    # Run the tests
    success = run_command(cmd, description)
    
    if success:
        print(f"\n🎉 {description} completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 {description} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
