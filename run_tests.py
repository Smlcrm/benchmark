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


def check_dependency(dependency_name, import_name):
    """Check if a dependency is available."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


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
        help="Generate coverage report (overrides pytest.ini settings)"
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Skip slow tests (if slow marker is defined)"
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
    
    # Handle coverage - only add if explicitly requested and pytest-cov is available
    if args.coverage:
        if check_dependency("pytest-cov", "pytest_cov"):
            # Override pytest.ini coverage settings with explicit ones
            cmd.extend([
                "--cov=benchmarking_pipeline", 
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml"
            ])
        else:
            print("❌ Warning: pytest-cov not available. Coverage option ignored.")
            print("   Install with: pip install pytest-cov")
    
    # Handle fast option - only add if slow marker exists
    if args.fast:
        # Check if we can find any tests with slow marker
        slow_check_cmd = ["python", "-m", "pytest", "--collect-only", "-m", "slow", "tests/"]
        try:
            result = subprocess.run(slow_check_cmd, capture_output=True, text=True)
            if "collected 0 items" not in result.stdout:
                cmd.extend(["-m", "not slow"])
                print("ℹ️  Fast mode: Skipping slow tests")
            else:
                print("ℹ️  Fast mode: No slow tests found, running all tests")
        except Exception:
            print("ℹ️  Fast mode: Could not determine slow tests, running all tests")
    
    # Handle parallel execution
    if args.parallel:
        if check_dependency("pytest-xdist", "xdist"):
            cmd.extend(["-n", "auto"])
            print("ℹ️  Parallel mode: Running tests in parallel")
        else:
            print("❌ Warning: pytest-xdist not available. Parallel option ignored.")
            print("   Install with: pip install pytest-xdist")
    
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
