#!/usr/bin/env python3
"""
Setup script for the SMLCRM Benchmarking Pipeline
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle version specifiers like >=1.0.0
                    requirements.append(line)
    return requirements

setup(
    name="benchmarking_pipeline",
    version="1.0.0",
    description="Time Series Forecasting Benchmarking Pipeline",
    author="SMLCRM Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'run_benchmark=benchmarking_pipeline.run_benchmark:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
