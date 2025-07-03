"""Setup configuration for Kaggle Automation Agent."""

import os
import sys
from setuptools import setup, find_packages

# Check if running in conda environment
is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))

if is_conda:
    print("🐍 Detected Conda environment")
    print("📦 Note: Some packages are better installed via conda.")
    print("   Please ensure you've run: conda env create -f environment.yml")
else:
    print("📦 Installing in standard Python environment")
    print("🐍 Consider using Conda for better package compatibility:")
    print("   conda env create -f environment.yml")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kaggle-agent",
    version="0.1.0",
    author="Kaggle Agent Team",
    description="A fully automated ML pipeline for Kaggle competitions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "kaggle>=1.5.12",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "lightgbm>=3.0.0",
        "xgboost>=1.5.0",
        "catboost>=1.0.0",
        "optuna>=3.0.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
    ],
    entry_points={
        "console_scripts": [
            "kaggle-agent=kaggle_agent.cli:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "pytest-mock",
            "black",
            "flake8",
            "mypy",
            "isort",
            "pre-commit",
            "bandit",
            "safety",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
        ],
    },
)