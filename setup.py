"""
Setup script for slurm_pipeline module.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="slurm-pipeline",
    version="0.1.0",
    author="Andreas Buttenschoen",
    author_email="andreas@buttenschoen.ca",
    description="A Python module for running simulation pipelines on SLURM clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adrs0049/slurm-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "slurm-pipeline-test=slurm_pipeline.tests.test_slurm_pipeline:main",
            "slurm-pipeline-diagnose=slurm_pipeline.diagnostics.diagnose_slurm:main",
        ],
    },
)
