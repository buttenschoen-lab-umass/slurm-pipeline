"""
SLURM Pipeline - A Python module for running simulation pipelines on SLURM clusters.
"""

__version__ = "0.1.0"

# Import main classes for easy access
from .core.pipeline import SimulationPipeline, SimulationResult
from .core.ensemble import Ensemble
from .core.parameter_scan import ParameterScan

from .slurm.config import SlurmConfig
from .slurm.pipeline import SlurmPipeline
from .slurm.monitor import SlurmMonitor
from .slurm.job import JobState, JobInfo

__all__ = [
    "SimulationPipeline",
    "SimulationResult",
    "Ensemble",
    "ParameterScan",
    "SlurmConfig",
    "SlurmPipeline",
    "SlurmMonitor",
    "JobState",
    "JobInfo",
]
