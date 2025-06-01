"""
SLURM job
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class JobState(Enum):
    """SLURM job states."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass
class JobInfo:
    """Information about a SLURM job."""
    job_id: str
    job_name: str
    state: JobState
    start_time: Optional[datetime] = None
    elapsed_time: Optional[timedelta] = None
    time_limit: Optional[timedelta] = None
    nodes: int = 0
    array_size: Optional[int] = None
    array_completed: int = 0
    reason: Optional[str] = None
    exit_code: Optional[str] = None



