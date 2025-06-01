"""
SLURM configuration dataclass for job submission.
"""

import os
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class SlurmConfig:
    """Configuration for SLURM job submission."""
    partition: str = ""
    time: str = "01:00:00"
    mem: str = ""
    cpus_per_task: int = 1
    nodes: int = 1
    ntasks: int = 1
    job_name: str = "simulation"
    account: Optional[str] = None
    qos: Optional[str] = None
    constraint: Optional[str] = None
    gres: Optional[str] = None
    modules: List[str] = field(default_factory=list)
    conda_env: Optional[str] = None
    setup_commands: List[str] = field(default_factory=list)
    output_dir: str = "slurm_output"  # Deprecated - use slurm_output_dir
    slurm_output_dir: Optional[str] = None  # Directory for SLURM logs (stdout/stderr)
    array_size: Optional[int] = None
    email: Optional[str] = None
    email_type: str = "END,FAIL"  # Email on job end and failure

    def to_sbatch_header(self) -> str:
        """Convert config to SBATCH header lines."""
        lines = ["#!/bin/bash"]
        lines.append(f"#SBATCH --job-name={self.job_name}")
        lines.append(f"#SBATCH --time={self.time}")
        lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        lines.append(f"#SBATCH --nodes={self.nodes}")
        lines.append(f"#SBATCH --ntasks={self.ntasks}")

        if self.mem:
            lines.append(f"#SBATCH --mem={self.mem}")
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
        if self.array_size:
            lines.append(f"#SBATCH --array=0-{self.array_size-1}")
        if self.email:
            lines.append(f"#SBATCH --mail-user={self.email}")
            lines.append(f"#SBATCH --mail-type={self.email_type}")

        # Output files
        if self.slurm_output_dir:
            output_dir = self.slurm_output_dir
        else:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        if self.array_size:
            lines.append(f"#SBATCH --output={output_dir}/%x_%A_%a.out")
            lines.append(f"#SBATCH --error={output_dir}/%x_%A_%a.err")
        else:
            lines.append(f"#SBATCH --output={output_dir}/%x_%j.out")
            lines.append(f"#SBATCH --error={output_dir}/%x_%j.err")

        lines.append("")

        # Environment setup
        if self.modules:
            for module in self.modules:
                lines.append(f"module load {module}")

        if self.conda_env:
            lines.append(f"conda activate {self.conda_env}")

        if self.setup_commands:
            for cmd in self.setup_commands:
                lines.append(cmd)

        lines.append("")

        return "\n".join(lines)
