"""
SLURM configuration dataclass for job submission.
"""

import os
from typing import List, Optional, Dict, Any, Literal
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
    nodelist: Optional[str] = None  # Specific nodes to use (e.g., "node001" or "node[001-003]")
    exclude_nodes: Optional[str] = None   # Nodes to exclude (e.g., "node004,node005")
    modules: List[str] = field(default_factory=list)
    conda_env: Optional[str] = None
    setup_commands: List[str] = field(default_factory=list)
    output_dir: str = "slurm_output"  # Deprecated - use slurm_output_dir
    slurm_output_dir: Optional[str] = None  # Directory for SLURM logs (stdout/stderr)
    array_size: Optional[int] = None
    array_throttle: Optional[int] = 250  # Default throttle for array jobs
    email: Optional[str] = None
    email_type: str = "END,FAIL"  # Email on job end and failure

    # Hyperthreading and CPU binding configuration
    hint: Optional[Literal["nomultithread", "multithread", "compute_bound", "memory_bound"]] = "nomultithread"
    threads_per_core: Optional[int] = 1  # Default: disable hyperthreading
    distribution: Optional[str] = "cyclic:cyclic:block"  # Round-robin nodes, round-robin sockets, block cores

    # Tmpfs configuration
    use_tmpfs: bool = True  # Enable tmpfs by default
    tmpfs_use_dev_shm: bool = True  # Try /dev/shm first
    tmpfs_use_tmp: bool = True  # Fall back to /tmp if needed
    tmpfs_custom_path: Optional[str] = None  # Custom tmpfs path (e.g., /scratch/local)
    tmpfs_exclude_patterns: List[str] = field(default_factory=lambda: ['*.tmp', '*.swp', '.nfs*'])

    def __post_init__(self):
        """
        Post-initialization validation and adjustments.
        This runs automatically after the dataclass __init__.
        """
        # Warn if potentially conflicting settings
        if self.hint == "multithread" and self.threads_per_core == 1:
            print("Warning: hint='multithread' but threads_per_core=1. Consider threads_per_core=2.")

        if self.hint == "nomultithread" and self.threads_per_core == 2:
            print("Warning: hint='nomultithread' but threads_per_core=2. These settings conflict.")

        # Don't auto-set memory for array jobs or if explicitly set to empty string
        # Only auto-set if mem is None (not specified at all)
        if self.mem is None and not self.array_size and self.cpus_per_task:
            self.mem = f"{4 * self.cpus_per_task}G"  # Default 4GB per CPU for non-array jobs

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
        if self.nodelist:
            lines.append(f"#SBATCH --nodelist={self.nodelist}")
        if self.exclude_nodes:
            lines.append(f"#SBATCH --exclude={self.exclude_nodes}")
        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
        if self.array_size:
            if self.array_throttle:
                lines.append(f"#SBATCH --array=0-{self.array_size-1}%{self.array_throttle}")
            else:
                lines.append(f"#SBATCH --array=0-{self.array_size-1}")
        if self.email:
            lines.append(f"#SBATCH --mail-user={self.email}")
            lines.append(f"#SBATCH --mail-type={self.email_type}")

        # Hyperthreading and CPU binding options
        if self.hint:
            lines.append(f"#SBATCH --hint={self.hint}")
        if self.threads_per_core is not None:
            lines.append(f"#SBATCH --threads-per-core={self.threads_per_core}")
        if self.distribution:
            lines.append(f"#SBATCH --distribution={self.distribution}")

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

    def get_tmpfs_config(self) -> Dict[str, Any]:
        """Get tmpfs configuration as a dictionary."""
        return {
            'use_tmpfs': self.use_tmpfs,
            'tmpfs_use_dev_shm': self.tmpfs_use_dev_shm,
            'tmpfs_use_tmp': self.tmpfs_use_tmp,
            'tmpfs_custom_path': self.tmpfs_custom_path,
            'tmpfs_exclude_patterns': self.tmpfs_exclude_patterns
        }

    def get_cpu_config(self) -> Dict[str, Any]:
        """Get CPU and hyperthreading configuration as a dictionary."""
        return {
            'cpus_per_task': self.cpus_per_task,
            'ntasks': self.ntasks,
            'hint': self.hint,
            'threads_per_core': self.threads_per_core,
            'distribution': self.distribution
        }

    @classmethod
    def for_array_job(cls,
                      job_name: str,
                      array_size: int,
                      cpus_per_task: int = 1,
                      time: str = "01:00:00",
                      mem: Optional[str] = None,
                      array_throttle: Optional[int] = 250,
                      **kwargs) -> 'SlurmConfig':
        """
        Create config for array jobs (embarrassingly parallel, independent tasks).

        Args:
            job_name: Name of the job
            array_size: Number of array tasks (0 to array_size-1)
            cpus_per_task: CPUs per array task
            time: Wall time limit
            mem: Memory limit per task (optional)
            **kwargs: Additional SLURM options

        Returns:
            SlurmConfig optimized for array jobs
        """
        defaults = {
            'array_size': array_size,
            'array_throttle': array_throttle,
            'ntasks': 1,  # Each array task is single-threaded
            'nodes': 1,   # Each array task gets one node allocation
            'hint': 'nomultithread',
            'threads_per_core': 1,
            'distribution': None  # Not needed for array jobs
        }

        # Don't set memory if not specified
        if mem:
            defaults['mem'] = mem

        defaults.update(kwargs)
        return cls(job_name=job_name,
                  cpus_per_task=cpus_per_task,
                  time=time,
                  **defaults)

    @classmethod
    def create(cls,
               job_name: str,
               tasks: int,
               cpus_per_task: int = 1,
               time: str = "01:00:00",
               mem: Optional[str] = None,
               nodes: Optional[int] = None,
               workload_type: Literal["auto", "cpu", "memory", "io"] = "auto",
               **kwargs) -> 'SlurmConfig':
        """
        Simplified constructor with smart defaults for common use cases.

        Args:
            job_name: Name of the job
            tasks: Total number of tasks
            cpus_per_task: CPUs per task (default: 1)
            time: Wall time limit
            mem: Memory limit (e.g., "100G")
            nodes: Number of nodes (default: calculated based on tasks)
            workload_type: Type of workload for optimization
            **kwargs: Additional SLURM options

        Returns:
            SlurmConfig with optimized settings
        """
        # Auto-detect workload type if needed
        if workload_type == "auto":
            if cpus_per_task > 4:
                # Likely parallel computation within task
                workload_type = "cpu"
            elif tasks > 50:
                # Many small tasks, likely I/O bound
                workload_type = "io"
            else:
                # Default to balanced approach
                workload_type = "cpu"

        # Set defaults based on workload type
        if workload_type == "io":
            defaults = {
                'hint': 'multithread',
                'threads_per_core': 2,
                'distribution': 'cyclic:cyclic:block'
            }
        elif workload_type == "memory":
            defaults = {
                'hint': 'nomultithread',
                'threads_per_core': 1,
                'distribution': 'cyclic:cyclic:block'
            }
        else:  # cpu or default
            defaults = {
                'hint': 'nomultithread',
                'threads_per_core': 1,
                'distribution': 'cyclic:cyclic:block' if nodes and nodes > 1 else 'block:cyclic'
            }

        # Merge with user kwargs
        defaults.update(kwargs)

        return cls(
            job_name=job_name,
            ntasks=tasks,
            cpus_per_task=cpus_per_task,
            time=time,
            mem=mem,
            nodes=nodes or (tasks // 44 + (1 if tasks % 44 else 0)),  # Assume 44 cores/node
            **defaults
        )

    @classmethod
    def for_cpu_intensive(cls, **kwargs) -> 'SlurmConfig':
        """
        Create a config optimized for CPU-intensive workloads.
        Disables hyperthreading and binds to physical cores.
        """
        defaults = {
            'hint': 'nomultithread',
            'threads_per_core': 1,
            'distribution': 'block:block'  # Keep tasks together for cache
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_io_bound(cls, **kwargs) -> 'SlurmConfig':
        """
        Create a config optimized for I/O-bound workloads.
        Enables hyperthreading to maximize throughput during I/O waits.
        """
        defaults = {
            'hint': 'multithread',
            'threads_per_core': 2
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_memory_bound(cls, **kwargs) -> 'SlurmConfig':
        """
        Create a config optimized for memory-bound workloads.
        Disables hyperthreading and spreads across NUMA nodes.
        """
        defaults = {
            'hint': 'memory_bound',
            'threads_per_core': 1,
            'distribution': 'cyclic:cyclic:block'
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_independent_tasks(cls, tasks_per_socket: int = 4, **kwargs) -> 'SlurmConfig':
        """
        Create config for independent parallel tasks.
        Distributes tasks across nodes and sockets with local CPU binding.
        """
        # Assuming dual-socket system with 44 total cores
        total_cores_per_node = kwargs.pop('total_cores_per_node', 44)
        sockets_per_node = kwargs.pop('sockets_per_node', 2)
        nodes = kwargs.get('nodes', 1)

        ntasks_per_node = tasks_per_socket * sockets_per_node
        cores_per_socket = total_cores_per_node // sockets_per_node
        cpus_per_task = cores_per_socket // tasks_per_socket

        defaults = {
            'ntasks': ntasks_per_node * nodes,
            'cpus_per_task': cpus_per_task,
            'distribution': 'cyclic:cyclic:block',
            'hint': 'nomultithread',
            'threads_per_core': 1
        }
        defaults.update(kwargs)
        return cls(**defaults)


# Example usage:
if __name__ == "__main__":
    # Simple interface - just specify what you need
    simple_job = SlurmConfig.create(
        job_name="my_analysis",
        tasks=32,
        cpus_per_task=4,
        time="02:00:00"
    )
    print("Simple job (auto-detected settings):")
    print(simple_job.to_sbatch_header())
    print("\n" + "="*50 + "\n")

    # CPU-intensive job
    cpu_job = SlurmConfig.for_cpu_intensive(
        job_name="matrix_mult",
        ntasks=44,
        time="04:00:00",
        mem="100G"
    )
    print("CPU-intensive job:")
    print(cpu_job.to_sbatch_header())
    print("\n" + "="*50 + "\n")

    # I/O-bound job
    io_job = SlurmConfig.for_io_bound(
        job_name="web_scraper",
        ntasks=88,
        time="02:00:00",
        mem="50G"
    )
    print("I/O-bound job:")
    print(io_job.to_sbatch_header())
    print("\n" + "="*50 + "\n")

    # Custom configuration
    custom_job = SlurmConfig(
        job_name="mixed_workload",
        ntasks=66,
        hint="multithread",
        time="01:00:00"
    )
    print("Custom job:")
    print(custom_job.to_sbatch_header())
