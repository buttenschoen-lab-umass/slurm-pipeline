"""
Main SLURM pipeline class for submitting and managing simulation jobs.

Simplified version: One ensemble = One SLURM job
- Single ensembles: Submit as regular job with N CPUs
- Parameter scans: Submit as array job, one task per parameter point
"""

import os
import sys
import json
import pickle
import subprocess
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime
from collections.abc import Callable

from .config import SlurmConfig
from .monitor import SlurmMonitor
from .job_tracker import JobTracker
from .runner_generator import get_runner_script_content
from .environment_capture import EnvironmentCapture


class SlurmPipeline:
    """Simplified SLURM pipeline wrapper - one job per ensemble."""

    def __init__(self,
                 nfs_work_dir: str = "/home/adrs0061/cluster/slurm_pipeline",
                 monitor_interval: int = 10,
                 track_jobs: bool = True,
                 capture_environment: bool = True,
                 capture_log_dir: Optional[str] = None,
                 log_level: str = "INFO"):
        """
        Initialize SLURM pipeline wrapper.

        Args:
            nfs_work_dir: NFS directory accessible from all nodes (can use $USER)
            monitor_interval: Seconds between status checks for monitoring
            track_jobs: Whether to track and auto-cancel jobs on exit
            capture_environment: Whether to capture and transmit Python environment
            capture_log_dir: Directory for environment capture logs
            log_level: Logging level
        """
        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(getattr(logging, log_level.upper()))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Expand environment variables in paths
        self.nfs_work_dir = Path(os.path.expandvars(nfs_work_dir))
        self.nfs_work_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories
        self.nfs_input_dir = self.nfs_work_dir / "inputs"
        self.nfs_input_dir.mkdir(exist_ok=True)

        self.nfs_scripts_dir = self.nfs_work_dir / "scripts"
        self.nfs_scripts_dir.mkdir(exist_ok=True)

        # Store submission info in NFS too
        self.submission_info_dir = self.nfs_work_dir / "submissions"
        self.submission_info_dir.mkdir(exist_ok=True)

        # Job tracking
        self.track_jobs = track_jobs
        if track_jobs:
            self.job_tracker = JobTracker()
        else:
            self.job_tracker = None

        self.monitor = SlurmMonitor(check_interval=monitor_interval, job_tracker=self.job_tracker)
        self.runner_script = None

        # Environment capture
        self.capture_environment = capture_environment
        if capture_log_dir:
            self.capture_log_dir = Path(capture_log_dir)
        else:
            self.capture_log_dir = self.nfs_work_dir / "capture_logs"
        self.capture_log_dir.mkdir(exist_ok=True, parents=True)

    def _normalize_output_dir(self, output_dir: Optional[str], default_subdir: str) -> str:
        """
        Normalize output directory path.

        If output_dir is:
        - None: use nfs_work_dir/outputs/default_subdir
        - Relative path: prepend nfs_work_dir/outputs/
        - Absolute path: use as-is

        Args:
            output_dir: User-provided output directory
            default_subdir: Default subdirectory name if output_dir is None

        Returns:
            Absolute path to output directory
        """
        if output_dir is None:
            # Use default under NFS outputs
            output_dir = str(self.nfs_work_dir / "outputs" / default_subdir)
        else:
            output_dir = os.path.expandvars(output_dir)

            # Check if it's a relative path
            if not os.path.isabs(output_dir):
                # Prepend NFS outputs directory
                output_dir = str(self.nfs_work_dir / "outputs" / output_dir)

        # Ensure directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        return output_dir

    def _serialize_object(self, obj: Any, name: str) -> Path:
        """Serialize a pipeline object to NFS directory."""
        filepath = self.nfs_input_dir / f"{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        return filepath

    def _serialize_with_environment(self, obj: Any, name: str,
                                   analysis_functions: Optional[Dict[str, Callable]] = None,
                                   plot_functions: Optional[Dict[str, Callable]] = None,
                                   ensemble_analysis_functions: Optional[Dict[str, Callable]] = None,
                                   ensemble_plot_functions: Optional[Dict[str, Callable]] = None) -> Tuple[Path, Optional[Path]]:
        """Serialize object and optionally capture environment."""

        # Serialize the object
        object_file = self._serialize_object(obj, name)

        if not self.capture_environment:
            return object_file, None

        # Set up logging for this capture
        log_file = self.capture_log_dir / f"{name}_capture.log"
        logger = logging.getLogger(f"capture_{name}")
        logger.setLevel(logging.INFO)
        logger.handlers = []  # Clear any existing handlers

        # File handler with full details
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        # Console handler - minimal output
        # Actually, let's just use print() for console output instead
        logger.propagate = False  # Don't propagate to root logger

        print(f"\n{'='*60}")
        print(f"ENVIRONMENT CAPTURE FOR: {name}")
        print(f"Capture log: {log_file}")
        print(f"{'='*60}\n")

        # Create environment capture
        env_capture = EnvironmentCapture(logger)

        # Capture current environment
        env_capture.capture_current_environment()

        # Capture object dependencies
        env_capture.capture_object_dependencies(
            obj, name,
            analysis_functions=analysis_functions,
            plot_functions=plot_functions,
            ensemble_analysis_functions=ensemble_analysis_functions,
            ensemble_plot_functions=ensemble_plot_functions
        )

        # Save environment
        env_file = self.nfs_input_dir / f"{name}_env.pkl"
        with open(env_file, 'wb') as f:
            pickle.dump(env_capture, f)

        logger.info(f"\nEnvironment saved to: {env_file}")

        # Save a summary report
        summary_file = self.capture_log_dir / f"{name}_capture_summary.json"
        summary = env_capture.get_capture_summary()

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nCapture summary saved to: {summary_file}")
        print(f"\nCapture Summary:")
        print(f"  - Required modules: {summary['total_required_modules']}")
        print(f"  - Inline functions: {summary['total_inline_functions']}")
        print(f"  - Python paths: {summary['total_paths']}")

        if summary['inline_functions']:
            print(f"\nCaptured inline functions:")
            for func_name in summary['inline_functions'][:10]:
                print(f"    - {func_name}")
            if len(summary['inline_functions']) > 10:
                print(f"    ... and {len(summary['inline_functions']) - 10} more")

        return object_file, env_file

    def _create_runner_script(self) -> Path:
        """Create the runner script in NFS directory with optional environment support."""
        if self.runner_script is None:
            runner_path = self.nfs_scripts_dir / "runner.py"

            # Get the runner script content
            runner_content = get_runner_script_content()

            # Write the runner script
            with open(runner_path, 'w') as f:
                f.write(runner_content)
            runner_path.chmod(0o755)

            # Also create the environment module if capture is enabled
            if self.capture_environment:
                from .runner_env_module import RunnerEnvironmentModule
                RunnerEnvironmentModule.create_module_file(self.nfs_scripts_dir)
                self.logger.info("Created runner environment module")

            self.runner_script = runner_path
            self.logger.info(f"Created runner script: {runner_path}")

        return self.runner_script

    def _create_slurm_script(self,
                           config: SlurmConfig,
                           runner_script: Path,
                           object_file: Path,
                           config_file: Path,
                           array_jobs: bool,
                           env_file: Optional[Path] = None) -> Path:
        """Create SLURM submission script."""
        script_content = config.to_sbatch_header()

        # Add directory change so runner_env.py can be imported
        script_content += f"\n# Change to scripts directory for imports\ncd {self.nfs_scripts_dir}\n"

        # Add Python command
        if array_jobs:
            cmd = f"python {runner_script} {object_file} {config_file} $SLURM_ARRAY_TASK_ID"
        else:
            cmd = f"python {runner_script} {object_file} {config_file}"

        if env_file:
            cmd += f" {env_file}"

        script_content += f"\n{cmd}\n"

        # Save script
        script_path = self.nfs_scripts_dir / f"{config.job_name}.sbatch"
        with open(script_path, 'w') as f:
            f.write(script_content)

        script_path.chmod(0o755)
        return script_path

    def submit_ensemble(self,
                       ensemble,
                       n_simulations: int,
                       T: float = 50.0,
                       output_dir: Optional[str] = None,
                       slurm_config: Optional[SlurmConfig] = None,
                       wait_for_completion: bool = False,
                       **kwargs) -> Dict[str, Any]:
        """
        Submit a single ensemble to SLURM.

        Args:
            ensemble: Ensemble object to run
            n_simulations: Number of simulations to run
            T: Simulation time
            output_dir: Output directory (if None, uses NFS work dir)
            slurm_config: SLURM configuration
            wait_for_completion: If True, wait for job to complete
            **kwargs: Additional arguments passed to ensemble.run()

        Returns:
            Dictionary with job_id and other submission info
        """
        if slurm_config is None:
            slurm_config = SlurmConfig()

        # Generate unique job name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slurm_config.job_name = f"ensemble_{timestamp}"

        # Normalize output directory
        output_dir = self._normalize_output_dir(output_dir, slurm_config.job_name)

        # Simple configuration
        config = {
            'type': 'ensemble',
            'n_simulations': n_simulations,
            'T': T,
            'output_dir': output_dir,
            'max_workers': slurm_config.cpus_per_task,  # Use all allocated CPUs
            **kwargs
        }

        # No array jobs for single ensemble
        slurm_config.array_size = None

        # Update SLURM output directory to use NFS
        slurm_config.slurm_output_dir = str(self.nfs_work_dir / "logs")

        # Serialize ensemble and config to NFS
        ensemble_file, env_file = self._serialize_with_environment(
            ensemble, slurm_config.job_name,
            analysis_functions=ensemble.analysis_functions,
            plot_functions=ensemble.plot_functions,
            ensemble_analysis_functions=ensemble.ensemble_analysis_functions,
            ensemble_plot_functions=ensemble.ensemble_plot_functions
        )

        config_file = self.nfs_input_dir / f"{slurm_config.job_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Create runner script
        runner_script = self._create_runner_script()

        # Create SLURM script
        slurm_script = self._create_slurm_script(
            slurm_config, runner_script, ensemble_file, config_file,
            array_jobs=False, env_file=env_file
        )

        # Submit job
        job_id = self._submit_job(slurm_script)

        result = {
            'job_id': job_id,
            'job_name': slurm_config.job_name,
            'output_dir': output_dir,
            'n_simulations': n_simulations,
            'config_file': str(config_file),
            'ensemble_file': str(ensemble_file)
        }

        if env_file:
            result['env_file'] = str(env_file)
            result['capture_log'] = str(self.capture_log_dir / f"{slurm_config.job_name}_capture.log")

        if wait_for_completion:
            # Monitor job progress
            job_info = self.monitor.monitor_job(
                job_id=result['job_id'],
                job_name=result['job_name'],
                output_dir=result['output_dir'],
                show_progress=True
            )

            result['job_info'] = job_info
            result['status'] = job_info.state.value

        return result

    def submit_parameter_scan(self,
                            scan,
                            n_simulations_per_point: int,
                            T: float = 50.0,
                            output_dir: Optional[str] = None,
                            slurm_config: Optional[SlurmConfig] = None,
                            wait_for_completion: bool = False,
                            **kwargs) -> Dict[str, Any]:
        """
        Submit a parameter scan to SLURM.

        Args:
            scan: ParameterScan object to run
            n_simulations_per_point: Simulations per parameter point
            T: Simulation time
            output_dir: Output directory (if None, uses NFS work dir)
            slurm_config: SLURM configuration
            wait_for_completion: If True, wait for job to complete
            **kwargs: Additional arguments passed to scan.run()

        Returns:
            Dictionary with job_id and other submission info
        """
        if slurm_config is None:
            slurm_config = SlurmConfig()

        # Calculate number of parameter points
        import itertools
        n_points = len(list(itertools.product(*scan.scan_params.values())))

        # Generate unique job name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slurm_config.job_name = f"scan_{timestamp}"

        # Normalize output directory
        output_dir = self._normalize_output_dir(output_dir, slurm_config.job_name)

        # Simple configuration
        config = {
            'type': 'parameter_scan',
            'n_simulations_per_point': n_simulations_per_point,
            'T': T,
            'scan_params': scan.scan_params,
            'base_params': scan.base_params,
            'output_dir': output_dir,
            'max_workers': slurm_config.cpus_per_task,  # Each task uses all its CPUs
            **kwargs
        }

        # Always use array jobs for scans
        slurm_config.array_size = n_points

        self.logger.info(f"Parameter scan configuration:")
        self.logger.info(f"  - {n_points} parameter points → {n_points} array tasks")
        self.logger.info(f"  - {slurm_config.cpus_per_task} CPUs per task")
        self.logger.info(f"  - {n_simulations_per_point} simulations per point")
        self.logger.info(f"  - Total: {n_points * n_simulations_per_point} simulations")

        # Update SLURM output directory
        slurm_config.slurm_output_dir = str(self.nfs_work_dir / "logs")

        # Serialize scan and config to NFS
        scan_file, env_file = self._serialize_with_environment(
            scan, slurm_config.job_name,
            analysis_functions=scan.analysis_functions,
            plot_functions=scan.plot_functions,
            ensemble_analysis_functions=scan.ensemble_analysis_functions,
            ensemble_plot_functions=scan.ensemble_plot_functions
        )

        config_file = self.nfs_input_dir / f"{slurm_config.job_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Create runner script
        runner_script = self._create_runner_script()

        # Create SLURM script
        slurm_script = self._create_slurm_script(
            slurm_config, runner_script, scan_file, config_file,
            array_jobs=True, env_file=env_file
        )

        # Submit job
        job_id = self._submit_job(slurm_script)

        result = {
            'job_id': job_id,
            'job_name': slurm_config.job_name,
            'output_dir': output_dir,
            'n_points': n_points,
            'n_simulations_per_point': n_simulations_per_point,
            'config_file': str(config_file),
            'scan_file': str(scan_file),
            'scan_params': scan.scan_params
        }

        if env_file:
            result['env_file'] = str(env_file)
            result['capture_log'] = str(self.capture_log_dir / f"{slurm_config.job_name}_capture.log")

        if wait_for_completion:
            # Monitor array job completion
            job_info = self.monitor.monitor_scan(result, show_progress=True)
            result['job_info'] = job_info
            result['status'] = job_info.state.value

        return result

    def _submit_job(self, script_path: Path) -> str:
        """Submit SLURM script and return job ID."""
        # Ensure script is executable
        script_path.chmod(0o755)

        # Verify script exists and is readable
        if not script_path.exists():
            raise RuntimeError(f"Script not found: {script_path}")

        print(f"Submitting script: {script_path}")

        result = subprocess.run(
            ['sbatch', str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(script_path.parent)  # Run from script directory
        )

        # Save submission info regardless of success
        submission_info = {
            'script_path': str(script_path),
            'submission_time': datetime.now().isoformat(),
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'cwd': str(script_path.parent)
        }

        # Extract job name from script if available
        job_name = None
        if script_path.exists():
            with open(script_path, 'r') as f:
                script_content = f.read()

            for line in script_content.split('\n'):
                if line.startswith('#SBATCH --job-name='):
                    job_name = line.split('=')[1].strip()
                    break

            submission_info['job_name'] = job_name

        if result.returncode != 0:
            # Save failed submission info
            info_file = self.submission_info_dir / f"submission_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(info_file, 'w') as f:
                json.dump(submission_info, f, indent=2)

            error_msg = f"SLURM submission failed (code {result.returncode})\n"
            error_msg += f"stdout: {result.stdout}\n"
            error_msg += f"stderr: {result.stderr}\n"
            error_msg += f"Script: {script_path}\n"
            error_msg += f"Details saved to: {info_file}"

            raise RuntimeError(error_msg)

        # Extract job ID from output (format: "Submitted batch job 12345")
        try:
            job_id = result.stdout.strip().split()[-1]
            # Verify it looks like a job ID (should be numeric)
            int(job_id)  # This will raise ValueError if not numeric
        except (IndexError, ValueError) as e:
            error_msg = f"Failed to parse job ID from sbatch output: '{result.stdout}'"
            raise RuntimeError(error_msg) from e

        print(f"Submitted job: {job_id}")

        # Track job if enabled
        if self.job_tracker:
            self.job_tracker.add_job(job_id, job_name)

        # Save successful submission info
        submission_info['job_id'] = job_id
        info_file = self.submission_info_dir / f"submission_{job_id}.json"
        with open(info_file, 'w') as f:
            json.dump(submission_info, f, indent=2)

        return job_id

    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status of a submitted job."""
        return self.monitor.get_job_info(job_id).__dict__

    def cancel_job(self, job_id: str, reason: str = "User requested") -> bool:
        """
        Cancel a specific SLURM job.

        Args:
            job_id: SLURM job ID to cancel
            reason: Reason for cancellation

        Returns:
            True if cancellation was successful
        """
        if self.job_tracker:
            return self.job_tracker.cancel_job(job_id, reason)
        else:
            # Direct cancellation without tracking
            result = subprocess.run(['scancel', job_id], capture_output=True, text=True)
            return result.returncode == 0

    def cancel_all_active_jobs(self) -> Dict[str, int]:
        """
        Cancel all currently active jobs.

        Returns:
            Dictionary with cancellation statistics
        """
        if self.job_tracker:
            return self.job_tracker.cancel_all_active()
        else:
            return {'cancelled': 0, 'failed': 0, 'already_done': 0}

    def get_tracking_status(self) -> Dict[str, Any]:
        """
        Get current job tracking status.

        Returns:
            Dictionary with tracking information
        """
        if self.job_tracker:
            return self.job_tracker.get_status()
        return {'tracking_enabled': False}

    def disable_auto_cancel(self):
        """Disable automatic job cancellation on script exit."""
        if self.job_tracker:
            self.job_tracker.disable_auto_cancel()

    def enable_auto_cancel(self):
        """Enable automatic job cancellation on script exit."""
        if self.job_tracker:
            self.job_tracker.enable_auto_cancel()

    def set_tracking_verbose(self, verbose: bool):
        """Set verbosity for job tracking messages."""
        if self.job_tracker:
            self.job_tracker.set_verbose(verbose)
