"""
Main SLURM pipeline class for submitting and managing simulation jobs.

This version properly handles NFS directories for cluster-wide access and includes
automatic job tracking and cancellation functionality.
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
from .monitor import SlurmMonitor, JobInfo, JobState
from .job_tracker import JobTracker
from .runner_generator import get_runner_script_content
from .environment_capture import EnvironmentCapture


class SlurmPipeline:
    """Wrapper to run pipeline objects on SLURM with automatic job tracking and environment capture."""

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

        self.monitor = SlurmMonitor(check_interval=monitor_interval, job_tracker=self.job_tracker if track_jobs else None)
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
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ],
            force=True
        )
        logger = logging.getLogger(f"capture_{name}")

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

            # Get the base runner script content
            from .runner_generator import get_runner_script_content
            runner_content = get_runner_script_content()

            # Add environment support if enabled
            if self.capture_environment:
                # Create the environment module
                from .runner_env_module import RunnerEnvironmentModule
                RunnerEnvironmentModule.create_module_file(self.nfs_scripts_dir)
                self.logger.info("Created runner environment module")

                # Modify runner script to use environment module
                lines = runner_content.split('\n')

                # Add import after other imports
                import_index = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ')):
                        import_index = i + 1
                    elif line.strip().startswith('def '):
                        break

                lines.insert(import_index, "\n# Environment support\nfrom runner_env import update_object_functions\n")

                # Add update call after "Object loaded successfully"
                for i, line in enumerate(lines):
                    if 'print("Object loaded successfully")' in line:
                        lines.insert(i + 1, "\n        # Apply captured environment\n        update_object_functions(obj)\n")
                        break

                runner_content = '\n'.join(lines)

            # Write the final runner script
            with open(runner_path, 'w') as f:
                f.write(runner_content)
            runner_path.chmod(0o755)

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
            if env_file:
                script_content += f"\npython {runner_script} {object_file} {config_file} $SLURM_ARRAY_TASK_ID {env_file}\n"
            else:
                script_content += f"\npython {runner_script} {object_file} {config_file} $SLURM_ARRAY_TASK_ID\n"
        else:
            if env_file:
                script_content += f"\npython {runner_script} {object_file} {config_file} 0 {env_file}\n"
            else:
                script_content += f"\npython {runner_script} {object_file} {config_file}\n"

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
                       array_jobs: bool = True,
                       sims_per_job: int = 10,
                       wait_for_completion: bool = False,
                       **kwargs) -> Dict[str, Any]:
        """
        Submit an ensemble to SLURM.

        Args:
            ensemble: Ensemble object to run
            n_simulations: Number of simulations to run
            T: Simulation time
            output_dir: Output directory (if None, uses NFS work dir)
            slurm_config: SLURM configuration
            array_jobs: Use array jobs to parallelize across simulations
            sims_per_job: Number of simulations per array job
            wait_for_completion: If True, wait for job to complete and run post-processing
            **kwargs: Additional arguments passed to ensemble.run()

        Returns:
            Dictionary with job_id and other submission info
        """
        if slurm_config is None:
            slurm_config = SlurmConfig()

        # Generate unique job name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"ensemble_{timestamp}"

        # Normalize output directory
        output_dir = self._normalize_output_dir(output_dir, base_name)

        # Prepare configuration
        config = {
            'type': 'ensemble',
            'n_simulations': n_simulations,
            'T': T,
            'sims_per_job': sims_per_job,
            'output_dir': output_dir,
            'parallel': False,  # Disable internal parallelization when using array jobs
            **kwargs
        }

        # Set up array jobs if requested
        if array_jobs:
            n_jobs = (n_simulations + sims_per_job - 1) // sims_per_job
            slurm_config.array_size = n_jobs
            slurm_config.job_name = f"{base_name}_array"
        else:
            slurm_config.cpus_per_task = kwargs.get('max_workers', 4)
            config['parallel'] = True
            slurm_config.job_name = base_name

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
            slurm_config, runner_script, ensemble_file, config_file, array_jobs, env_file
        )

        # Submit job
        job_id = self._submit_job(slurm_script)

        result = {
            'job_id': job_id,
            'job_name': slurm_config.job_name,
            'output_dir': output_dir,
            'n_chunks': n_jobs if array_jobs else 1,
            'array_jobs': array_jobs,
            'config_file': str(config_file),
            'ensemble_file': str(ensemble_file)
        }

        if env_file:
            result['env_file'] = str(env_file)
            result['capture_log'] = str(self.capture_log_dir / f"{slurm_config.job_name}_capture.log")

        if wait_for_completion:
            # Monitor job progress
            job_info = self.monitor.monitor_ensemble(result, show_progress=True)

            if job_info.state == JobState.COMPLETED:
                # Run post-processing
                if array_jobs:
                    # Collect results from chunks
                    results = self.collect_ensemble_results(
                        output_dir, n_jobs, cleanup=True
                    )
                    ensemble.results = results

                    # Run ensemble analysis
                    ensemble_analysis = ensemble.analyze()

                    # Create ensemble visualizations
                    ensemble.visualize(output_dir,
                                     individual_plots=False,
                                     ensemble_plots=True)

                    result['analysis'] = ensemble_analysis
                    result['status'] = 'completed'
            else:
                result['status'] = job_info.state.value
                result['job_info'] = job_info

        return result

    def submit_parameter_scan(self,
                            scan,
                            n_simulations_per_point: int,
                            T: float = 50.0,
                            output_dir: Optional[str] = None,
                            slurm_config: Optional[SlurmConfig] = None,
                            array_jobs: bool = True,
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
            array_jobs: Use array jobs to parallelize across parameter points
            wait_for_completion: If True, wait for job to complete and run post-processing
            **kwargs: Additional arguments passed to scan.run()

        Returns:
            Dictionary with job_id and other submission info
        """
        if slurm_config is None:
            slurm_config = SlurmConfig()

        # Calculate number of parameter points
        import itertools
        param_combinations = list(itertools.product(*scan.scan_params.values()))
        n_points = len(param_combinations)

        # Generate unique job name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"scan_{timestamp}"

        # Normalize output directory
        output_dir = self._normalize_output_dir(output_dir, base_name)

        # Prepare configuration
        config = {
            'type': 'parameter_scan',
            'n_simulations_per_point': n_simulations_per_point,
            'T': T,
            'scan_params': scan.scan_params,
            'base_params': scan.base_params,
            'output_dir': output_dir,
            **kwargs
        }

        # Set up array jobs if requested
        if array_jobs:
            slurm_config.array_size = n_points
            slurm_config.job_name = f"{base_name}_array"
            config['parallel'] = True
            config['max_workers'] = slurm_config.cpus_per_task
            self.logger.info(f"Array jobs: Each task will use {slurm_config.cpus_per_task} CPUs in parallel")
        else:
            slurm_config.job_name = base_name
            if 'max_workers' not in kwargs:
                # Default to number of parameter points, but cap at reasonable number
                config['max_workers'] = min(n_points, slurm_config.cpus_per_task)

            config['parallel'] = True
            config['parallel_mode'] = 'scan'
            self.logger.info(f"Single job: Will parallelize across {n_points} parameter points")

        # Log the configuration
        self.logger.info(f"Parameter scan configuration:")
        self.logger.info(f"  - {n_points} parameter points")
        self.logger.info(f"  - {n_simulations_per_point} simulations per point")
        self.logger.info(f"  - Total simulations: {n_points * n_simulations_per_point}")
        self.logger.info(f"  - Array jobs: {array_jobs}")
        self.logger.info(f"  - CPUs per task: {slurm_config.cpus_per_task}")
        self.logger.info(f"  - Parallel within task: {config.get('parallel', False)}")
        self.logger.info(f"  - Workers per task: {config.get('max_workers', 1)}")

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
            slurm_config, runner_script, scan_file, config_file, array_jobs, env_file
        )

        # Submit job
        job_id = self._submit_job(slurm_script)

        result = {
            'job_id': job_id,
            'job_name': slurm_config.job_name,
            'output_dir': output_dir,
            'n_points': n_points,
            'array_jobs': array_jobs,
            'config_file': str(config_file),
            'scan_file': str(scan_file),
            'scan_params': scan.scan_params
        }

        if env_file:
            result['env_file'] = str(env_file)
            result['capture_log'] = str(self.capture_log_dir / f"{slurm_config.job_name}_capture.log")

        if wait_for_completion:
            # Monitor job progress
            job_info = self.monitor.monitor_scan(result, show_progress=True)

            if job_info.state == JobState.COMPLETED:
                # Run post-processing
                if array_jobs:
                    # Collect results from parameter points
                    scan_results = self.collect_scan_results(
                        output_dir, scan.scan_params, cleanup=False
                    )
                    scan.scan_results = scan_results

                    # Run scan-level analysis
                    scan_analysis = scan.analyze()

                    # Create scan-level visualizations
                    if scan.scan_plot_function:
                        scan.scan_plot_function(scan_results, output_dir)

                    result['analysis'] = scan_analysis
                    result['status'] = 'completed'
            else:
                result['status'] = job_info.state.value
                result['job_info'] = job_info

        return result

    def submit_pipeline(self,
                       pipeline,
                       T: float = 50.0,
                       output_dir: Optional[str] = None,
                       slurm_config: Optional[SlurmConfig] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Submit a single simulation pipeline to SLURM.

        Args:
            pipeline: SimulationPipeline object to run
            T: Simulation time
            output_dir: Output directory (if None, uses NFS work dir)
            slurm_config: SLURM configuration
            **kwargs: Additional arguments passed to pipeline.run_complete()

        Returns:
            Dictionary with job_id and other submission info
        """
        if slurm_config is None:
            slurm_config = SlurmConfig()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slurm_config.job_name = f"pipeline_{timestamp}"

        # Normalize output directory
        output_dir = self._normalize_output_dir(output_dir, slurm_config.job_name)

        # Update SLURM output directory
        slurm_config.slurm_output_dir = str(self.nfs_work_dir / "logs")

        # Prepare configuration
        config = {
            'type': 'pipeline',
            'T': T,
            'output_dir': output_dir,
            **kwargs
        }

        # Serialize pipeline and config to NFS
        pipeline_file, env_file = self._serialize_with_environment(
            pipeline, slurm_config.job_name,
            analysis_functions=pipeline.analysis_functions,
            plot_functions=pipeline.plot_functions
        )

        config_file = self.nfs_input_dir / f"{slurm_config.job_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Create runner script
        runner_script = self._create_runner_script()

        # Create SLURM script
        slurm_script = self._create_slurm_script(
            slurm_config, runner_script, pipeline_file, config_file, False, env_file
        )

        # Submit job
        job_id = self._submit_job(slurm_script)

        result = {
            'job_id': job_id,
            'job_name': slurm_config.job_name,
            'output_dir': output_dir,
            'config_file': str(config_file),
            'pipeline_file': str(pipeline_file)
        }

        if env_file:
            result['env_file'] = str(env_file)
            result['capture_log'] = str(self.capture_log_dir / f"{slurm_config.job_name}_capture.log")

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

    def monitor_and_collect(self, submission_result: Dict[str, Any],
                           show_progress: bool = True) -> Dict[str, Any]:
        """
        Monitor a submitted job and collect results when done.

        Args:
            submission_result: Result from submit_* method
            show_progress: Show progress bar

        Returns:
            Updated submission result with collected data
        """
        # Determine job type and monitor
        if 'n_chunks' in submission_result:
            # Ensemble job
            job_info = self.monitor.monitor_ensemble(submission_result, show_progress)

            if job_info.state == JobState.COMPLETED and submission_result.get('output_dir'):
                # Load the ensemble object
                with open(submission_result['ensemble_file'], 'rb') as f:
                    ensemble = pickle.load(f)

                # Collect and process results
                results = self.collect_ensemble_results(
                    submission_result['output_dir'],
                    submission_result['n_chunks'],
                    cleanup=True
                )
                ensemble.results = results
                ensemble_analysis = ensemble.analyze()

                # Create visualizations
                ensemble.visualize(submission_result['output_dir'],
                                 individual_plots=False,
                                 ensemble_plots=True)

                submission_result['analysis'] = ensemble_analysis
                submission_result['ensemble'] = ensemble

        elif 'n_points' in submission_result:
            # Parameter scan job
            job_info = self.monitor.monitor_scan(submission_result, show_progress)

            if job_info.state == JobState.COMPLETED and submission_result.get('output_dir'):
                # Load the scan object
                with open(submission_result['scan_file'], 'rb') as f:
                    scan = pickle.load(f)

                # Collect results
                scan_results = self.collect_scan_results(
                    submission_result['output_dir'],
                    submission_result['scan_params'],
                    cleanup=False
                )
                scan.scan_results = scan_results
                scan_analysis = scan.analyze()

                # Create visualizations
                if scan.scan_plot_function:
                    scan.scan_plot_function(scan_results, submission_result['output_dir'])

                submission_result['analysis'] = scan_analysis
                submission_result['scan'] = scan
        else:
            # Single pipeline job
            job_info = self.monitor.monitor_job(
                submission_result['job_id'],
                submission_result.get('job_name'),
                show_progress=show_progress
            )

        submission_result['job_info'] = job_info
        submission_result['status'] = job_info.state.value

        # Update tracker if job completed
        job_id = submission_result['job_id']
        if self.job_tracker and job_info.state in [JobState.COMPLETED, JobState.FAILED,
                                                   JobState.CANCELLED, JobState.TIMEOUT]:
            self.job_tracker.complete_job(job_id)

        return submission_result

    def collect_ensemble_results(self,
                               output_dir: str,
                               n_chunks: int,
                               cleanup: bool = True) -> List:
        """
        Collect results from array job ensemble run.

        Args:
            output_dir: Directory containing chunk results
            n_chunks: Number of chunks (array size)
            cleanup: Remove chunk files after collection

        Returns:
            Combined list of simulation results
        """
        all_results = []
        missing_chunks = []

        output_path = Path(output_dir)

        for i in range(n_chunks):
            chunk_file = output_path / f"results_chunk_{i}.pkl"
            marker_file = output_path / f"completed_chunk_{i}.marker"

            if chunk_file.exists() and marker_file.exists():
                with open(chunk_file, 'rb') as f:
                    chunk_results = pickle.load(f)
                    all_results.extend(chunk_results)

                if cleanup:
                    chunk_file.unlink()
                    marker_file.unlink()
            else:
                missing_chunks.append(i)

        if missing_chunks:
            print(f"Warning: Missing chunks: {missing_chunks}")

        return all_results

    def collect_scan_results(self,
                           output_dir: str,
                           scan_params: Dict[str, List],
                           cleanup: bool = False) -> Dict:
        """
        Collect results from array job parameter scan.

        Args:
            output_dir: Base directory containing parameter subdirectories
            scan_params: Original scan parameters
            cleanup: Remove individual result files

        Returns:
            Dictionary mapping parameter tuples to results
        """
        import itertools

        param_names = list(scan_params.keys())
        param_combinations = list(itertools.product(*scan_params.values()))

        scan_results = {}
        missing_points = []

        output_path = Path(output_dir)

        for combination in param_combinations:
            # Reconstruct directory name
            param_str = "_".join([f"{k}_{v:.3g}" if isinstance(v, (int, float)) else f"{k}_{v}"
                                for k, v in zip(param_names, combination)])
            param_dir = output_path / param_str
            results_file = param_dir / "ensemble_results.pkl"
            marker_file = param_dir / "completed.marker"

            if results_file.exists() and marker_file.exists():
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)

                param_key = tuple(zip(param_names, combination))
                scan_results[param_key] = results

                if cleanup:
                    results_file.unlink()
                    marker_file.unlink()
            else:
                missing_points.append(combination)

        if missing_points:
            print(f"Warning: Missing parameter points: {missing_points}")

        return scan_results

    def configure_environment_capture(self,
                                    local_paths: list[str] | None = None,
                                    installed_paths: list[str] | None = None,
                                    excluded_modules: list[str] | None = None) -> None:
        """
        Configure environment capture behavior.

        Args:
            local_paths: Paths to always treat as containing local modules
            installed_paths: Paths to always treat as containing installed packages
            excluded_modules: Module names to never capture

        Example:
            slurm = SlurmPipeline()
            slurm.configure_environment_capture(
                local_paths=['/home/user/my_project', '/home/user/utils'],
                installed_paths=['/home/user/.local/lib/python3.9/site-packages/my_editable_package'],
                excluded_modules=['test_module', 'debug_utils']
            )
        """
        if not hasattr(self, 'env_config'):
            self.env_config = types.SimpleNamespace()

        self.env_config.local_paths = local_paths or []
        self.env_config.installed_paths = installed_paths or []
        self.env_config.excluded_modules = excluded_modules or []

        print("Environment capture configured:")
        if self.env_config.local_paths:
            print(f"  Force local paths: {self.env_config.local_paths}")
        if self.env_config.installed_paths:
            print(f"  Force installed paths: {self.env_config.installed_paths}")
        if self.env_config.excluded_modules:
            print(f"  Excluded modules: {self.env_config.excluded_modules}")

    def preview_environment_capture(self, obj: Any,
                                  analysis_functions: dict[str, Callable] | None = None,
                                  plot_functions: dict[str, Callable] | None = None) -> dict[str, list[str]]:
        """
        Preview what would be captured without actually submitting.

        Returns:
            Dictionary with 'local_modules', 'installed_modules', 'inline_functions'
        """
        # Create a temporary logger
        import io
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger('preview')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Create temporary capture
        env_capture = EnvironmentCapture(logger)

        # Apply configuration if exists
        if hasattr(self, 'env_config'):
            for path in self.env_config.local_paths:
                env_capture.add_local_path(path)
            for path in self.env_config.installed_paths:
                env_capture.add_installed_path(path)
            for module in self.env_config.excluded_modules:
                env_capture.exclude_module(module)

        # Capture environment
        env_capture.capture_current_environment()
        env_capture.symbol_tracker.analyze_object(obj, "preview_object")
        env_capture.capture_local_modules(obj)

        # Capture functions
        all_functions = {}
        if analysis_functions:
            all_functions.update(analysis_functions)
        if plot_functions:
            all_functions.update(plot_functions)

        env_capture.capture_inline_functions(all_functions)

        # Return summary
        return {
            'local_modules': list(env_capture.local_modules.keys()),
            'installed_modules': [
                f"{name} ({info['reason']})"
                for name, info in env_capture.installed_modules.items()
            ],
            'inline_functions': list(env_capture.inline_functions.keys()),
            'excluded_modules': list(env_capture.excluded_modules)
        }
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
