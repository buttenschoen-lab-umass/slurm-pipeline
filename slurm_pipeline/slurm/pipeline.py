"""
Main SLURM pipeline class for submitting and managing simulation jobs.

This version properly handles NFS directories for cluster-wide access.
"""

import os
import sys
import json
import pickle
import subprocess
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime

from .config import SlurmConfig
from .monitor import SlurmMonitor, JobInfo, JobState


class SlurmPipeline:
    """Wrapper to run pipeline objects on SLURM."""

    def __init__(self,
                 nfs_work_dir: str = "/home/adrs0061/cluster/slurm_pipeline",
                 monitor_interval: int = 10):
        """
        Initialize SLURM pipeline wrapper.

        Args:
            nfs_work_dir: NFS directory accessible from all nodes (can use $USER)
            monitor_interval: Seconds between status checks for monitoring
        """
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

        self.monitor = SlurmMonitor(check_interval=monitor_interval)
        self.runner_script = None

    def _serialize_object(self, obj: Any, name: str) -> Path:
        """Serialize a pipeline object to NFS directory."""
        filepath = self.nfs_input_dir / f"{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        return filepath

    def _create_runner_script(self) -> Path:
        """Create the runner script in NFS directory."""
        if self.runner_script is None:
            runner_path = self.nfs_scripts_dir / "runner.py"

            # Write the runner script content
            runner_content = '''#!/usr/bin/env python
"""
SLURM runner script for simulation pipeline objects.
This script is executed on compute nodes to run serialized pipeline objects.
"""

import sys
import pickle
import json
from pathlib import Path
import traceback
import os


def run_ensemble_array(obj, config, array_index):
    """Run a chunk of ensemble simulations."""
    n_sims = config['n_simulations']
    sims_per_job = config.get('sims_per_job', 1)
    start_idx = array_index * sims_per_job
    end_idx = min(start_idx + sims_per_job, n_sims)

    print(f"Array task {array_index}: Running simulations {start_idx}-{end_idx-1}")

    # Update starting_sim_id for this chunk
    obj.starting_sim_id = start_idx

    # Run subset of simulations
    obj.run(
        n_simulations=end_idx - start_idx,
        T=config['T'],
        parallel=config.get('parallel', False),
        output_dir=config.get('output_dir'),
        create_individual_plots=config.get('create_individual_plots', True),
        create_individual_animations=config.get('create_individual_animations', True),
        max_individual_plots=config.get('max_individual_plots')
    )

    # Save results for this chunk
    output_path = Path(config['output_dir'])
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"results_chunk_{array_index}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(obj.results, f)

    # Save completion marker
    marker_file = output_path / f"completed_chunk_{array_index}.marker"
    marker_file.touch()

    print(f"Array task {array_index}: Completed successfully")


def run_scan_array(obj, config, array_index):
    """Run a single parameter point from scan."""
    import itertools

    # Reconstruct parameter combinations
    param_names = list(config['scan_params'].keys())
    param_values = [config['scan_params'][k] for k in param_names]
    param_combinations = list(itertools.product(*param_values))

    if array_index >= len(param_combinations):
        print(f"Array index {array_index} exceeds number of parameter combinations")
        return

    # Get parameter combination for this job
    param_combination = param_combinations[array_index]

    print(f"Array task {array_index}: Running parameter point {dict(zip(param_names, param_combination))}")

    # Update base parameters
    params = config['base_params'].copy()
    for name, value in zip(param_names, param_combination):
        params[name] = value

    # Create output directory for this parameter point
    param_str = "_".join([f"{k}_{v:.3g}" if isinstance(v, (int, float)) else f"{k}_{v}"
                        for k, v in zip(param_names, param_combination)])
    output_dir = Path(config['output_dir']) / param_str
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import Ensemble class - try multiple strategies
    try:
        # Strategy 2: Direct import
        from slurm_pipeline import Ensemble
    except:
        # Strategy 3: Add parent directory to path and try again
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from slurm_pipeline import Ensemble

    # Create ensemble for this parameter point
    ensemble = Ensemble(
        obj.model_type, params, obj.integrator_params,
        obj.analysis_functions, obj.plot_functions, obj.animation_function,
        obj.ensemble_analysis_functions, obj.ensemble_plot_functions
    )

    # Run ensemble
    ensemble.run(
        n_simulations=config['n_simulations_per_point'],
        T=config['T'],
        output_dir=str(output_dir),
        parallel=config.get('parallel', False),
        create_individual_plots=config.get('create_individual_plots', True),
        create_individual_animations=config.get('create_individual_animations', True),
        max_individual_plots=config.get('max_individual_plots')
    )

    # Run ensemble analysis
    ensemble_analysis = ensemble.analyze()

    # Create ensemble visualizations if requested
    if config.get('create_ensemble_plots', True):
        ensemble.visualize(str(output_dir), individual_plots=False, ensemble_plots=True)

    # Save results
    results_file = output_dir / "ensemble_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump({
            'ensemble': ensemble,
            'analysis': ensemble_analysis
        }, f)

    # Save completion marker
    marker_file = output_dir / "completed.marker"
    marker_file.touch()

    print(f"Array task {array_index}: Completed parameter point successfully")


def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: runner.py <pickle_file> <config_file> [array_index]")
        sys.exit(1)

    pickle_file = Path(sys.argv[1])
    config_file = Path(sys.argv[2])
    array_index = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # Log execution info
    print(f"Runner started on host: {os.uname().nodename}")
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"SLURM_ARRAY_TASK_ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}")

    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Load pickled object
        with open(pickle_file, 'rb') as f:
            obj = pickle.load(f)

        # Determine object type and run appropriately
        obj_type = config['type']

        if obj_type == 'ensemble':
            if array_index is not None:
                run_ensemble_array(obj, config, array_index)
            else:
                # Run complete ensemble with post-processing
                print("Running complete ensemble")
                obj.run(
                    n_simulations=config['n_simulations'],
                    T=config['T'],
                    parallel=config.get('parallel', True),
                    output_dir=config.get('output_dir'),
                    create_individual_plots=config.get('create_individual_plots', True),
                    create_individual_animations=config.get('create_individual_animations', True),
                    max_individual_plots=config.get('max_individual_plots')
                )

                # Run analysis
                ensemble_analysis = obj.analyze()

                # Create visualizations
                if config.get('output_dir') and config.get('create_ensemble_plots', True):
                    obj.visualize(config['output_dir'],
                                individual_plots=False,
                                ensemble_plots=True)

                # Save final results
                if config.get('output_dir'):
                    results_file = Path(config['output_dir']) / "ensemble_complete.pkl"
                    with open(results_file, 'wb') as f:
                        pickle.dump({
                            'ensemble': obj,
                            'analysis': ensemble_analysis,
                            'results': obj.results
                        }, f)

        elif obj_type == 'parameter_scan':
            if array_index is not None:
                run_scan_array(obj, config, array_index)
            else:
                # Run complete parameter scan
                print("Running complete parameter scan")
                obj.run(
                    n_simulations_per_point=config['n_simulations_per_point'],
                    T=config['T'],
                    parallel=config.get('parallel', True),
                    parallel_mode=config.get('parallel_mode', 'auto'),
                    output_dir=config.get('output_dir'),
                    create_individual_plots=config.get('create_individual_plots', True),
                    create_individual_animations=config.get('create_individual_animations', True),
                    max_individual_plots=config.get('max_individual_plots')
                )

                # Run scan-level analysis
                scan_analysis = obj.analyze()

                # Create scan-level visualizations
                if config.get('output_dir'):
                    obj.visualize(config['output_dir'],
                                create_ensemble_plots=config.get('create_ensemble_plots', True))

                # Save final results
                if config.get('output_dir'):
                    results_file = Path(config['output_dir']) / "scan_complete.pkl"
                    with open(results_file, 'wb') as f:
                        pickle.dump({
                            'scan': obj,
                            'analysis': scan_analysis,
                            'results': obj.scan_results
                        }, f)

        elif obj_type == 'pipeline':
            # Run single simulation pipeline
            print("Running single simulation pipeline")
            result = obj.run(
                T=config['T'],
                initial_state=config.get('initial_state'),
                progress=config.get('progress', False)
            )

            # Analyze
            result = obj.analyze(result)

            # Visualize
            if config.get('output_dir'):
                result = obj.visualize(
                    result,
                    config['output_dir'],
                    plots=config.get('create_plots', True),
                    animation=config.get('create_animation', True)
                )

                # Save data
                if config.get('save_data', True) and obj.save_function:
                    data_path = Path(config['output_dir']) / "simulation_data.npz"
                    obj.save(result, str(data_path))

        print("Job completed successfully")

    except Exception as e:
        print(f"Job failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
'''

            with open(runner_path, 'w') as f:
                f.write(runner_content)
            runner_path.chmod(0o755)
            self.runner_script = runner_path

        return self.runner_script

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

        # Set up output directory
        if output_dir is None:
            output_dir = str(self.nfs_work_dir / "outputs" / base_name)
        output_dir = os.path.expandvars(output_dir)

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
        ensemble_file = self._serialize_object(ensemble, slurm_config.job_name)
        config_file = self.nfs_input_dir / f"{slurm_config.job_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Create runner script
        runner_script = self._create_runner_script()

        # Create SLURM script
        slurm_script = self._create_slurm_script(
            slurm_config, runner_script, ensemble_file, config_file, array_jobs
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

        # Set up output directory
        if output_dir is None:
            output_dir = str(self.nfs_work_dir / "outputs" / base_name)
        output_dir = os.path.expandvars(output_dir)

        # Prepare configuration
        config = {
            'type': 'parameter_scan',
            'n_simulations_per_point': n_simulations_per_point,
            'T': T,
            'scan_params': scan.scan_params,
            'base_params': scan.base_params,
            'output_dir': output_dir,
            'parallel': False,  # Disable internal parallelization
            **kwargs
        }

        # Set up array jobs if requested
        if array_jobs:
            slurm_config.array_size = n_points
            slurm_config.job_name = f"{base_name}_array"
            # Increase resources per job since each runs an ensemble
            slurm_config.cpus_per_task = min(n_simulations_per_point, 8)
            #slurm_config.mem = "8G"
        else:
            slurm_config.cpus_per_task = kwargs.get('max_workers', 8)
            config['parallel'] = True
            config['parallel_mode'] = 'scan'
            slurm_config.job_name = base_name

        # Update SLURM output directory
        slurm_config.slurm_output_dir = str(self.nfs_work_dir / "logs")

        # Serialize scan and config to NFS
        scan_file = self._serialize_object(scan, slurm_config.job_name)
        config_file = self.nfs_input_dir / f"{slurm_config.job_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Create runner script
        runner_script = self._create_runner_script()

        # Create SLURM script
        slurm_script = self._create_slurm_script(
            slurm_config, runner_script, scan_file, config_file, array_jobs
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

        # Set up output directory
        if output_dir is None:
            output_dir = str(self.nfs_work_dir / "outputs" / slurm_config.job_name)
        output_dir = os.path.expandvars(output_dir)

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
        pipeline_file = self._serialize_object(pipeline, slurm_config.job_name)
        config_file = self.nfs_input_dir / f"{slurm_config.job_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Create runner script
        runner_script = self._create_runner_script()

        # Create SLURM script
        slurm_script = self._create_slurm_script(
            slurm_config, runner_script, pipeline_file, config_file, False
        )

        # Submit job
        job_id = self._submit_job(slurm_script)

        return {
            'job_id': job_id,
            'job_name': slurm_config.job_name,
            'output_dir': output_dir,
            'config_file': str(config_file),
            'pipeline_file': str(pipeline_file)
        }

    def _create_slurm_script(self,
                           config: SlurmConfig,
                           runner_script: Path,
                           object_file: Path,
                           config_file: Path,
                           array_jobs: bool) -> Path:
        """Create SLURM submission script."""
        script_content = config.to_sbatch_header()

        # Add Python command
        if array_jobs:
            script_content += f"\npython {runner_script} {object_file} {config_file} $SLURM_ARRAY_TASK_ID\n"
        else:
            script_content += f"\npython {runner_script} {object_file} {config_file}\n"

        # Save script to NFS scripts directory
        script_path = self.nfs_scripts_dir / f"{config.job_name}.sbatch"
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make script executable
        script_path.chmod(0o755)

        return script_path

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
