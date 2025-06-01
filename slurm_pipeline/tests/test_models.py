"""
Test models and analysis functions for SLURM pipeline testing.

These models are designed to:
1. Track where and when they execute
2. Take a predictable amount of time
3. Write traceable outputs
4. Test the analysis/visualization pipeline
"""

import os
import time
import socket
import numpy as np
from datetime import datetime
from pathlib import Path
import json


class TestModel:
    """
    Simple test model that tracks execution information.
    """

    def __init__(self, work_time: float = 30.0, complexity: float = 1.0,
                 test_param: float = 1.0, write_trace: bool = True):
        """
        Args:
            work_time: Seconds to spend "computing" per simulation
            complexity: Multiplier for array size
            test_param: Test parameter to vary
            write_trace: Write execution trace files
        """
        self.work_time = work_time
        self.complexity = complexity
        self.test_param = test_param
        self.write_trace = write_trace
        self.n = int(100 * complexity)  # State size

        # Execution tracking
        self.execution_info = {
            'hostname': socket.gethostname(),
            'pid': os.getpid(),
            'init_time': datetime.now().isoformat(),
            'test_param': test_param
        }

    def random_ic(self):
        """Generate random initial conditions."""
        return np.random.randn(self.n)

    def simulate(self, T, initial_state, dt=0.1, progress=False):
        """
        Simulate the model, spending the specified work_time.
        """
        start_time = time.time()

        # Create time array
        t = np.arange(0, T, dt)
        nt = len(t)

        # Pre-allocate solution
        solution = np.zeros((nt, self.n))
        solution[0] = initial_state

        # Track simulation info
        sim_info = {
            'hostname': socket.gethostname(),
            'pid': os.getpid(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'local'),
            'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID', 'none'),
            'start_time': datetime.now().isoformat(),
            'T': T,
            'dt': dt,
            'test_param': self.test_param
        }

        # Write trace file if requested
        if self.write_trace:
            trace_dir = Path("slurm_traces")
            trace_dir.mkdir(exist_ok=True)

            trace_file = trace_dir / f"sim_{sim_info['slurm_job_id']}_{sim_info['slurm_array_task_id']}_{os.getpid()}_{int(time.time()*1000)}.json"
            with open(trace_file, 'w') as f:
                json.dump(sim_info, f, indent=2)

        # Simulate with controlled computation time
        work_per_step = self.work_time / nt

        for i in range(1, nt):
            # Do some "work"
            step_start = time.time()

            # Simple dynamics with test_param influence
            solution[i] = solution[i-1] + dt * (
                -0.1 * solution[i-1] +
                self.test_param * np.sin(t[i]) +
                0.1 * np.random.randn(self.n)
            )

            # Ensure we spend the right amount of time
            while time.time() - step_start < work_per_step:
                # Busy work
                _ = np.sum(solution[i]**2)

        actual_time = time.time() - start_time

        # Add timing info to trace
        if self.write_trace:
            sim_info['end_time'] = datetime.now().isoformat()
            sim_info['actual_compute_time'] = actual_time
            sim_info['requested_compute_time'] = self.work_time

            # Update trace file
            with open(trace_file, 'w') as f:
                json.dump(sim_info, f, indent=2)

        return t, solution


def analyze_test_execution(time, solution, model):
    """Analysis function that tracks execution."""
    analysis_info = {
        'hostname': socket.gethostname(),
        'pid': os.getpid(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'local'),
        'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID', 'none'),
        'analysis_time': datetime.now().isoformat(),
        'mean_value': float(np.mean(solution)),
        'std_value': float(np.std(solution)),
        'final_norm': float(np.linalg.norm(solution[-1])),
        'test_param': model.test_param
    }

    # Write analysis trace
    trace_dir = Path("slurm_traces")
    trace_dir.mkdir(exist_ok=True)

    trace_file = trace_dir / f"analysis_{analysis_info['slurm_job_id']}_{analysis_info['slurm_array_task_id']}_{os.getpid()}_{int(time.time()*1000)}.json"
    with open(trace_file, 'w') as f:
        json.dump(analysis_info, f, indent=2)

    return analysis_info


def plot_test_result(time, solution, model, save_path):
    """Simple plot function that tracks execution."""
    import matplotlib.pyplot as plt

    plot_info = {
        'hostname': socket.gethostname(),
        'pid': os.getpid(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'local'),
        'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID', 'none'),
        'plot_time': datetime.now().isoformat(),
        'save_path': str(save_path),
        'test_param': model.test_param
    }

    # Create simple plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot a few trajectories
    for i in range(min(5, solution.shape[1])):
        ax.plot(time, solution[:, i], alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Test Model (param={model.test_param}, host={plot_info["hostname"]})')
    ax.grid(True, alpha=0.3)

    # Add execution info to plot
    info_text = f"Job: {plot_info['slurm_job_id']}\nTask: {plot_info['slurm_array_task_id']}\nPID: {plot_info['pid']}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # Write plot trace
    trace_dir = Path("slurm_traces")
    trace_dir.mkdir(exist_ok=True)

    trace_file = trace_dir / f"plot_{plot_info['slurm_job_id']}_{plot_info['slurm_array_task_id']}_{os.getpid()}_{int(time.time()*1000)}.json"
    with open(trace_file, 'w') as f:
        json.dump(plot_info, f, indent=2)

    return fig


def ensemble_analysis_test(results):
    """Ensemble analysis that tracks execution."""
    ensemble_info = {
        'hostname': socket.gethostname(),
        'pid': os.getpid(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'local'),
        'ensemble_analysis_time': datetime.now().isoformat(),
        'n_simulations': len(results),
        'sim_hosts': [],
        'sim_jobs': [],
        'test_params': []
    }

    # Collect info from individual analyses
    all_means = []
    for result in results:
        if result.analysis and 'execution_info' in result.analysis:
            info = result.analysis['execution_info']
            ensemble_info['sim_hosts'].append(info.get('hostname', 'unknown'))
            ensemble_info['sim_jobs'].append(info.get('slurm_job_id', 'unknown'))
            ensemble_info['test_params'].append(info.get('test_param', 'unknown'))
            all_means.append(info.get('mean_value', 0))

    # Calculate ensemble statistics
    ensemble_info['ensemble_mean'] = float(np.mean(all_means)) if all_means else 0
    ensemble_info['ensemble_std'] = float(np.std(all_means)) if all_means else 0
    ensemble_info['unique_hosts'] = list(set(ensemble_info['sim_hosts']))
    ensemble_info['unique_jobs'] = list(set(ensemble_info['sim_jobs']))

    # Write ensemble trace
    trace_dir = Path("slurm_traces")
    trace_dir.mkdir(exist_ok=True)

    trace_file = trace_dir / f"ensemble_{ensemble_info['slurm_job_id']}_{os.getpid()}_{int(time.time()*1000)}.json"
    with open(trace_file, 'w') as f:
        json.dump(ensemble_info, f, indent=2)

    return ensemble_info


def scan_analysis_test(scan_results):
    """Parameter scan analysis that tracks execution."""
    scan_info = {
        'hostname': socket.gethostname(),
        'pid': os.getpid(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'local'),
        'scan_analysis_time': datetime.now().isoformat(),
        'n_parameter_points': len(scan_results),
        'parameter_values': {},
        'ensemble_means': {}
    }

    # Analyze each parameter point
    for param_key, results in scan_results.items():
        param_str = str(param_key)

        if 'analysis' in results:
            ensemble_analysis = results['analysis']
            scan_info['parameter_values'][param_str] = dict(param_key)
            scan_info['ensemble_means'][param_str] = ensemble_analysis.get('ensemble_mean', 0)

    # Write scan trace
    trace_dir = Path("slurm_traces")
    trace_dir.mkdir(exist_ok=True)

    trace_file = trace_dir / f"scan_{scan_info['slurm_job_id']}_{os.getpid()}_{int(time.time()*1000)}.json"
    with open(trace_file, 'w') as f:
        json.dump(scan_info, f, indent=2)

    return scan_info


# Analysis and plotting function dictionaries
test_analysis_functions = {
    'execution_info': analyze_test_execution
}

test_plot_functions = {
    'trajectories': plot_test_result
}

test_ensemble_analysis = {
    'ensemble_execution': ensemble_analysis_test
}

test_scan_analysis = scan_analysis_test
