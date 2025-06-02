"""
Debugging utilities for SLURM pipeline to diagnose missing results.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pickle


def check_job_output_files(job_id: str, slurm_output_dir: str = None) -> Dict[str, Any]:
    """
    Check SLURM output files for a job.

    Args:
        job_id: SLURM job ID
        slurm_output_dir: Directory containing SLURM output files

    Returns:
        Dictionary with file information
    """
    result = {
        'job_id': job_id,
        'output_files': [],
        'error_files': [],
        'file_contents': {}
    }

    if slurm_output_dir:
        output_path = Path(slurm_output_dir)
        if output_path.exists():
            # Look for output files
            for pattern in [f"*{job_id}*.out", f"*{job_id}*.err"]:
                for file in output_path.glob(pattern):
                    if file.suffix == '.out':
                        result['output_files'].append(str(file))
                    else:
                        result['error_files'].append(str(file))

                    # Read last 50 lines
                    try:
                        with open(file, 'r') as f:
                            lines = f.readlines()
                            result['file_contents'][str(file)] = {
                                'total_lines': len(lines),
                                'last_lines': lines[-50:] if len(lines) > 50 else lines
                            }
                    except Exception as e:
                        result['file_contents'][str(file)] = {'error': str(e)}

    return result


def check_output_directory(output_dir: str, expected_type: str = 'ensemble') -> Dict[str, Any]:
    """
    Check the contents of an output directory.

    Args:
        output_dir: Directory to check
        expected_type: Type of output expected ('ensemble', 'scan', 'pipeline')

    Returns:
        Dictionary with directory information
    """
    result = {
        'directory': output_dir,
        'exists': False,
        'files': [],
        'subdirs': [],
        'markers': [],
        'results': [],
        'missing': []
    }

    output_path = Path(output_dir)
    if not output_path.exists():
        return result

    result['exists'] = True

    # List all contents
    for item in output_path.iterdir():
        if item.is_file():
            result['files'].append(item.name)
            if item.suffix == '.marker':
                result['markers'].append(item.name)
            elif item.suffix == '.pkl':
                result['results'].append(item.name)
        elif item.is_dir():
            result['subdirs'].append(item.name)

    # Check for expected files based on type
    if expected_type == 'ensemble':
        # Check for chunk files
        chunk_files = [f for f in result['files'] if f.startswith('results_chunk_')]
        result['chunk_files'] = chunk_files

        # Check which chunks are missing
        if chunk_files:
            chunk_numbers = []
            for f in chunk_files:
                try:
                    num = int(f.split('_')[2].split('.')[0])
                    chunk_numbers.append(num)
                except:
                    pass

            # Find missing chunks
            if chunk_numbers:
                max_chunk = max(chunk_numbers)
                for i in range(max_chunk + 1):
                    if i not in chunk_numbers:
                        result['missing'].append(f'chunk_{i}')

    elif expected_type == 'scan':
        # Check subdirectories for parameter points
        for subdir in result['subdirs']:
            subpath = output_path / subdir
            subdir_info = {
                'name': subdir,
                'files': [],
                'has_marker': False,
                'has_results': False
            }

            for item in subpath.iterdir():
                if item.is_file():
                    subdir_info['files'].append(item.name)
                    if item.name == 'completed.marker':
                        subdir_info['has_marker'] = True
                    elif item.name == 'ensemble_results.pkl':
                        subdir_info['has_results'] = True

            result[f'subdir_{subdir}'] = subdir_info

    return result


def check_array_job_tasks(job_id: str) -> Dict[str, Any]:
    """
    Check the status of array job tasks.

    Args:
        job_id: SLURM array job ID

    Returns:
        Dictionary with task information
    """
    result = {
        'job_id': job_id,
        'tasks': {},
        'summary': {}
    }

    # Get detailed information about array tasks
    cmd = ['sacct', '-j', job_id, '--format=JobID,State,ExitCode,Start,End,Elapsed', '--parsable2']

    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = output.stdout.strip().split('\n')

        if len(lines) > 1:  # Skip header
            for line in lines[1:]:
                parts = line.split('|')
                if len(parts) >= 6:
                    task_id = parts[0]
                    state = parts[1]
                    exit_code = parts[2]

                    # Skip batch and extern jobs
                    if '.batch' in task_id or '.extern' in task_id:
                        continue

                    result['tasks'][task_id] = {
                        'state': state,
                        'exit_code': exit_code,
                        'start': parts[3],
                        'end': parts[4],
                        'elapsed': parts[5]
                    }

                    # Update summary
                    if state not in result['summary']:
                        result['summary'][state] = 0
                    result['summary'][state] += 1

    except subprocess.CalledProcessError as e:
        result['error'] = f"Failed to get job info: {e}"

    return result


def diagnose_missing_results(submission_result: Dict[str, Any],
                           slurm_work_dir: str = "/home/adrs0061/cluster/slurm_pipeline") -> None:
    """
    Comprehensive diagnosis of why results might be missing.

    Args:
        submission_result: Result from submit_* method
        slurm_work_dir: Base SLURM work directory
    """
    print(f"\n{'='*60}")
    print("DIAGNOSING MISSING RESULTS")
    print(f"{'='*60}")

    job_id = submission_result.get('job_id', 'unknown')
    job_name = submission_result.get('job_name', 'unknown')
    output_dir = submission_result.get('output_dir', 'unknown')

    print(f"Job ID: {job_id}")
    print(f"Job Name: {job_name}")
    print(f"Output Directory: {output_dir}")

    # 1. Check SLURM output files
    print(f"\n1. SLURM Output Files")
    print("-" * 30)
    slurm_logs = Path(slurm_work_dir) / "logs"
    job_output = check_job_output_files(job_id, str(slurm_logs))

    print(f"Output files found: {len(job_output['output_files'])}")
    print(f"Error files found: {len(job_output['error_files'])}")

    # Show last few lines of output
    for file, content in job_output['file_contents'].items():
        if 'error' not in content:
            print(f"\nLast lines of {Path(file).name}:")
            print("-" * 30)
            for line in content['last_lines'][-10:]:
                print(line.rstrip())

    # 2. Check array job tasks (if applicable)
    if submission_result.get('array_jobs'):
        print(f"\n2. Array Job Task Status")
        print("-" * 30)
        task_info = check_array_job_tasks(job_id)

        if 'error' in task_info:
            print(f"Error: {task_info['error']}")
        else:
            print(f"Total tasks: {len(task_info['tasks'])}")
            print("Task summary:")
            for state, count in task_info['summary'].items():
                print(f"  {state}: {count}")

            # Show failed tasks
            failed_tasks = [tid for tid, info in task_info['tasks'].items()
                           if info['state'] != 'COMPLETED']
            if failed_tasks:
                print(f"\nFailed/incomplete tasks: {failed_tasks}")

    # 3. Check output directory
    print(f"\n3. Output Directory Contents")
    print("-" * 30)

    if 'ensemble' in job_name:
        dir_info = check_output_directory(output_dir, 'ensemble')
    elif 'scan' in job_name:
        dir_info = check_output_directory(output_dir, 'scan')
    else:
        dir_info = check_output_directory(output_dir, 'pipeline')

    if not dir_info['exists']:
        print(f"ERROR: Output directory does not exist!")
    else:
        print(f"Files: {len(dir_info['files'])}")
        print(f"Subdirectories: {len(dir_info['subdirs'])}")
        print(f"Marker files: {len(dir_info['markers'])}")
        print(f"Result files: {len(dir_info['results'])}")

        if dir_info['files']:
            print("\nFiles found:")
            for f in sorted(dir_info['files'])[:20]:  # Show first 20
                print(f"  - {f}")
            if len(dir_info['files']) > 20:
                print(f"  ... and {len(dir_info['files']) - 20} more")

        if dir_info.get('missing'):
            print(f"\nMissing items: {dir_info['missing']}")

    # 4. Check pickle files
    print(f"\n4. Configuration Files")
    print("-" * 30)

    config_file = submission_result.get('config_file')
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Config type: {config.get('type')}")
        print(f"Output dir in config: {config.get('output_dir')}")

        if config.get('type') == 'ensemble':
            print(f"N simulations: {config.get('n_simulations')}")
            print(f"Sims per job: {config.get('sims_per_job')}")

    # 5. File system sync check
    print(f"\n5. File System Check")
    print("-" * 30)
    print("Running 'sync' to ensure file system is synchronized...")
    subprocess.run(['sync'], check=False)

    # Re-check directory after sync
    import time
    time.sleep(2)

    dir_info_after = check_output_directory(output_dir, 'ensemble')
    if dir_info_after['files'] != dir_info['files']:
        print("Files changed after sync!")
        print(f"Before: {len(dir_info['files'])} files")
        print(f"After: {len(dir_info_after['files'])} files")
    else:
        print("No change after sync")

    print(f"\n{'='*60}\n")


def find_trace_files(trace_dir: str = "slurm_traces") -> Dict[str, List[str]]:
    """
    Find and categorize trace files.

    Args:
        trace_dir: Directory containing trace files

    Returns:
        Dictionary of trace files by category
    """
    traces = {
        'simulations': [],
        'analyses': [],
        'plots': [],
        'ensembles': [],
        'scans': []
    }

    trace_path = Path(trace_dir)
    if not trace_path.exists():
        return traces

    for trace_file in trace_path.glob("*.json"):
        name = trace_file.name
        if name.startswith('sim_'):
            traces['simulations'].append(str(trace_file))
        elif name.startswith('analysis_'):
            traces['analyses'].append(str(trace_file))
        elif name.startswith('plot_'):
            traces['plots'].append(str(trace_file))
        elif name.startswith('ensemble_'):
            traces['ensembles'].append(str(trace_file))
        elif name.startswith('scan_'):
            traces['scans'].append(str(trace_file))

    return traces


def quick_check(job_id: str, output_dir: str) -> None:
    """
    Quick check of job results.

    Args:
        job_id: SLURM job ID
        output_dir: Expected output directory
    """
    print(f"\nQuick check for job {job_id}:")
    print(f"Output dir: {output_dir}")

    # Check if directory exists
    if not Path(output_dir).exists():
        print("❌ Output directory does not exist")
        return

    # Count files
    output_path = Path(output_dir)
    files = list(output_path.rglob("*"))
    markers = list(output_path.glob("**/*.marker"))
    results = list(output_path.glob("**/*.pkl"))

    print(f"✓ Directory exists")
    print(f"  Total files: {len(files)}")
    print(f"  Marker files: {len(markers)}")
    print(f"  Result files: {len(results)}")

    # Check SLURM status
    cmd = ['sacct', '-j', job_id, '--format=State', '--noheader', '-X']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        state = result.stdout.strip()
        print(f"  Job state: {state}")
