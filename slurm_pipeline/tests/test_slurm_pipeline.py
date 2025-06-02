"""
Test suite for SLURM pipeline functionality.

Run these tests to verify:
1. Jobs execute on compute nodes
2. Work distribution is correct
3. Results are collected properly
4. Analysis pipeline works end-to-end
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime

# Handle imports based on how the script is run
if __name__ == "__main__":
    # Direct execution - add parent directories to path
    current_dir = Path(__file__).resolve().parent
    package_dir = current_dir.parent.parent  # Go up to package root
    sys.path.insert(0, str(package_dir))

    # Use absolute imports
    from slurm_pipeline.slurm import SlurmConfig, SlurmPipeline
    from slurm_pipeline.slurm.debug_utils import diagnose_missing_results, quick_check
    from slurm_pipeline.core import Ensemble, ParameterScan, SimulationPipeline
    from slurm_pipeline.tests.test_models import (
        TestModel,
        test_analysis_functions,
        test_plot_functions,
        test_ensemble_analysis,
        test_scan_analysis
    )
else:
    # Module execution - use relative imports
    from ..slurm import SlurmConfig, SlurmPipeline
    from ..slurm.debug_utils import diagnose_missing_results, quick_check
    from ..core import Ensemble, ParameterScan, SimulationPipeline
    from .test_models import (
        TestModel,
        test_analysis_functions,
        test_plot_functions,
        test_ensemble_analysis,
        test_scan_analysis
    )


def clean_test_directories():
    """Clean up test directories."""
    # Clean local trace directory
    if Path('slurm_traces').exists():
        shutil.rmtree('slurm_traces')

    # Note: We don't clean the NFS test_results directory here
    # as it may contain results we want to inspect
    print("Cleaned local trace directories")


def analyze_traces(trace_dir="slurm_traces", nfs_outputs=None):
    """
    Analyze execution traces to verify distribution.

    Args:
        trace_dir: Local directory with trace files
        nfs_outputs: NFS outputs directory to check for traces
    """
    # First check local traces
    trace_path = Path(trace_dir)
    traces_found = False

    if trace_path.exists():
        print(f"Checking local traces in {trace_dir}")
        traces = analyze_trace_directory(trace_path)
        if any(traces.values()):
            traces_found = True
            print_trace_analysis(traces)
    else:
        print(f"No local traces found in {trace_dir}")

    # Then check NFS directory if provided
    if nfs_outputs:
        nfs_trace_path = Path(nfs_outputs) / "slurm_traces"
        if nfs_trace_path.exists():
            print(f"\nChecking NFS traces in {nfs_trace_path}")
            nfs_traces = analyze_trace_directory(nfs_trace_path)
            if any(nfs_traces.values()):
                traces_found = True
                print_trace_analysis(nfs_traces)

        # Also check for traces in output directories
        for output_dir in Path(nfs_outputs).glob("test_results/*"):
            if output_dir.is_dir():
                trace_subdir = output_dir / "slurm_traces"
                if trace_subdir.exists():
                    print(f"\nChecking traces in {trace_subdir}")
                    sub_traces = analyze_trace_directory(trace_subdir)
                    if any(sub_traces.values()):
                        traces_found = True
                        print_trace_analysis(sub_traces)

    if not traces_found:
        print("\nNo execution traces found anywhere!")
        print("This might indicate the jobs didn't run or trace writing is disabled.")

    return traces_found


def analyze_trace_directory(trace_path: Path) -> dict:
    """Analyze traces in a specific directory."""
    traces = {
        'simulations': [],
        'analyses': [],
        'plots': [],
        'ensembles': [],
        'scans': []
    }

    # Collect all trace files
    for trace_file in trace_path.glob("*.json"):
        with open(trace_file, 'r') as f:
            data = json.load(f)

        if trace_file.name.startswith('sim_'):
            traces['simulations'].append(data)
        elif trace_file.name.startswith('analysis_'):
            traces['analyses'].append(data)
        elif trace_file.name.startswith('plot_'):
            traces['plots'].append(data)
        elif trace_file.name.startswith('ensemble_'):
            traces['ensembles'].append(data)
        elif trace_file.name.startswith('scan_'):
            traces['scans'].append(data)

    return traces


def print_trace_analysis(traces: dict):
    """Print analysis of trace data."""
    print(f"\n{'='*60}")
    print("EXECUTION TRACE ANALYSIS")
    print(f"{'='*60}")

    print(f"\nSimulations: {len(traces['simulations'])}")
    if traces['simulations']:
        # Group by job and array task
        job_tasks = {}
        hosts = set()
        for sim in traces['simulations']:
            job_id = sim['slurm_job_id']
            task_id = sim['slurm_array_task_id']
            host = sim['hostname']
            hosts.add(host)

            key = f"{job_id}_{task_id}"
            if key not in job_tasks:
                job_tasks[key] = []
            job_tasks[key].append(sim)

        print(f"Unique SLURM jobs: {len(set(sim['slurm_job_id'] for sim in traces['simulations']))}")
        print(f"Unique hosts: {len(hosts)} - {hosts}")
        print(f"Job/Task combinations: {len(job_tasks)}")

        for key, sims in sorted(job_tasks.items()):
            print(f"  {key}: {len(sims)} simulations")

    print(f"\nAnalyses: {len(traces['analyses'])}")
    print(f"Plots: {len(traces['plots'])}")
    print(f"Ensemble analyses: {len(traces['ensembles'])}")
    print(f"Scan analyses: {len(traces['scans'])}")

    # Check timing
    if traces['simulations']:
        compute_times = [s.get('actual_compute_time', 0) for s in traces['simulations']]
        if compute_times:
            print(f"\nCompute times:")
            print(f"  Mean: {sum(compute_times)/len(compute_times):.2f}s")
            print(f"  Min: {min(compute_times):.2f}s")
            print(f"  Max: {max(compute_times):.2f}s")

    print(f"{'='*60}\n")


def test_single_simulation(work_time=5.0):
    """Test submitting a single simulation."""
    print("\n" + "="*60)
    print("TEST 1: Single Simulation Pipeline")
    print("="*60)

    # Create pipeline
    pipeline = SimulationPipeline(
        TestModel,
        params={'work_time': work_time, 'test_param': 1.5, 'write_trace': True},
        integrator_params={'dt': 0.5},
        analysis_functions=test_analysis_functions,
        plot_functions=test_plot_functions
    )

    # Create SLURM config
    config = SlurmConfig(
        job_name="test_single_sim",
        time="00:10:00"
    )

    # Submit
    slurm = SlurmPipeline()
    result = slurm.submit_pipeline(
        pipeline,
        T=10.0,
        slurm_config=config,
        output_dir="test_results/single_sim"
    )

    print(f"Submitted job: {result['job_id']}")

    # Monitor
    job_info = slurm.monitor.monitor_job(
        result['job_id'],
        result['job_name'],
        show_progress=True
    )

    slurm.monitor.print_job_summary(job_info)

    return result


def test_ensemble_array_jobs(n_sims=20, sims_per_job=5, work_time=10.0):
    """Test ensemble with array jobs."""
    print("\n" + "="*60)
    print("TEST 2: Ensemble with Array Jobs")
    print("="*60)
    print(f"Total simulations: {n_sims}")
    print(f"Simulations per job: {sims_per_job}")
    print(f"Expected array jobs: {(n_sims + sims_per_job - 1) // sims_per_job}")

    # Create ensemble
    ensemble = Ensemble(
        TestModel,
        params={'work_time': work_time, 'test_param': 2.0, 'write_trace': True},
        integrator_params={'dt': 0.5},
        analysis_functions=test_analysis_functions,
        plot_functions=test_plot_functions,
        ensemble_analysis_functions=test_ensemble_analysis
    )

    # Configure SLURM
    config = SlurmConfig(
        job_name="test_ensemble_array",
        time="00:15:00",
        cpus_per_task=1
    )

    # Submit with monitoring
    slurm = SlurmPipeline()
    result = slurm.submit_ensemble(
        ensemble,
        n_simulations=n_sims,
        T=10.0,
        slurm_config=config,
        array_jobs=True,
        sims_per_job=sims_per_job,
        output_dir="test_results/ensemble_array",
        create_individual_plots=True,
        create_individual_animations=False,
        wait_for_completion=True
    )

    print(f"\nJob completed with status: {result['status']}")

    # Show where output was written
    print(f"Output directory: {result['output_dir']}")

    # Quick check of results
    quick_check(result['job_id'], result['output_dir'])

    if 'analysis' in result:
        print("\nEnsemble analysis results:")
        analysis = result['analysis']
        if 'ensemble_execution' in analysis:
            exec_info = analysis['ensemble_execution']
            print(f"  Total simulations: {exec_info['n_simulations']}")
            print(f"  Unique hosts: {exec_info['unique_hosts']}")
            print(f"  Unique jobs: {exec_info['unique_jobs']}")
            print(f"  Ensemble mean: {exec_info.get('ensemble_mean', 'N/A')}")
    else:
        # If no analysis, diagnose why
        print("\nNo analysis found - diagnosing...")
        diagnose_missing_results(result)

    return result


def test_parameter_scan(work_time=5.0):
    """Test parameter scan with array jobs."""
    print("\n" + "="*60)
    print("TEST 3: Parameter Scan with Array Jobs")
    print("="*60)

    # Define scan
    scan_params = {
        'test_param': [0.5, 1.0, 1.5],
        'complexity': [0.5, 1.0]
    }
    n_points = len(scan_params['test_param']) * len(scan_params['complexity'])
    print(f"Parameter combinations: {n_points}")

    # Create scan
    scan = ParameterScan(
        TestModel,
        base_params={'work_time': work_time, 'write_trace': True},
        integrator_params={'dt': 0.5},
        scan_params=scan_params,
        analysis_functions=test_analysis_functions,
        plot_functions=test_plot_functions,
        ensemble_analysis_functions=test_ensemble_analysis,
        scan_analysis_function=test_scan_analysis
    )

    # Configure SLURM
    config = SlurmConfig(
        job_name="test_param_scan",
        time="00:20:00",
        cpus_per_task=2
    )

    # Submit with monitoring
    slurm = SlurmPipeline()
    result = slurm.submit_parameter_scan(
        scan,
        n_simulations_per_point=10,
        T=10.0,
        slurm_config=config,
        array_jobs=True,
        output_dir="test_results/param_scan",
        create_individual_plots=True,
        create_ensemble_plots=True,
        wait_for_completion=True
    )

    print(f"\nJob completed with status: {result['status']}")

    # Show where output was written
    print(f"Output directory: {result['output_dir']}")

    # Quick check of results
    quick_check(result['job_id'], result['output_dir'])

    if 'analysis' in result:
        print("\nScan analysis results:")
        analysis = result['analysis']
        print(f"  Parameter points analyzed: {analysis.get('n_parameter_points', 'N/A')}")
        if 'ensemble_means' in analysis:
            print("  Ensemble means by parameter:")
            for param_str, mean_val in analysis['ensemble_means'].items():
                print(f"    {param_str}: {mean_val:.4f}")
    else:
        # If no analysis, diagnose why
        print("\nNo analysis found - diagnosing...")
        diagnose_missing_results(result)

    return result


def test_batch_submission(work_time=5.0):
    """Test submitting multiple jobs simultaneously."""
    print("\n" + "="*60)
    print("TEST 4: Batch Job Submission")
    print("="*60)

    # Create multiple ensembles with different parameters
    ensembles = []
    for temp in [0.5, 1.0, 2.0]:
        ens = Ensemble(
            TestModel,
            params={'work_time': work_time, 'test_param': temp, 'write_trace': True},
            integrator_params={'dt': 0.5},
            analysis_functions=test_analysis_functions,
            ensemble_analysis_functions=test_ensemble_analysis
        )
        ensembles.append(ens)

    # Submit individually but monitor together
    slurm = SlurmPipeline()
    submissions = []

    for ens in ensembles:
        submission = slurm.submit_ensemble(
            ens,
            n_simulations=15,
            T=10.0,
            array_jobs=True,
            sims_per_job=5,
            output_dir=f"test_results/batch_temp_{ens.params['test_param']}",
            slurm_config=SlurmConfig(
                job_name=f"batch_temp_{ens.params['test_param']}",
                time="00:15:00"
            ),
            wait_for_completion=False  # Don't wait yet
        )
        submissions.append(submission)

    print(f"Submitted {len(submissions)} ensemble jobs")

    # Monitor all jobs together
    job_infos = slurm.monitor.monitor_multiple(submissions, show_progress=True)

    # Collect results
    results = []
    for i, (submission, job_info) in enumerate(zip(submissions, job_infos)):
        if job_info.state.value == "COMPLETED":
            result = slurm.monitor_and_collect(submission, show_progress=False)
            results.append(result)
        else:
            results.append({'status': job_info.state.value})

    print("\nBatch results:")
    for i, result in enumerate(results):
        temp = ensembles[i].params['test_param']
        print(f"  Temperature {temp}: {result['status']}")
        if 'analysis' in result and 'ensemble_execution' in result['analysis']:
            exec_info = result['analysis']['ensemble_execution']
            print(f"    Simulations: {exec_info['n_simulations']}")
            print(f"    Mean value: {exec_info.get('ensemble_mean', 'N/A')}")

    return results


def test_restart_monitoring():
    """Test monitoring a job from a previous submission."""
    print("\n" + "="*60)
    print("TEST 5: Restart Monitoring")
    print("="*60)

    # First, submit a job without waiting
    ensemble = Ensemble(
        TestModel,
        params={'work_time': 30.0, 'test_param': 3.0, 'write_trace': True},
        integrator_params={'dt': 0.5},
        analysis_functions=test_analysis_functions
    )

    slurm = SlurmPipeline()
    submission = slurm.submit_ensemble(
        ensemble,
        n_simulations=20,
        T=10.0,
        array_jobs=True,
        sims_per_job=5,
        output_dir="test_results/restart_test",
        wait_for_completion=False  # Don't wait
    )

    print(f"Submitted job {submission['job_id']} without waiting")

    # Wait a bit
    print("Waiting 5 seconds before checking status...")
    time.sleep(5)

    # Check status
    job_info = slurm.monitor.get_job_info(submission['job_id'])
    print(f"Current status: {job_info.state.value}")

    # Now monitor from where we left off
    print("\nResuming monitoring...")
    result = slurm.monitor_and_collect(submission, show_progress=True)

    print(f"Final status: {result['status']}")

    return result


def run_all_tests(quick=False):
    """Run all tests in sequence."""
    print("\n" + "="*60)
    print("SLURM PIPELINE TEST SUITE")
    print("="*60)
    print(f"Start time: {datetime.now()}")

    # Clean up from previous runs
    clean_test_directories()

    # Create SlurmPipeline to get NFS directory
    slurm = SlurmPipeline()
    nfs_outputs = str(slurm.nfs_work_dir / "outputs")
    print(f"NFS outputs directory: {nfs_outputs}")

    # Adjust parameters for quick testing
    if quick:
        work_time = 2.0
        n_sims = 6
        sims_per_job = 2
    else:
        work_time = 10.0
        n_sims = 20
        sims_per_job = 5

    results = {}

    # Run tests
    try:
        results['single'] = test_single_simulation(work_time=work_time)
        results['ensemble'] = test_ensemble_array_jobs(
            n_sims=n_sims,
            sims_per_job=sims_per_job,
            work_time=work_time
        )
        results['scan'] = test_parameter_scan(work_time=work_time)
        results['batch'] = test_batch_submission(work_time=work_time)

        if not quick:
            results['restart'] = test_restart_monitoring()

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

    # Analyze execution traces
    print("\n" + "="*60)
    print("ANALYZING EXECUTION TRACES")
    print("="*60)
    traces_found = analyze_traces(nfs_outputs=nfs_outputs)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, result in results.items():
        if isinstance(result, dict) and 'status' in result:
            print(f"{test_name}: {result['status']}")
            if 'output_dir' in result:
                print(f"  Output: {result['output_dir']}")
        elif isinstance(result, list):
            # Batch results
            statuses = [r.get('status', 'unknown') for r in result]
            print(f"{test_name}: {statuses}")
        else:
            print(f"{test_name}: completed")

    print(f"\nEnd time: {datetime.now()}")

    # List all test output directories
    print(f"\n" + "="*60)
    print("TEST OUTPUT LOCATIONS")
    print("="*60)
    test_results_dir = Path(nfs_outputs) / "test_results"
    if test_results_dir.exists():
        print(f"Test results in: {test_results_dir}")
        for item in sorted(test_results_dir.iterdir()):
            if item.is_dir():
                files = list(item.rglob("*"))
                markers = list(item.glob("**/*.marker"))
                pkls = list(item.glob("**/*.pkl"))
                print(f"  {item.name}/")
                print(f"    Total files: {len(files)}, Markers: {len(markers)}, PKLs: {len(pkls)}")
    else:
        print("No test results directory found!")

    return results, traces_found


def main():
    """Main entry point for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Test SLURM pipeline")
    parser.add_argument('--quick', action='store_true',
                        help='Run quick tests with reduced work time')
    parser.add_argument('--test', choices=['single', 'ensemble', 'scan', 'batch', 'restart', 'all'],
                        default='all', help='Which test to run')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze existing traces')

    args = parser.parse_args()

    if args.analyze_only:
        slurm = SlurmPipeline()
        nfs_outputs = str(slurm.nfs_work_dir / "outputs")
        analyze_traces(nfs_outputs=nfs_outputs)
    elif args.test == 'all':
        run_all_tests(quick=args.quick)
    else:
        # Run individual test
        if args.test == 'single':
            test_single_simulation()
        elif args.test == 'ensemble':
            test_ensemble_array_jobs()
        elif args.test == 'scan':
            test_parameter_scan()
        elif args.test == 'batch':
            test_batch_submission()
        elif args.test == 'restart':
            test_restart_monitoring()

        # Always analyze traces after individual test
        slurm = SlurmPipeline()
        nfs_outputs = str(slurm.nfs_work_dir / "outputs")
        analyze_traces(nfs_outputs=nfs_outputs)


if __name__ == "__main__":
    main()
