#!/usr/bin/env python
"""
Test script to demonstrate SLURM job cancellation functionality.

This script shows various ways to use the job tracking and cancellation features:
1. Automatic cancellation on script termination (Ctrl+C)
2. Context managers for safe job management
3. Manual job cancellation
4. Tracking status inspection
"""

import os
import sys
import time
import signal
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    package_dir = current_dir.parent.parent
    sys.path.insert(0, str(package_dir))

from slurm_pipeline.slurm import SlurmConfig, SlurmPipeline
from slurm_pipeline.slurm.context_managers import (
    auto_cancel_jobs, no_auto_cancel, slurm_job_batch
)
from slurm_pipeline.core import Ensemble
from slurm_pipeline.tests.test_models import (
    TestModel, test_analysis_functions, test_plot_functions
)


def test_auto_cancellation():
    """Test automatic job cancellation on Ctrl+C."""
    print("\n" + "="*60)
    print("TEST: Automatic Cancellation on Script Exit")
    print("="*60)
    print("This test will submit jobs and wait.")
    print("Press Ctrl+C to test automatic cancellation.")
    print("="*60)

    # Create pipeline with tracking enabled (default)
    slurm = SlurmPipeline()

    # Create ensemble
    ensemble = Ensemble(
        TestModel,
        params={'work_time': 300.0, 'test_param': 1.0, 'write_trace': True},  # 5 min work
        integrator_params={'dt': 0.5},
        analysis_functions=test_analysis_functions
    )

    # Submit job
    config = SlurmConfig(
        job_name="test_auto_cancel",
        time="00:30:00"
    )

    result = slurm.submit_ensemble(
        ensemble,
        n_simulations=20,
        T=10.0,
        slurm_config=config,
        array_jobs=True,
        sims_per_job=5,
        output_dir="test_results/auto_cancel",
        wait_for_completion=False  # Don't wait, let user interrupt
    )

    print(f"\nSubmitted job {result['job_id']}")
    print("Jobs are now running. Press Ctrl+C to test cancellation...")
    print("(Or wait for completion)\n")

    # Check status
    status = slurm.get_tracking_status()
    print(f"Active jobs: {status['stats']['active']}")

    try:
        # Monitor job
        job_info = slurm.monitor.monitor_job(
            result['job_id'],
            result['job_name'],
            show_progress=True
        )
    except KeyboardInterrupt:
        print("\nInterrupted! Jobs should be cancelled automatically...")
        time.sleep(2)  # Give time for cleanup messages


def test_context_manager():
    """Test context manager for automatic cleanup."""
    print("\n" + "="*60)
    print("TEST: Context Manager Auto-Cleanup")
    print("="*60)

    print("Submitting jobs inside context manager...")

    # Use context manager
    with auto_cancel_jobs() as slurm:
        # Create ensemble
        ensemble = Ensemble(
            TestModel,
            params={'work_time': 120.0, 'test_param': 2.0, 'write_trace': True},
            integrator_params={'dt': 0.5}
        )

        # Submit job
        config = SlurmConfig(
            job_name="test_context_mgr",
            time="00:15:00"
        )

        result = slurm.submit_ensemble(
            ensemble,
            n_simulations=10,
            T=10.0,
            slurm_config=config,
            array_jobs=True,
            sims_per_job=5,
            output_dir="test_results/context_mgr"
        )

        print(f"Submitted job {result['job_id']}")

        # Check status
        status = slurm.get_tracking_status()
        print(f"Active jobs: {status['active_jobs']}")

        # Simulate an error or early exit
        print("\nSimulating early exit from context...")
        time.sleep(5)

        # Raise exception to test cleanup
        if input("Raise exception to test cleanup? (y/n): ").lower() == 'y':
            raise RuntimeError("Simulated error!")

    print("\nExited context manager - jobs should be cancelled")


def test_manual_cancellation():
    """Test manual job cancellation."""
    print("\n" + "="*60)
    print("TEST: Manual Job Cancellation")
    print("="*60)

    slurm = SlurmPipeline()

    # Submit multiple jobs
    jobs = []
    for i in range(3):
        ensemble = Ensemble(
            TestModel,
            params={'work_time': 60.0, 'test_param': i+1, 'write_trace': True},
            integrator_params={'dt': 0.5}
        )

        config = SlurmConfig(
            job_name=f"test_manual_{i}",
            time="00:10:00"
        )

        result = slurm.submit_ensemble(
            ensemble,
            n_simulations=5,
            T=10.0,
            slurm_config=config,
            output_dir=f"test_results/manual_{i}"
        )

        jobs.append(result)
        print(f"Submitted job {i+1}: {result['job_id']}")

    # Check status
    print("\nCurrent status:")
    status = slurm.get_tracking_status()
    print(f"Active jobs: {status['active_jobs']}")

    # Wait a bit
    print("\nWaiting 10 seconds...")
    time.sleep(10)

    # Cancel specific job
    if len(jobs) > 0:
        job_to_cancel = jobs[0]['job_id']
        print(f"\nCancelling job {job_to_cancel}...")
        success = slurm.cancel_job(job_to_cancel, "Manual test cancellation")
        print(f"Cancellation {'successful' if success else 'failed'}")

    # Cancel all remaining
    print("\nCancelling all remaining jobs...")
    stats = slurm.cancel_all_active_jobs()
    print(f"Cancelled: {stats['cancelled']}, Failed: {stats['failed']}")

    # Final status
    print("\nFinal status:")
    status = slurm.get_tracking_status()
    print(f"Active: {status['stats']['active']}")
    print(f"Completed: {status['stats']['completed']}")
    print(f"Cancelled: {status['stats']['cancelled']}")


def test_no_auto_cancel():
    """Test disabling auto-cancellation."""
    print("\n" + "="*60)
    print("TEST: Disabled Auto-Cancellation")
    print("="*60)

    slurm = SlurmPipeline()

    # Disable auto-cancellation
    with no_auto_cancel(slurm):
        ensemble = Ensemble(
            TestModel,
            params={'work_time': 30.0, 'test_param': 5.0, 'write_trace': True},
            integrator_params={'dt': 0.5}
        )

        config = SlurmConfig(
            job_name="test_no_auto_cancel",
            time="00:10:00"
        )

        result = slurm.submit_ensemble(
            ensemble,
            n_simulations=5,
            T=10.0,
            slurm_config=config,
            output_dir="test_results/no_auto_cancel"
        )

        print(f"Submitted job {result['job_id']} with auto-cancel disabled")
        print("This job will continue running after script exits")

    print("\nExited no_auto_cancel context")
    print("Job should still be running...")


def test_batch_submission():
    """Test batch submission with tracking."""
    print("\n" + "="*60)
    print("TEST: Batch Submission with Tracking")
    print("="*60)

    with slurm_job_batch("test_batch", verbose=True) as batch:
        # Submit multiple jobs
        for temp in [0.5, 1.0, 1.5]:
            ensemble = Ensemble(
                TestModel,
                params={'work_time': 20.0, 'test_param': temp, 'write_trace': True},
                integrator_params={'dt': 0.5}
            )

            config = SlurmConfig(
                job_name=batch.get_next_job_name(f"temp_{temp}"),
                time="00:10:00"
            )

            result = batch.pipeline.submit_ensemble(
                ensemble,
                n_simulations=5,
                T=10.0,
                slurm_config=config,
                output_dir=f"test_results/batch_temp_{temp}"
            )

            batch.track_submission(result)

        print(f"\nSubmitted {len(batch.submissions)} jobs in batch")

        # Simulate error if requested
        if input("Simulate error to test batch cancellation? (y/n): ").lower() == 'y':
            raise RuntimeError("Simulated batch error!")

    print("\nBatch submission complete")


def run_interactive_tests():
    """Run interactive tests for job cancellation."""
    print("\n" + "="*70)
    print("SLURM JOB CANCELLATION TEST SUITE")
    print("="*70)
    print("This suite tests the job tracking and cancellation functionality")
    print("="*70)

    tests = [
        ("Auto-cancellation on Ctrl+C", test_auto_cancellation),
        ("Context manager cleanup", test_context_manager),
        ("Manual cancellation", test_manual_cancellation),
        ("Disabled auto-cancellation", test_no_auto_cancel),
        ("Batch submission", test_batch_submission)
    ]

    while True:
        print("\nAvailable tests:")
        for i, (name, _) in enumerate(tests, 1):
            print(f"{i}. {name}")
        print("0. Exit")

        try:
            choice = input("\nSelect test (0-5): ").strip()
            if choice == '0':
                break

            idx = int(choice) - 1
            if 0 <= idx < len(tests):
                tests[idx][1]()
            else:
                print("Invalid choice")

        except KeyboardInterrupt:
            print("\n\nInterrupted! Cleaning up...")
            break
        except Exception as e:
            print(f"\nTest error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("Testing complete")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test SLURM job cancellation")
    parser.add_argument('--test', choices=['auto', 'context', 'manual', 'no-cancel', 'batch'],
                        help='Run specific test')
    parser.add_argument('--all', action='store_true', help='Run all tests')

    args = parser.parse_args()

    if args.all:
        # Run all tests non-interactively
        test_manual_cancellation()
        test_no_auto_cancel()
        print("\nAll non-interactive tests complete")
    elif args.test:
        # Run specific test
        test_map = {
            'auto': test_auto_cancellation,
            'context': test_context_manager,
            'manual': test_manual_cancellation,
            'no-cancel': test_no_auto_cancel,
            'batch': test_batch_submission
        }
        test_map[args.test]()
    else:
        # Interactive mode
        run_interactive_tests()
