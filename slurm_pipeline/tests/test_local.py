"""
Local test runner to verify the test models work without SLURM.

This helps debug issues before submitting to the cluster.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slurm_pipeline.core import Ensemble
from slurm_pipeline.core import ParameterScan
from slurm_pipeline.core import SimulationPipeline

from .test_models import (
    TestModel,
    test_analysis_functions,
    test_plot_functions,
    test_ensemble_analysis,
    test_scan_analysis
)


def test_local_simulation():
    """Test a single simulation locally."""
    print("Testing single simulation locally...")

    # Create pipeline
    pipeline = SimulationPipeline(
        TestModel,
        params={'work_time': 2.0, 'test_param': 1.5, 'write_trace': True},
        integrator_params={'dt': 0.5},
        analysis_functions=test_analysis_functions,
        plot_functions=test_plot_functions
    )

    # Run
    result = pipeline.run_complete(
        T=5.0,
        output_dir="test_results/local_single"
    )

    print(f"Simulation completed")
    print(f"Analysis: {result.analysis}")

    return result


def test_local_ensemble():
    """Test ensemble locally."""
    print("\nTesting ensemble locally...")

    # Create ensemble
    ensemble = Ensemble(
        TestModel,
        params={'work_time': 1.0, 'test_param': 2.0, 'write_trace': True},
        integrator_params={'dt': 0.5},
        analysis_functions=test_analysis_functions,
        plot_functions=test_plot_functions,
        ensemble_analysis_functions=test_ensemble_analysis
    )

    # Run small ensemble
    analysis = ensemble.run_complete(
        n_simulations=5,
        T=5.0,
        output_dir="test_results/local_ensemble",
        parallel=True,
        max_workers=2,
        create_individual_plots=True,
        create_individual_animations=False
    )

    print(f"Ensemble completed")
    print(f"Ensemble analysis: {analysis}")

    return analysis


def test_local_parameter_scan():
    """Test parameter scan locally."""
    print("\nTesting parameter scan locally...")

    # Create scan
    scan = ParameterScan(
        TestModel,
        base_params={'work_time': 0.5, 'write_trace': True},
        integrator_params={'dt': 0.5},
        scan_params={
            'test_param': [0.5, 1.0],
            'complexity': [0.5, 1.0]
        },
        analysis_functions=test_analysis_functions,
        plot_functions=test_plot_functions,
        ensemble_analysis_functions=test_ensemble_analysis,
        scan_analysis_function=test_scan_analysis
    )

    # Run scan
    results = scan.run_complete(
        n_simulations_per_point=3,
        T=5.0,
        output_dir="test_results/local_scan",
        parallel=True,
        parallel_mode='scan',
        max_workers=2,
        create_individual_plots=True,
        create_ensemble_plots=True
    )

    print(f"Parameter scan completed")
    print(f"Scanned {len(results)} parameter points")

    # Check traces
    trace_dir = Path("slurm_traces")
    if trace_dir.exists():
        traces = list(trace_dir.glob("*.json"))
        print(f"\nGenerated {len(traces)} trace files")

    return results


def verify_models():
    """Verify the test models work correctly."""
    print("="*60)
    print("VERIFYING TEST MODELS LOCALLY")
    print("="*60)

    # Clean up
    import shutil
    for dir_name in ['test_results', 'slurm_traces']:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)

    # Run tests
    try:
        test_local_simulation()
        test_local_ensemble()
        test_local_parameter_scan()

        print("\n" + "="*60)
        print("ALL LOCAL TESTS PASSED")
        print("Models are ready for SLURM testing")
        print("="*60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nFix these errors before running SLURM tests")


if __name__ == "__main__":
    verify_models()
