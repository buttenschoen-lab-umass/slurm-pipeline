#!/usr/bin/env python
"""
SLURM runner script for simulation pipeline objects.
Simplified version: Each SLURM job runs one complete ensemble.
"""

import sys
import pickle
import json
from pathlib import Path
import traceback
import os

from slurm_pipeline import Ensemble


def run_ensemble_for_parameter_point(obj, config, array_index=None):
    """
    Run a complete ensemble for a single parameter point.
    Used for both standalone ensembles and parameter scans.
    """
    import itertools

    # Determine parameters for this run
    if config['type'] == 'parameter_scan' and array_index is not None:
        # This is a parameter scan - get the specific parameter combination
        param_names = list(config['scan_params'].keys())
        param_values = [config['scan_params'][k] for k in param_names]
        param_combinations = list(itertools.product(*param_values))

        if array_index >= len(param_combinations):
            raise ValueError(f"Array index {array_index} exceeds number of parameter combinations")

        # Get parameter combination for this job
        param_combination = param_combinations[array_index]

        # Update base parameters with scan values
        params = config['base_params'].copy()
        for name, value in zip(param_names, param_combination):
            params[name] = value

        # Create output directory for this parameter point
        param_str = "_".join([f"{k}_{v:.3g}" if isinstance(v, (int, float)) else f"{k}_{v}"
                            for k, v in zip(param_names, param_combination)])
        output_dir = Path(config['output_dir']) / param_str

        print(f"Array task {array_index}: Running parameter point {dict(zip(param_names, param_combination))}")

    else:
        # Standalone ensemble - use base parameters
        params = obj.params
        output_dir = Path(config['output_dir'])
        print("Running standalone ensemble")

    # Create ensemble
    if config['type'] == 'parameter_scan':
        # For parameter scans, create new ensemble with updated parameters
        ensemble = Ensemble(
            obj.model_type, params, obj.integrator_params,
            obj.analysis_functions, obj.plot_functions, obj.animation_function,
            obj.ensemble_analysis_functions, obj.ensemble_plot_functions
        )
    else:
        # For standalone ensembles, use the provided object
        ensemble = obj

    # Get configuration values
    n_sims = config.get('n_simulations_per_point', config.get('n_simulations', 20))
    max_workers = config.get('max_workers', config.get('cpus_per_task', 4))
    always_plot_all = config.get('always_plot_all', False)
    always_animate_all = config.get('always_animate_all', False)
    max_plots = n_sims if always_plot_all else config.get('max_individual_plots', 5)
    max_animations = n_sims if always_animate_all else config.get('max_individual_animations', 5)

    print(f"Running {n_sims} simulations using {max_workers} workers")
    print(f"Will create plots for {'all' if always_plot_all else f'up to {max_plots}'} simulations")

    # Run the complete ensemble pipeline - this handles everything!
    ensemble_analysis = ensemble.run_complete(
        n_simulations=n_sims,
        T=config['T'],
        output_dir=str(output_dir),
        parallel=True,  # Always use parallel execution
        max_workers=max_workers,  # Use allocated CPUs
        progress=True,  # Show progress
        create_individual_plots=config.get('create_individual_plots', True),
        create_individual_animations=config.get('create_individual_animations', False),
        max_individual_plots=max_plots,
        max_individual_animations=max_animations,
        ensemble_plots=config.get('create_ensemble_plots', True)
    )

    # Save ensemble results for later collection
    results_file = output_dir / "ensemble_results.pkl"
    results_data = {
        # 'results': ensemble.results, Each pipeline saves its sim data.
        'analysis': ensemble_analysis,
        'params': params,
    }
    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)

    # Save completion marker
    marker_file = output_dir / "completed.marker"
    marker_file.touch()

    print(f"Ensemble completed successfully")


def main():
    """Main entry point for runner script."""
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: runner.py <pickle_file> <config_file> [array_index] [env_file]")
        sys.exit(1)

    pickle_file = Path(sys.argv[1])
    config_file = Path(sys.argv[2])
    array_index = int(sys.argv[3]) if len(sys.argv) > 3 else None
    env_file = Path(sys.argv[4]) if len(sys.argv) > 4 else None

    # Log execution info
    print(f"{'='*60}")
    print(f"SLURM Runner Started")
    print(f"{'='*60}")
    print(f"Host: {os.uname().nodename}")
    print(f"PID: {os.getpid()}")
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}")
    print(f"SLURM_ARRAY_TASK_ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}")
    print(f"Pickle file: {pickle_file}")
    print(f"Config file: {config_file}")
    print(f"Array index: {array_index}")
    print(f"Environment file: {env_file if env_file else 'None'}")
    print(f"{'='*60}")

    try:
        # Load environment if provided
        if env_file and env_file.exists():
            print("\nLoading environment...")

            # The runner_env.py file is created dynamically by the pipeline
            # It should be in the same directory as this runner script
            try:
                from runner_env import _env_loader, update_object_functions

                _env_loader.load_from_file(env_file)

                # Make recreated functions available in __main__ namespace
                import __main__
                for name, func in _env_loader.recreated_functions.items():
                    setattr(__main__, name, func)

            except ImportError:
                print("Note: runner_env module not found - environment capture may not be enabled")
                # Define no-op function
                def update_object_functions(obj):
                    return 0
        else:
            # No environment file, define no-op function
            def update_object_functions(obj):
                return 0

        # Load configuration
        print("\nLoading configuration...")
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Configuration type: {config['type']}")

        # Load pickled object
        print("\nLoading pickled object...")
        with open(pickle_file, 'rb') as f:
            obj = pickle.load(f)
        print("Object loaded successfully")

        # Apply captured environment updates
        update_object_functions(obj)

        # Run ensemble
        print(f"\nExecuting ensemble job...")
        run_ensemble_for_parameter_point(obj, config, array_index)

        print(f"\n{'='*60}")
        print("Job completed successfully")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"JOB FAILED: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
