#!/usr/bin/env python
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
        # Strategy 1: Direct import
        from slurm_pipeline import Ensemble
    except ImportError:
        try:
            # Strategy 2: Try core submodule
            from slurm_pipeline.core import Ensemble
        except ImportError:
            # Strategy 3: Add parent directory to path and try again
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                from slurm_pipeline import Ensemble
            except ImportError:
                from slurm_pipeline.core import Ensemble

    # Create ensemble for this parameter point
    ensemble = Ensemble(
        obj.model_type, params, obj.integrator_params,
        obj.analysis_functions, obj.plot_functions, obj.animation_function,
        obj.ensemble_analysis_functions, obj.ensemble_plot_functions,
        save_function=obj.save_function,  # Make sure save function is passed
        load_function=obj.load_function
    )

    # Get number of simulations from config
    n_sims = config.get('n_simulations_per_point', 20)
    print(f"Running {n_sims} simulations for this parameter point")

    # Determine plotting behavior
    # If always_plot_all is True, set max_individual_plots to n_sims
    always_plot_all = config.get('always_plot_all', False)
    max_plots = n_sims if always_plot_all else config.get('max_individual_plots', 5)

    # Run ensemble
    ensemble.run(
        n_simulations=n_sims,
        T=config['T'],
        output_dir=str(output_dir),
        parallel=config.get('parallel', False),
        max_workers=config.get('max_workers', None),
        progress=False,  # No progress bar for individual ensembles in array jobs
        create_individual_plots=config.get('create_individual_plots', True),
        create_individual_animations=config.get('create_individual_animations', False),
        max_individual_plots=max_plots  # Use computed value
    )

    # Run ensemble analysis
    ensemble_analysis = ensemble.analyze()

    # IMPORTANT: Save simulation data for ALL simulations
    if config.get('save_simulation_data', True) and obj.save_function:
        print(f"Saving simulation data for {len(ensemble.results)} simulations...")
        for i, result in enumerate(ensemble.results):
            sim_dir = output_dir / f"simulation_{result.sim_id:03d}"
            sim_dir.mkdir(exist_ok=True)

            # Save the .npz file with simulation data
            data_path = sim_dir / "simulation_data.npz"
            obj.save_function(result.time, result.solution, result.params, str(data_path), sim_id=result.sim_id)

        print(f"Saved data for all {len(ensemble.results)} simulations")

    # Create ensemble visualizations if requested
    if config.get('create_ensemble_plots', True):
        ensemble.visualize(str(output_dir), individual_plots=False, ensemble_plots=True)

    # Save results - IMPORTANT: Don't save the ensemble object directly
    # as it contains unpickleable recreated functions
    results_file = output_dir / "ensemble_results.pkl"

    # Clear functions before pickling to avoid issues
    results_data = {
        'results': ensemble.results,  # Just the simulation results
        'analysis': ensemble_analysis,  # Just the analysis data
        'params': params,  # Parameters for this point
    }

    with open(results_file, 'wb') as f:
        pickle.dump(results_data, f)

    # Save completion marker
    marker_file = output_dir / "completed.marker"
    marker_file.touch()

    print(f"Array task {array_index}: Completed parameter point successfully")


def run_complete_ensemble(obj, config):
    """Run complete ensemble with post-processing."""
    print("Running complete ensemble")

    # Determine plotting behavior
    n_sims = config['n_simulations']
    always_plot_all = config.get('always_plot_all', False)
    max_plots = n_sims if always_plot_all else config.get('max_individual_plots', 5)

    obj.run(
        n_simulations=n_sims,
        T=config['T'],
        parallel=config.get('parallel', True),
        output_dir=config.get('output_dir'),
        create_individual_plots=config.get('create_individual_plots', True),
        create_individual_animations=config.get('create_individual_animations', False),
        max_individual_plots=max_plots
    )

    # Run analysis
    ensemble_analysis = obj.analyze()

    # Save simulation data if requested
    if config.get('save_simulation_data', True) and obj.save_function and config.get('output_dir'):
        output_dir = Path(config['output_dir'])
        print(f"Saving simulation data for {len(obj.results)} simulations...")

        for result in obj.results:
            sim_dir = output_dir / f"simulation_{result.sim_id:03d}"
            if sim_dir.exists():  # Only save if the directory was created
                data_path = sim_dir / "simulation_data.npz"
                obj.save_function(result.time, result.solution, result.params, str(data_path), sim_id=result.sim_id)

        print(f"Saved data for all simulations")

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


def run_complete_scan(obj, config):
    """Run complete parameter scan."""
    print("Running complete parameter scan")

    # Determine plotting behavior
    always_plot_all = config.get('always_plot_all', False)
    n_sims = config.get('n_simulations_per_point', 20)
    max_plots = n_sims if always_plot_all else config.get('max_individual_plots', 5)

    obj.run(
        n_simulations_per_point=n_sims,
        T=config['T'],
        parallel=config.get('parallel', True),
        parallel_mode=config.get('parallel_mode', 'auto'),
        output_dir=config.get('output_dir'),
        create_individual_plots=config.get('create_individual_plots', True),
        create_individual_animations=config.get('create_individual_animations', False),
        max_individual_plots=max_plots
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


def run_single_pipeline(obj, config):
    """Run single simulation pipeline."""
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
            animation=config.get('create_animation', False)
        )

        # Save data
        if config.get('save_data', True) and obj.save_function:
            data_path = Path(config['output_dir']) / "simulation_data.npz"
            obj.save(result, str(data_path))


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
        # IMPORTANT: Load environment FIRST, before loading pickle
        # This ensures functions are available when unpickling
        if env_file and env_file.exists():
            print("\nLoading environment before unpickling...")
            from runner_env import _env_loader, RECREATED_FUNCTIONS, update_object_functions

            # Force reload of environment
            _env_loader.load_from_file(env_file)

            # Make recreated functions available in __main__ namespace
            # This is crucial for unpickling to work
            import __main__
            for name, func in _env_loader.recreated_functions.items():
                setattr(__main__, name, func)
                print(f"  Added {name} to __main__")

        # Load configuration
        print("\nLoading configuration...")
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Configuration type: {config['type']}")

        # NOW load pickled object - functions should be available
        print("\nLoading pickled object...")
        with open(pickle_file, 'rb') as f:
            obj = pickle.load(f)
        print("Object loaded successfully")

        # Apply captured environment updates to the object
        # This updates the object's function references to use the recreated versions
        if env_file and env_file.exists():
            update_object_functions(obj)

        # Determine object type and run appropriately
        obj_type = config['type']
        print(f"\nExecuting {obj_type} job...")

        if obj_type == 'ensemble':
            if array_index is not None:
                run_ensemble_array(obj, config, array_index)
            else:
                run_complete_ensemble(obj, config)

        elif obj_type == 'parameter_scan':
            if array_index is not None:
                run_scan_array(obj, config, array_index)
            else:
                run_complete_scan(obj, config)

        elif obj_type == 'pipeline':
            run_single_pipeline(obj, config)

        else:
            raise ValueError(f"Unknown object type: {obj_type}")

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
