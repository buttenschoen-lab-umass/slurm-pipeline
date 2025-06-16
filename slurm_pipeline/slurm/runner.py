#!/usr/bin/env python
"""
SLURM runner script for simulation pipeline objects.
Simplified version: Each SLURM job runs one complete ensemble.
Now with tmpfs support for improved I/O performance.
"""

import sys
import copy
import pickle
import itertools
import json
from pathlib import Path
import traceback
import os

# Import will be updated after we add tmpfs_manager to the package
from slurm_pipeline import Ensemble
from slurm_pipeline.slurm.json_encoder import reconstruct_from_dict
from slurm_pipeline.slurm.tmpfs_manager import TmpfsManager, get_tmpfs_info


def reconstruct_config_objects(params):
    """
    Reconstruct config objects from serialized dictionaries.

    Uses the json_encoder's reconstruct_from_dict function to handle
    objects with __class_module__ and __class_name__ metadata.
    """
    if 'config' in params and isinstance(params['config'], dict):
        reconstructed = reconstruct_from_dict(params['config'])
        if reconstructed is not params['config']:
            # Reconstruction succeeded
            params['config'] = reconstructed
            print(f"Reconstructed config object from dictionary")
        else:
            print("No class metadata found or reconstruction failed, keeping as dict")

    return params


def run_ensemble_for_parameter_point(obj, config, array_index=None, use_tmpfs=True):
    """
    Run a complete ensemble for a single parameter point.
    Used for both standalone ensembles and parameter scans.

    Args:
        use_tmpfs: Whether to use local tmpfs for execution
    """

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

        # Handle special case where we have a config object
        if 'config' in params:
            # If it's a dict (from JSON), reconstruct it first
            params = reconstruct_config_objects(params)

            # Now update the config object with scan parameters
            if hasattr(params['config'], '__dict__'):
                # It's a config object, update its attributes
                params['config'] = copy.deepcopy(params['config'])
                for name, value in zip(param_names, param_combination):
                    setattr(params['config'], name, value)

                # Reinitialize interactions if the config has this method
                if hasattr(params['config'], 'init'):
                    params['config'].init()
            else:
                # Fallback: treat as dict
                for name, value in zip(param_names, param_combination):
                    params[name] = value
        else:
            # Regular parameter update
            for name, value in zip(param_names, param_combination):
                params[name] = value

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
        params = reconstruct_config_objects(params)
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

    # Determine if we should use tmpfs
    use_tmpfs = use_tmpfs and config.get('use_tmpfs', True)

    if use_tmpfs:
        print("\nUsing tmpfs for improved I/O performance")

        # Create tmpfs manager
        job_name = config.get('job_name', 'simulation')
        if array_index is not None:
            job_name = f"{job_name}_array_{array_index}"

        tmpfs = TmpfsManager(
            nfs_base_dir=str(output_dir),
            job_name=job_name,
            use_dev_shm=config.get('tmpfs_use_dev_shm', True),
            use_tmp=config.get('tmpfs_use_tmp', True),
            custom_tmpfs=config.get('tmpfs_custom_path', None)
        )

        # Set up tmpfs workspace
        local_work_dir = tmpfs.setup()

        # Map output directory to local tmpfs
        local_output_dir = tmpfs.map_path(str(output_dir))
        print(f"Local work directory: {local_output_dir}")

        # No input files to copy for this use case
        # (ensemble creates everything from scratch)

    else:
        print("\nRunning directly on NFS")
        local_output_dir = str(output_dir)
        tmpfs = None

    try:
        # Run the complete ensemble pipeline - this handles everything!
        ensemble_analysis = ensemble.run_complete(
            n_simulations=n_sims,
            T=config['T'],
            output_dir=local_output_dir,  # Use local directory
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
        results_file = Path(local_output_dir) / "ensemble_results.pkl"
        results_data = {
            # 'results': ensemble.results, Each pipeline saves its sim data.
            'analysis': ensemble_analysis,
            'params': params,
        }
        with open(results_file, 'wb') as f:
            pickle.dump(results_data, f)

        # Save completion marker
        marker_file = Path(local_output_dir) / "completed.marker"
        marker_file.touch()

        print(f"Ensemble completed successfully")

    finally:
        # Sync back to NFS if using tmpfs
        if tmpfs and tmpfs.is_active:
            print("\nSyncing results back to NFS...")
            try:
                # Exclude temporary files
                exclude_patterns = config.get('tmpfs_exclude_patterns', ['*.tmp', '*.swp'])
                tmpfs.sync_back(exclude_patterns=exclude_patterns)
                print("Sync completed successfully")
            except Exception as e:
                print(f"ERROR during sync: {e}")
                # Try emergency sync without exclusions
                try:
                    tmpfs.sync_back(exclude_patterns=None)
                except:
                    pass
                raise


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

    # Check tmpfs availability
    try:
        tmpfs_info = get_tmpfs_info()
        print("\nTmpfs locations:")
        for loc, info in tmpfs_info['locations'].items():
            if info.get('exists') and info.get('writable'):
                print(f"  {loc}: {info.get('available_gb', 0):.1f} GB available ({info.get('mount_type', 'unknown')})")
        if tmpfs_info['recommended']:
            print(f"Recommended: {tmpfs_info['recommended']}")
    except Exception as e:
        print(f"Could not check tmpfs: {e}")

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
        from slurm_pipeline.slurm.json_encoder import load
        with open(config_file, 'r') as f:
            config = load(f)  # This will auto-reconstruct objects with metadata
        print(f"Configuration type: {config['type']}")

        # Reconstruct config objects if needed
        if 'base_params' in config:
            config['base_params'] = reconstruct_config_objects(config['base_params'])

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

