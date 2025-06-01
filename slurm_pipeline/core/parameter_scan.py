import os
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from dataclasses import dataclass
from tqdm.auto import tqdm
import concurrent.futures
from functools import partial

from .ensemble import Ensemble


def _run_parameter_point(model_type: type,
                        base_params: Dict[str, Any],
                        integrator_params: Dict[str, Any],
                        analysis_functions: Optional[Dict[str, Callable]],
                        plot_functions: Optional[Dict[str, Callable]],
                        animation_function: Optional[Callable],
                        ensemble_analysis_functions: Optional[Dict[str, Callable]],
                        ensemble_plot_functions: Optional[Dict[str, Callable]],
                        param_names: List[str],
                        param_combination: Tuple,
                        n_simulations: int,
                        T: float,
                        output_dir: Optional[str],
                        create_individual_plots: bool,
                        create_individual_animations: bool,
                        max_individual_plots: Optional[int],
                        force_sequential_ensemble: bool = True) -> Tuple[Tuple, Dict[str, Any]]:
    """
    Helper function to run ensemble at a single parameter point.
    This is a module-level function so it can be pickled for multiprocessing.

    Args:
        force_sequential_ensemble: If True, force ensemble to run sequentially
                                  to avoid nested parallelization

    Returns:
        Tuple of (param_key, results_dict)
    """
    # Create parameter dictionary for this point
    params = base_params.copy()
    for name, value in zip(param_names, param_combination):
        params[name] = value

    # Create ensemble directory if output_dir is provided
    ensemble_output_dir = None
    if output_dir:
        param_str = "_".join([f"{k}_{v:.3g}" if isinstance(v, (int, float)) else f"{k}_{v}"
                            for k, v in zip(param_names, param_combination)])
        ensemble_output_dir = os.path.join(output_dir, param_str)

    # Create and run ensemble
    ensemble = Ensemble(
        model_type, params, integrator_params,
        analysis_functions, plot_functions, animation_function,
        ensemble_analysis_functions, ensemble_plot_functions
    )

    # Run ensemble - IMPORTANT: force sequential if we're already in a parallel context
    ensemble.run(
        n_simulations, T,
        parallel=not force_sequential_ensemble,  # Only parallelize if not forced sequential
        progress=False,  # No progress bar for individual ensembles
        output_dir=ensemble_output_dir,
        create_individual_plots=create_individual_plots,
        create_individual_animations=create_individual_animations,
        max_individual_plots=max_individual_plots
    )

    ensemble_analysis = ensemble.analyze()
    ensemble.visualize(ensemble_output_dir, individual_plots=False)

    # Create param key
    param_key = tuple(zip(param_names, param_combination))

    return param_key, {
        'ensemble': ensemble,
        'analysis': ensemble_analysis
    }


class ParameterScan:
    """
    Scan over parameter space, running ensembles at each point.
    """

    def __init__(self,
                 model_type: type,
                 base_params: Dict[str, Any],
                 integrator_params: Dict[str, Any],
                 scan_params: Dict[str, List[Any]],
                 analysis_functions: Optional[Dict[str, Callable]] = None,
                 plot_functions: Optional[Dict[str, Callable]] = None,
                 animation_function: Optional[Callable] = None,
                 ensemble_analysis_functions: Optional[Dict[str, Callable]] = None,
                 ensemble_plot_functions: Optional[Dict[str, Callable]] = None,
                 scan_analysis_function: Optional[Callable] = None,
                 scan_plot_function: Optional[Callable] = None):
        """
        Initialize parameter scan.

        Args:
            model_type, base_params, integrator_params: Base configuration
            scan_params: Dict of parameter_name -> list of values to scan
            analysis_functions, plot_functions: For individual simulations
            animation_function: Animation function for individual simulations
            ensemble_analysis_functions, ensemble_plot_functions: For ensembles
            scan_analysis_function: Function to analyze the whole scan
            scan_plot_function: Function to plot scan results
        """
        self.model_type = model_type
        self.base_params = base_params
        self.integrator_params = integrator_params
        self.scan_params = scan_params

        # Functions for different levels
        self.analysis_functions = analysis_functions
        self.plot_functions = plot_functions
        self.animation_function = animation_function
        self.ensemble_analysis_functions = ensemble_analysis_functions
        self.ensemble_plot_functions = ensemble_plot_functions
        self.scan_analysis_function = scan_analysis_function
        self.scan_plot_function = scan_plot_function

        # Results storage
        self.scan_results: Dict[Tuple, Dict[str, Any]] = {}

    def run(self,
            n_simulations_per_point: int,
            T: float = 50.0,
            parallel: bool = True,
            parallel_mode: str = 'auto',
            max_workers: Optional[int] = None,
            progress: bool = True,
            output_dir: Optional[str] = None,
            create_individual_plots: bool = True,
            create_individual_animations: bool = True,
            max_individual_plots: Optional[int] = None) -> Dict[Tuple, Dict[str, Any]]:
        """
        Run parameter scan.

        Args:
            n_simulations_per_point: Number of simulations per parameter point
            T: Simulation time
            parallel: Use parallel execution
            parallel_mode: 'auto', 'scan', or 'ensemble'
                - 'auto': Parallelize scan if n_simulations_per_point <= n_workers/2
                - 'scan': Always parallelize across parameter points
                - 'ensemble': Always parallelize within ensembles
            max_workers: Maximum number of parallel workers
            progress: Show progress bar
            output_dir: Base output directory for all results
            create_individual_plots: Create plots for individual simulations
            create_individual_animations: Create animations for individual simulations
            max_individual_plots: Maximum number of individual simulations to visualize per ensemble

        Returns:
            Dictionary mapping parameter tuples to ensemble results
        """
        import itertools

        # Generate parameter combinations
        param_names = list(self.scan_params.keys())
        param_values = list(self.scan_params.values())
        param_combinations = list(itertools.product(*param_values))

        print(f"Parameter scan: {len(param_combinations)} points × {n_simulations_per_point} simulations")
        print(f"Total simulations: {len(param_combinations) * n_simulations_per_point}")

        # Determine parallelization strategy
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        effective_workers = max_workers if max_workers is not None else cpu_count

        if parallel and parallel_mode == 'auto':
            # Simple math-based strategy
            # If we can fit at least 2 parameter points per worker with their simulations,
            # then parallelize at scan level
            if n_simulations_per_point <= effective_workers // 2:
                parallelize_scan = True
                strategy_reason = f"{n_simulations_per_point} sims/point <= {effective_workers}//2 workers"
            else:
                parallelize_scan = False
                strategy_reason = f"{n_simulations_per_point} sims/point > {effective_workers}//2 workers"

            print(f"Auto mode: {'scan' if parallelize_scan else 'ensemble'} parallelization ({strategy_reason})")
            print(f"  {len(param_combinations)} parameter points × {n_simulations_per_point} simulations each")
            print(f"  {effective_workers} workers available")

        elif parallel_mode == 'scan':
            parallelize_scan = parallel
        else:  # 'ensemble'
            parallelize_scan = False

        if parallel and parallelize_scan and len(param_combinations) > 1:
            # Determine optimal number of workers
            if max_workers is None:
                # Limit workers based on both CPU count and number of tasks
                optimal_workers = min(cpu_count, len(param_combinations))
                # Also consider memory constraints - maybe limit to 75% of CPUs for large scans
                if len(param_combinations) > cpu_count * 2:
                    optimal_workers = int(cpu_count * 0.75)
            else:
                optimal_workers = min(max_workers, len(param_combinations))

            print(f"Parallelizing across {len(param_combinations)} parameter points")
            print(f"Using {optimal_workers} workers")

            # Create partial function with fixed parameters
            run_func = partial(
                _run_parameter_point,
                self.model_type,
                self.base_params,
                self.integrator_params,
                self.analysis_functions,
                self.plot_functions,
                self.animation_function,
                self.ensemble_analysis_functions,
                self.ensemble_plot_functions,
                param_names
            )

            # Run parameter points in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
                # Submit all jobs
                futures = []
                for combination in param_combinations:
                    future = executor.submit(
                        run_func,
                        combination,
                        n_simulations_per_point,
                        T,
                        output_dir,
                        create_individual_plots,
                        create_individual_animations,
                        max_individual_plots,
                        True  # force_sequential_ensemble=True to avoid nested parallelization
                    )
                    futures.append(future)

                # Collect results with progress bar
                if progress:
                    iterator = tqdm(concurrent.futures.as_completed(futures),
                                  total=len(param_combinations),
                                  desc="Parameter scan")
                else:
                    iterator = concurrent.futures.as_completed(futures)

                for future in iterator:
                    try:
                        param_key, results = future.result()
                        self.scan_results[param_key] = results
                    except Exception as e:
                        print(f"Parameter point failed: {e}")
                        import traceback
                        traceback.print_exc()

        else:
            # Sequential execution or ensemble-level parallelization
            if parallel and not parallelize_scan:
                print(f"Ensembles will parallelize internally across {n_simulations_per_point} simulations")

            # Progress bar for scan points
            if progress:
                iterator = tqdm(param_combinations, desc="Parameter scan")
            else:
                iterator = param_combinations

            for combination in iterator:
                # Create parameter dictionary for this point
                params = self.base_params.copy()
                for name, value in zip(param_names, combination):
                    params[name] = value

                # Create ensemble directory if output_dir is provided
                ensemble_output_dir = None
                if output_dir:
                    param_str = "_".join([f"{k}_{v:.3g}" if isinstance(v, (int, float)) else f"{k}_{v}"
                                        for k, v in zip(param_names, combination)])
                    ensemble_output_dir = os.path.join(output_dir, param_str)

                # Create and run ensemble
                ensemble = Ensemble(
                    self.model_type, params, self.integrator_params,
                    self.analysis_functions, self.plot_functions, self.animation_function,
                    self.ensemble_analysis_functions, self.ensemble_plot_functions
                )

                # Run with ensemble parallelization if not parallelizing scan
                ensemble.run(
                    n_simulations_per_point, T,
                    parallel=parallel and not parallelize_scan,
                    max_workers=max_workers,
                    progress=False,  # Use outer progress bar
                    output_dir=ensemble_output_dir,
                    create_individual_plots=create_individual_plots,
                    create_individual_animations=create_individual_animations,
                    max_individual_plots=max_individual_plots
                )

                ensemble_analysis = ensemble.analyze()

                # Store results
                param_key = tuple(zip(param_names, combination))
                self.scan_results[param_key] = {
                    'ensemble': ensemble,
                    'analysis': ensemble_analysis
                }

        return self.scan_results

    def analyze(self) -> Any:
        """Run scan-level analysis."""
        if self.scan_analysis_function:
            return self.scan_analysis_function(self.scan_results)
        return None

    def visualize(self, output_dir: str, create_ensemble_plots: bool = True) -> None:
        """
        Create scan-level visualizations.

        Args:
            output_dir: Output directory
            create_ensemble_plots: Whether to create ensemble-level plots
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create ensemble plots if requested
        if create_ensemble_plots:
            for param_key, results in self.scan_results.items():
                # Create subdirectory name from parameters
                param_str = "_".join([f"{k}_{v:.3g}" if isinstance(v, (int, float)) else f"{k}_{v}"
                                    for k, v in param_key])
                ensemble_dir = os.path.join(output_dir, param_str)

                ensemble = results['ensemble']
                # Only create ensemble plots, not individual ones (already created during run)
                ensemble.visualize(ensemble_dir, individual_plots=False, ensemble_plots=True)

        # Create scan-level plots
        if self.scan_plot_function:
            self.scan_plot_function(self.scan_results, output_dir)

    def run_complete(self,
                     n_simulations_per_point: int,
                     T: float = 50.0,
                     output_dir: Optional[str] = None,
                     parallel: bool = True,
                     parallel_mode: str = 'auto',
                     max_workers: Optional[int] = None,
                     progress: bool = True,
                     create_individual_plots: bool = True,
                     create_individual_animations: bool = True,
                     max_individual_plots: Optional[int] = None,
                     create_ensemble_plots: bool = True) -> Dict[Tuple, Dict[str, Any]]:
        """
        Run complete parameter scan pipeline.

        Args:
            n_simulations_per_point: Number of simulations per parameter point
            T: Simulation time
            output_dir: Output directory for all results
            parallel: Use parallel execution
            parallel_mode: 'auto', 'scan', or 'ensemble' (see run() for details)
            max_workers: Maximum number of parallel workers
            progress: Show progress bar
            create_individual_plots: Create plots for individual simulations
            create_individual_animations: Create animations for individual simulations
            max_individual_plots: Maximum number of individual simulations to visualize per ensemble
            create_ensemble_plots: Whether to create ensemble-level plots

        Returns:
            Dictionary mapping parameter tuples to ensemble results
        """
        # Run scan with visualization
        self.run(
            n_simulations_per_point, T, parallel, parallel_mode, max_workers, progress,
            output_dir=output_dir,
            create_individual_plots=create_individual_plots,
            create_individual_animations=create_individual_animations,
            max_individual_plots=max_individual_plots
        )

        # Analyze
        scan_analysis = self.analyze()

        # Create scan-level visualizations
        if output_dir:
            self.visualize(output_dir, create_ensemble_plots=create_ensemble_plots)

        return self.scan_results
