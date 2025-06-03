import os
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from dataclasses import dataclass
from tqdm.auto import tqdm
import concurrent.futures
from functools import partial

from .pipeline import SimulationPipeline
from .pipeline import SimulationResult


def _run_single_simulation(model_type: type,
                          params: Dict[str, Any],
                          integrator_params: Dict[str, Any],
                          analysis_functions: Optional[Dict[str, Callable]],
                          plot_functions: Optional[Dict[str, Callable]],
                          animation_function: Optional[Callable],
                          T: float,
                          sim_index: int,
                          output_dir: Optional[str] = None,
                          create_plots: bool = True,
                          create_animation: bool = True) -> SimulationResult:
    """
    Helper function to run a single simulation.
    """
    # Create pipeline with sim_index as sim_id
    pipeline = SimulationPipeline(
        model_type, params, integrator_params,
        analysis_functions, plot_functions, animation_function,
        sim_id=sim_index
    )
    result = pipeline.run(T, progress=False)
    result = pipeline.analyze(result)

    # Create visualizations if requested
    if output_dir and (create_plots or create_animation):
        sim_output_dir = os.path.join(output_dir, f"simulation_{sim_index:03d}")
        result = pipeline.visualize(result, sim_output_dir,
                                  plots=create_plots,
                                  animation=create_animation)

        # Save simulation data using model's save method
        if hasattr(model_type, 'save_trajectory'):
            data_path = os.path.join(sim_output_dir, "simulation_data.npz")
            pipeline.save(result, data_path)

    return result


class Ensemble:
    """
    Collection of simulations with the same parameters but different initial conditions.
    """

    def __init__(self,
                 model_type: type,
                 params: Dict[str, Any],
                 integrator_params: Dict[str, Any],
                 analysis_functions: Optional[Dict[str, Callable]] = None,
                 plot_functions: Optional[Dict[str, Callable]] = None,
                 animation_function: Optional[Callable] = None,
                 ensemble_analysis_functions: Optional[Dict[str, Callable]] = None,
                 ensemble_plot_functions: Optional[Dict[str, Callable]] = None,
                 starting_sim_id: int = 0):
        """
        Initialize ensemble.

        Args:
            model_type: Model class (must have save_trajectory/load_trajectory methods)
            params, integrator_params: Same as SimulationPipeline
            analysis_functions, plot_functions, animation_function: For individual sims
            ensemble_analysis_functions: Functions that analyze the whole ensemble
            ensemble_plot_functions: Functions that plot the whole ensemble
            starting_sim_id: Starting ID for simulations in this ensemble
        """
        self.model_type = model_type
        self.params = params
        self.integrator_params = integrator_params

        # Individual simulation functions
        self.analysis_functions = analysis_functions
        self.plot_functions = plot_functions
        self.animation_function = animation_function

        # Ensemble-level functions
        self.ensemble_analysis_functions = ensemble_analysis_functions or {}
        self.ensemble_plot_functions = ensemble_plot_functions or {}

        # Storage for results
        self.results: List[SimulationResult] = []
        self.ensemble_analysis: Dict[str, Any] = {}

        # Starting ID for simulations
        self.starting_sim_id = starting_sim_id

    def run(self,
            n_simulations: int,
            T: float = 50.0,
            parallel: bool = True,
            max_workers: Optional[int] = None,
            progress: bool = True,
            output_dir: Optional[str] = None,
            create_individual_plots: bool = True,
            create_individual_animations: bool = True,
            max_individual_plots: Optional[int] = None,
            max_individual_animations: Optional[int] = None) -> List[SimulationResult]:
        """
        Run ensemble of simulations.

        Args:
            n_simulations: Number of simulations to run
            T: Simulation time for each
            parallel: Use parallel execution
            max_workers: Max parallel workers (None = CPU count)
            progress: Show progress bar
            output_dir: Directory for individual simulation outputs
            create_individual_plots: Create plots for each simulation during run
            create_individual_animations: Create animations for each simulation during run
            max_individual_plots: Maximum number of individual simulations to visualize
                                (None = visualize all)

        Returns:
            List of SimulationResult objects
        """
        self.results = []

        # Determine which simulations to visualize
        if max_individual_plots is not None:
            visualize_indices = set(range(min(n_simulations, max_individual_plots)))
        else:
            visualize_indices = set(range(n_simulations))

        # Determine which simulations to visualize
        if max_individual_animations is not None:
            animation_indices = set(range(min(n_simulations, max_individual_animations)))
        else:
            animation_indices = set(range(n_simulations))

        if parallel and n_simulations > 1:
            # Create partial function with all the fixed parameters
            run_func = partial(
                _run_single_simulation,
                self.model_type,
                self.params,
                self.integrator_params,
                self.analysis_functions,
                self.plot_functions,
                self.animation_function,
                T
            )

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs with visualization settings
                futures = []
                for i in range(n_simulations):
                    should_visualize = i in visualize_indices
                    should_animate = i in animation_indices
                    # Use starting_sim_id + i as the actual sim_id
                    sim_id = self.starting_sim_id + i
                    future = executor.submit(
                        run_func, sim_id, output_dir,
                        create_individual_plots and should_visualize,
                        create_individual_animations and should_animate,
                    )
                    futures.append(future)

                if progress:
                    iterator = tqdm(concurrent.futures.as_completed(futures),
                                  total=n_simulations, desc="Running ensemble")
                else:
                    iterator = concurrent.futures.as_completed(futures)

                for future in iterator:
                    try:
                        self.results.append(future.result())
                    except Exception as e:
                        print(f"Simulation failed: {e}")
                        import traceback
                        traceback.print_exc()
        else:
            # Sequential execution
            iterator = range(n_simulations)
            if progress:
                iterator = tqdm(iterator, desc="Running ensemble")

            for i in iterator:
                # Use starting_sim_id + i as the actual sim_id
                sim_id = self.starting_sim_id + i
                pipeline = SimulationPipeline(
                    self.model_type, self.params, self.integrator_params,
                    self.analysis_functions, self.plot_functions, self.animation_function,
                    sim_id=sim_id  # Pass the correct sim_id
                )
                result = pipeline.run(T, progress=False)
                result = pipeline.analyze(result)

                # Create visualizations if requested
                should_visualize = i in visualize_indices
                if output_dir and should_visualize and (create_individual_plots or create_individual_animations):
                    sim_output_dir = os.path.join(output_dir, f"simulation_{sim_id:03d}")
                    result = pipeline.visualize(result, sim_output_dir,
                                              plots=create_individual_plots,
                                              animation=create_individual_animations)

                self.results.append(result)

        return self.results

    def analyze(self) -> Dict[str, Any]:
        """
        Run ensemble-level analysis.

        Returns:
            Dictionary of ensemble analysis results
        """
        self.ensemble_analysis = {
            'n_simulations': len(self.results),
            'params': self.params
        }

        # Run ensemble analysis functions
        for name, func in self.ensemble_analysis_functions.items():
            try:
                self.ensemble_analysis[name] = func(self.results)
            except Exception as e:
                print(f"Ensemble analysis '{name}' failed: {e}")
                self.ensemble_analysis[name] = None

        # Aggregate individual analyses
        if self.results and self.results[0].analysis:
            # Example: rotation statistics
            if 'rotation' in self.results[0].analysis:
                rotation_types = [r.analysis['rotation'] for r in self.results
                                if r.analysis and 'rotation' in r.analysis]
                if rotation_types:
                    self.ensemble_analysis['rotation_types'] = rotation_types
                    self.ensemble_analysis['rotation_statistics'] = {
                        'CW': rotation_types.count('CW') / len(rotation_types) * 100,
                        'CCW': rotation_types.count('CCW') / len(rotation_types) * 100,
                        'NR': rotation_types.count('NR') / len(rotation_types) * 100
                    }

        return self.ensemble_analysis

    def visualize(self,
                  output_dir: str,
                  individual_plots: bool = True,
                  max_individual: int = 10,
                  ensemble_plots: bool = True,
                  overwrite_individual: bool = False) -> None:
        """
        Create visualizations for the ensemble.

        Args:
            output_dir: Output directory
            individual_plots: Create plots for individual simulations
            max_individual: Max number of individual sims to plot
            ensemble_plots: Create ensemble-level plots
            overwrite_individual: If True, recreate individual plots even if they exist
        """
        os.makedirs(output_dir, exist_ok=True)

        # Individual simulation plots
        if individual_plots:
            for i, result in enumerate(self.results[:max_individual]):
                # Use the actual sim_id from the result
                sim_id = result.sim_id
                ind_dir = os.path.join(output_dir, f"simulation_{sim_id:03d}")

                # Check if individual plots already exist
                if not overwrite_individual and os.path.exists(ind_dir) and os.listdir(ind_dir):
                    continue

                # Create pipeline with the same sim_id
                pipeline = SimulationPipeline(
                    self.model_type, self.params, self.integrator_params,
                    self.analysis_functions, self.plot_functions, self.animation_function,
                    sim_id=sim_id
                )
                pipeline.visualize(result, ind_dir, animation=False)

        # Ensemble plots
        if ensemble_plots:
            for name, func in self.ensemble_plot_functions.items():
                try:
                    save_path = os.path.join(output_dir, f"ensemble_{name}.png")

                    # Call with appropriate arguments
                    import inspect
                    sig = inspect.signature(func)
                    kwargs = {'save_path': save_path}

                    if 'all_results' in sig.parameters:
                        kwargs['all_results'] = self.results
                    if 'params' in sig.parameters:
                        kwargs['params'] = self.params
                    # TODO REMOVE ME
                    if 'rotation_types' in sig.parameters:
                        kwargs['rotation_types'] = self.ensemble_analysis.get('rotation_types')

                    func(**kwargs)
                except Exception as e:
                    print(f"Ensemble plot '{name}' failed: {e}")
                    import traceback
                    traceback.print_exc()

    def run_complete(self,
                     n_simulations: int,
                     T: float = 50.0,
                     output_dir: Optional[str] = None,
                     parallel: bool = True,
                     progress: bool = True,
                     max_workers: Optional[int] = None,
                     create_individual_plots: bool = True,
                     create_individual_animations: bool = True,
                     max_individual_plots: Optional[int] = None,
                     max_individual_animations: Optional[int] = None,
                     ensemble_plots: bool = True) -> Dict[str, Any]:
        """
        Run complete ensemble pipeline.

        Args:
            n_simulations: Number of simulations to run
            T: Simulation time
            output_dir: Directory for outputs
            parallel: Use parallel execution
            progress: Show progress bar
            create_individual_plots: Create plots during simulation run
            create_individual_animations: Create animations during simulation run
            max_individual_plots: Max number of individual simulations to visualize
            ensemble_plots: Create ensemble-level plots

        Returns:
            Ensemble analysis results
        """
        # Run simulations with visualization
        self.run(n_simulations, T, parallel, progress=progress,
                output_dir=output_dir,
                max_workers=max_workers,
                create_individual_plots=create_individual_plots,
                create_individual_animations=create_individual_animations,
                max_individual_plots=max_individual_plots,
                max_individual_animations=max_individual_animations)

        # Analyze ensemble
        self.analyze()

        # Create ensemble visualizations if output directory provided
        if output_dir and ensemble_plots:
            # Only create ensemble plots, individual plots were already created
            self.visualize(output_dir, individual_plots=False, ensemble_plots=True)

        return self.ensemble_analysis
