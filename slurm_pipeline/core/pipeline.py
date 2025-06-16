import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from dataclasses import dataclass
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


@dataclass
class SimulationResult:
    """Container for simulation results"""
    time: np.ndarray
    solution: np.ndarray
    params: Dict[str, Any]
    sim_id: int
    analysis: Optional[Dict[str, Any]] = None
    output_dir: Optional[str] = None


class SimulationPipeline:
    def __init__(self,
                 model_type: type,
                 params: Dict[str, Any],
                 integrator_params: Dict[str, Any],
                 analysis_functions: Optional[Dict[str, Callable]] = None,
                 plot_functions: Optional[Dict[str, Callable]] = None,
                 animation_function: Optional[Callable] = None,
                 sim_id: int = 0):
        """
        Initialize pipeline for a single simulation.

        Args:
            model_type: Model class to instantiate (must have save_trajectory/load_trajectory methods)
            params: Model parameters
            integrator_params: Integration parameters (dt, etc.)
            analysis_functions: Dict of functions for analysis
            plot_functions: Dict of functions for plotting
            animation_function: Function for animation
            sim_id: Simulation ID (default: 0)
        """
        self.model_type = model_type
        self.params = params
        self.integrator_params = integrator_params
        self.sim_id = sim_id

        # Analysis and visualization
        self.analysis_functions = analysis_functions or {}
        self.plot_functions = plot_functions or {}
        self.animation_function = animation_function

        # Create model instance
        self.model = model_type(**params)

    def save_readable_parameters(self, output_dir: str):
        """Save parameters in human-readable format."""
        params_file = Path(output_dir) / "parameters.txt"

        with open(params_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SIMULATION PARAMETERS\n")
            f.write("="*60 + "\n\n")

            f.write(f"Simulation ID: {self.sim_id}\n")
            f.write(f"Model Type: {self.model_type.__name__}\n\n")

            # Model parameters
            if hasattr(self.model, 'get_parameter_summary'):
                f.write("Model Configuration:\n")
                f.write("-"*30 + "\n")
                f.write(self.model.get_parameter_summary())
                f.write("\n")

            # Integration parameters
            f.write("Integration Parameters:\n")
            f.write("-"*30 + "\n")
            for key, value in self.integrator_params.items():
                f.write(f"  {key:20s}: {value}\n")

            f.write("\n" + "="*60 + "\n")

    def save(self, result: SimulationResult, filepath: str) -> None:
        """Save simulation data using model's save_trajectory method."""
        if not hasattr(self.model_type, 'save_trajectory'):
            raise AttributeError(
                f"Model {self.model_type.__name__} must implement save_trajectory class method"
            )

        self.model_type.save_trajectory(
            result.time, result.solution, result.params,
            filepath, sim_id=result.sim_id
        )

    def load(self, filepath: str) -> SimulationResult:
        """Load simulation data using model's load_trajectory method."""
        if not hasattr(self.model_type, 'load_trajectory'):
            raise AttributeError(
                f"Model {self.model_type.__name__} must implement load_trajectory class method"
            )

        data = self.model_type.load_trajectory(filepath)

        if isinstance(data, tuple) and len(data) == 4:
            t, solution, params, sim_id = data
        else:
            raise ValueError(f"Unexpected data format from load_trajectory: {type(data)}")

        return SimulationResult(time=t, solution=solution, params=params, sim_id=sim_id)

    def run(self,
            T: float = 50.0,
            initial_state: Optional[np.ndarray] = None,
            progress: bool = False) -> SimulationResult:
        """
        Run the simulation.

        Args:
            T: Simulation time
            initial_state: Initial conditions (uses random if None)
            progress: Show progress bar

        Returns:
            SimulationResult object
        """
        # Run simulation
        t, solution = self.model.simulate(
            T=T,
            initial_state=initial_state,
            progress=progress,
            **self.integrator_params
        )

        return SimulationResult(
            time=t,
            solution=solution,
            params=self.params,
            sim_id=self.sim_id
        )

    def analyze(self, result: SimulationResult) -> SimulationResult:
        """
        Run all analysis functions on the result.

        Args:
            result: SimulationResult to analyze

        Returns:
            SimulationResult with analysis added
        """
        analysis = {}

        for name, func in self.analysis_functions.items():
            try:
                analysis[name] = func(result.time, result.solution, self.model)
            except Exception as e:
                print(f"Analysis '{name}' failed: {e}")
                analysis[name] = None

        result.analysis = analysis
        return result

    def visualize(self,
                  result: SimulationResult,
                  output_dir: str,
                  plots: bool = True,
                  animation: bool = True) -> SimulationResult:
        """
        Create visualizations for the simulation.

        Args:
            result: SimulationResult to visualize
            output_dir: Directory for outputs
            plots: Create static plots
            animation: Create animation

        Returns:
            SimulationResult with output_dir set
        """
        os.makedirs(output_dir, exist_ok=True)
        result.output_dir = output_dir

        # Create plots
        if plots:
            for name, func in self.plot_functions.items():
                try:
                    save_path = os.path.join(output_dir, f"{name}.png")
                    fig = func(result.time, result.solution, model=self.model, save_path=save_path)

                    # If the function doesn't save the figure itself, close it
                    if fig is not None and isinstance(fig, plt.Figure):
                        plt.close(fig)
                except Exception as e:
                    print(f"Plot '{name}' failed: {e}")
                    # Make sure any open figures are closed on error
                    plt.close('all')

        # Create animation
        if animation and self.animation_function:
            try:
                anim_path = os.path.join(output_dir, "animation.mp4")
                self.animation_function(result.time, result.solution,
                                      model=self.model, save_path=anim_path)
            except Exception as e:
                print(f"Animation failed: {e}")

        return result

    def run_complete(self,
                     T: float = 50.0,
                     initial_state: Optional[np.ndarray] = None,
                     output_dir: Optional[str] = None,
                     save_data: bool = True,
                     progress: bool = True) -> SimulationResult:
        """
        Run complete pipeline: simulate, analyze, visualize, save.
        """
        # Run simulation
        result = self.run(T, initial_state, progress)

        # Analyze
        result = self.analyze(result)

        # Visualize if output directory provided
        if output_dir:
            result = self.visualize(result, output_dir)

            # Save parameters in human readable form
            self.save_readable_parameters(output_dir)

            # Save data
            if save_data:
                data_path = os.path.join(output_dir, "simulation_data.npz")
                self.save(result, data_path)

        return result


