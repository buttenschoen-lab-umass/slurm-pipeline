"""
Generator for SLURM runner scripts.

This module provides the runner script content by reading it from the runner.py file.
"""

from pathlib import Path


def get_runner_script_content() -> str:
    """
    Get the content of the runner script by reading runner.py.

    Returns:
        String containing the complete runner script

    Raises:
        FileNotFoundError: If runner.py cannot be found
    """
    # Get the path to runner.py in the same directory
    current_dir = Path(__file__).parent
    runner_file = current_dir / "runner.py"

    if not runner_file.exists():
        raise FileNotFoundError(
            f"runner.py not found at {runner_file}. "
            "Make sure runner.py exists in the slurm directory."
        )

    # Read and return the content
    with open(runner_file, 'r') as f:
        return f.read()
