"""
Environment handling module for SLURM runner with testing utilities.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile


class RunnerEnvironmentModule:
    """Manages the runner environment module creation and testing."""

    # Template for the environment module
    MODULE_TEMPLATE = '''#!/usr/bin/env python
"""
Environment handling for SLURM runner.
Auto-generated module for loading captured Python environments.
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Optional


class EnvironmentLoader:
    """Handles loading and applying captured environments."""

    def __init__(self):
        self.recreated_functions: Dict[str, Any] = {}
        self.env_loaded: bool = False
        self.env_file: Optional[Path] = None

    def load_from_argv(self) -> bool:
        """Load environment from command line arguments."""
        if len(sys.argv) > 4:
            env_file = Path(sys.argv[4])
            return self.load_from_file(env_file)
        return False

    def load_from_file(self, env_file: Path) -> bool:
        """Load environment from a specific file."""
        if not env_file.exists():
            return False

        self.env_file = env_file
        print("\\n" + "="*60)
        print("LOADING CAPTURED ENVIRONMENT")
        print("="*60)
        print(f"Environment file: {env_file}")

        try:
            with open(env_file, 'rb') as f:
                env_capture = pickle.load(f)

            # Apply the environment
            self.recreated_functions = env_capture.apply_environment()
            self.env_loaded = True

            print(f"\\nEnvironment loaded successfully")
            print(f"Recreated functions: {len(self.recreated_functions)}")

            # Show first few function names
            func_names = list(self.recreated_functions.keys())
            for name in func_names[:5]:
                print(f"  - {name}")
            if len(func_names) > 5:
                print(f"  ... and {len(func_names) - 5} more")

            print("="*60 + "\\n")
            return True

        except Exception as e:
            print(f"\\nERROR loading environment: {e}")
            import traceback
            traceback.print_exc()
            print("\\nContinuing without captured environment...")
            print("="*60 + "\\n")
            return False

    def update_object(self, obj: Any) -> int:
        """Update an object's functions with recreated versions."""
        if not self.env_loaded or not self.recreated_functions:
            return 0

        print("\\nUpdating functions with recreated versions...")
        updates_made = 0

        # Function dictionary attributes
        dict_attrs = [
            'analysis_functions',
            'plot_functions',
            'ensemble_analysis_functions',
            'ensemble_plot_functions'
        ]

        # Single function attributes
        single_attrs = [
            'animation_function',
            'scan_analysis_function',
            'scan_plot_function',
            'save_function',
            'load_function'
        ]

        # Update dictionaries
        for attr_name in dict_attrs:
            if hasattr(obj, attr_name):
                func_dict = getattr(obj, attr_name)
                if func_dict and isinstance(func_dict, dict):
                    for name, func in self.recreated_functions.items():
                        if name in func_dict:
                            func_dict[name] = func
                            print(f"  ✓ Updated {attr_name}.{name}")
                            updates_made += 1

        # Update single functions
        for attr_name in single_attrs:
            if hasattr(obj, attr_name):
                current_func = getattr(obj, attr_name)
                if current_func and callable(current_func):
                    func_name = getattr(current_func, '__name__', '')
                    if func_name in self.recreated_functions:
                        setattr(obj, attr_name, self.recreated_functions[func_name])
                        print(f"  ✓ Updated {attr_name}: {func_name}")
                        updates_made += 1

        print(f"\\nTotal function updates: {updates_made}")
        return updates_made


# Global instance
_env_loader = EnvironmentLoader()

# Convenience functions for backward compatibility
RECREATED_FUNCTIONS = _env_loader.recreated_functions
ENV_LOADED = _env_loader.env_loaded


def load_environment_if_provided() -> bool:
    """Load environment from command line arguments."""
    global RECREATED_FUNCTIONS, ENV_LOADED
    result = _env_loader.load_from_argv()
    RECREATED_FUNCTIONS = _env_loader.recreated_functions
    ENV_LOADED = _env_loader.env_loaded
    return result


def update_object_functions(obj: Any) -> int:
    """Update object functions with recreated versions."""
    return _env_loader.update_object(obj)


# Auto-initialize on import
if __name__ != "__main__":
    load_environment_if_provided()
'''

    @classmethod
    def create_module_file(cls, scripts_dir: Path, filename: str = "runner_env.py") -> Path:
        """
        Create the runner environment module file.

        Args:
            scripts_dir: Directory to create the module in
            filename: Name for the module file

        Returns:
            Path to the created module
        """
        module_path = scripts_dir / filename

        with open(module_path, 'w') as f:
            f.write(cls.MODULE_TEMPLATE)

        module_path.chmod(0o755)
        return module_path

    @classmethod
    def test_module(cls) -> bool:
        """Test that the module can be created and imported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create module
            module_path = cls.create_module_file(tmppath)

            # Try to import it
            sys.path.insert(0, str(tmppath))
            try:
                import runner_env
                # Check expected functions exist
                assert hasattr(runner_env, 'update_object_functions')
                assert hasattr(runner_env, 'load_environment_if_provided')
                assert hasattr(runner_env, 'EnvironmentLoader')
                return True
            except Exception as e:
                print(f"Module test failed: {e}")
                return False
            finally:
                sys.path.remove(str(tmppath))
