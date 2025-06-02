"""
Environment capture for SLURM pipeline execution.
"""

import os
import sys
import inspect
import importlib.util
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path
from datetime import datetime
import types
from collections.abc import Callable
import ast

from .symbol_tracker import SymbolTracker


class EnvironmentCapture:
    """Captures the current Python environment for remote execution."""

    def __init__(self, logger):
        self.logger = logger
        self.python_paths = []
        self.env_vars = {}
        self.symbol_tracker = SymbolTracker(logger)

        # Captured content
        self.inline_functions = {}  # Locally defined functions to recreate
        self.required_modules = set()  # Modules that must be available
        self.module_imports = {}  # Import statements to execute

        # Configuration
        self.capture_inline_functions = True
        self.verify_module_availability = True

    def capture_current_environment(self):
        """Capture the current Python environment."""
        self.logger.info("="*60)
        self.logger.info("CAPTURING PYTHON ENVIRONMENT")
        self.logger.info("="*60)

        # Capture Python path
        self.python_paths = sys.path.copy()
        self.logger.info(f"Captured {len(self.python_paths)} Python paths")

        # Capture relevant environment variables
        for var in ['PYTHONPATH', 'LD_LIBRARY_PATH', 'PATH', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV']:
            if var in os.environ:
                self.env_vars[var] = os.environ[var]
                self.logger.info(f"  {var}: {os.environ[var][:50]}...")

        # Find the calling script's directory
        self._add_script_directory()

    def _add_script_directory(self):
        """Add the calling script's directory to Python path."""
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the user's script
            while frame.f_back:
                frame = frame.f_back
                filename = frame.f_code.co_filename
                if not filename.startswith('<') and 'slurm_pipeline' not in filename:
                    script_dir = Path(filename).parent.resolve()
                    script_dir_str = str(script_dir)

                    if script_dir_str not in self.python_paths:
                        self.python_paths.insert(0, script_dir_str)
                        self.logger.info(f"Added script directory: {script_dir}")

                    # Also add parent directory if it contains __init__.py
                    parent_dir = script_dir.parent
                    if (parent_dir / '__init__.py').exists():
                        parent_dir_str = str(parent_dir)
                        if parent_dir_str not in self.python_paths:
                            self.python_paths.insert(0, parent_dir_str)
                            self.logger.info(f"Added parent package directory: {parent_dir}")
                    break
        finally:
            del frame

    def capture_object_dependencies(self, obj: Any, name: str = "main_object",
                                  analysis_functions: Optional[Dict[str, Callable]] = None,
                                  plot_functions: Optional[Dict[str, Callable]] = None,
                                  ensemble_analysis_functions: Optional[Dict[str, Callable]] = None,
                                  ensemble_plot_functions: Optional[Dict[str, Callable]] = None):
        """
        Capture all dependencies of an object and its associated functions.

        Args:
            obj: Main object to analyze
            name: Name for the object
            analysis_functions: Dictionary of analysis functions
            plot_functions: Dictionary of plot functions
            ensemble_analysis_functions: Dictionary of ensemble analysis functions
            ensemble_plot_functions: Dictionary of ensemble plot functions
        """
        self.logger.info(f"\nAnalyzing dependencies for {name}...")

        # Analyze main object
        self.symbol_tracker.analyze_object(obj, name)

        # Analyze all provided functions
        all_functions = {}

        if analysis_functions:
            self.logger.info(f"\nAnalyzing {len(analysis_functions)} analysis functions...")
            all_functions.update(analysis_functions)
            for func_name, func in analysis_functions.items():
                self.symbol_tracker.analyze_object(func, f"analysis_functions.{func_name}")

        if plot_functions:
            self.logger.info(f"\nAnalyzing {len(plot_functions)} plot functions...")
            all_functions.update(plot_functions)
            for func_name, func in plot_functions.items():
                self.symbol_tracker.analyze_object(func, f"plot_functions.{func_name}")

        if ensemble_analysis_functions:
            self.logger.info(f"\nAnalyzing {len(ensemble_analysis_functions)} ensemble analysis functions...")
            all_functions.update(ensemble_analysis_functions)
            for func_name, func in ensemble_analysis_functions.items():
                self.symbol_tracker.analyze_object(func, f"ensemble_analysis_functions.{func_name}")

        if ensemble_plot_functions:
            self.logger.info(f"\nAnalyzing {len(ensemble_plot_functions)} ensemble plot functions...")
            all_functions.update(ensemble_plot_functions)
            for func_name, func in ensemble_plot_functions.items():
                self.symbol_tracker.analyze_object(func, f"ensemble_plot_functions.{func_name}")

        # Process captured symbols
        self._process_captured_symbols(all_functions)

        # Log summary
        summary = self.symbol_tracker.get_summary()
        self.logger.info("\n" + "="*60)
        self.logger.info("CAPTURE SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Required modules: {summary['modules']}")
        self.logger.info(f"Inline functions: {summary['inline_code']}")
        self.logger.info(f"Classes found: {summary['classes']}")
        self.logger.info(f"Functions found: {summary['functions']}")

        if summary['missing_modules']:
            self.logger.warning(f"\nMissing modules: {summary['missing_modules']}")

        if summary['inline_functions']:
            self.logger.info(f"\nCaptured inline functions:")
            for func_name in summary['inline_functions']:
                self.logger.info(f"  - {func_name}")

    def _process_captured_symbols(self, all_functions: Dict[str, Callable]):
        """Process captured symbols to determine what needs to be transmitted."""

        # Get required modules
        self.required_modules = self.symbol_tracker.get_required_modules()
        self.logger.info(f"\nRequired modules: {', '.join(sorted(self.required_modules))}")

        # Capture inline functions
        inline_functions = self.symbol_tracker.get_inline_functions()

        # Also check the provided function dictionaries for local functions
        for func_name, func in all_functions.items():
            if callable(func) and getattr(func, '__module__', None) == '__main__':
                try:
                    source = inspect.getsource(func)
                    self.inline_functions[func_name] = {
                        'source': source,
                        'name': func.__name__,
                        'original_name': func_name,
                        'type': 'function'
                    }
                    self.logger.info(f"  Captured inline function: {func_name}")
                except Exception as e:
                    self.logger.warning(f"  Could not capture source for {func_name}: {e}")

        # Add inline functions from symbol tracker
        for key, info in inline_functions.items():
            if info['source'] and key not in self.inline_functions:
                self.inline_functions[key] = info

        # Generate import statements for required modules
        self._generate_import_statements()

    def _generate_import_statements(self):
        """Generate import statements for required modules."""
        for module_name in sorted(self.required_modules):
            # Check if it's a package or module
            module_info = self.symbol_tracker.captured_symbols['modules'].get(module_name, {})

            if module_info.get('is_package'):
                # For packages, we might need to import submodules
                self.module_imports[module_name] = f"import {module_name}"
            else:
                # For regular modules
                self.module_imports[module_name] = f"import {module_name}"

            # Also check if we need from imports
            for func_key, func_info in self.symbol_tracker.captured_symbols['functions'].items():
                if func_info['module'] == module_name:
                    func_name = func_info['name']
                    import_key = f"{module_name}.{func_name}"
                    self.module_imports[import_key] = f"from {module_name} import {func_name}"

    def apply_environment(self):
        """Apply the captured environment in the remote execution context."""
        self.logger.info("\n" + "="*60)
        self.logger.info("APPLYING CAPTURED ENVIRONMENT")
        self.logger.info("="*60)

        # Update Python path
        for path in self.python_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
                self.logger.info(f"Added to path: {path}")

        # Update environment variables
        for var, value in self.env_vars.items():
            if var not in os.environ or os.environ[var] != value:
                os.environ[var] = value
                self.logger.info(f"Set {var}")

        # Verify required modules are available
        missing_modules = []
        if self.verify_module_availability:
            self.logger.info(f"\nVerifying {len(self.required_modules)} required modules...")
            for module_name in sorted(self.required_modules):
                try:
                    importlib.import_module(module_name)
                    self.logger.info(f"  ✓ {module_name}")
                except ImportError as e:
                    missing_modules.append(module_name)
                    self.logger.error(f"  ✗ {module_name}: {e}")

        if missing_modules:
            self.logger.error(f"\nMissing modules on compute node: {missing_modules}")
            self.logger.error("These modules must be installed in the compute environment!")
            # Note: We don't raise an exception here because some modules might be optional

        # Execute import statements
        import_namespace = {}
        self.logger.info(f"\nExecuting {len(self.module_imports)} import statements...")
        for import_key, import_stmt in self.module_imports.items():
            try:
                exec(import_stmt, import_namespace)
                self.logger.debug(f"  ✓ {import_stmt}")
            except ImportError as e:
                self.logger.warning(f"  ✗ {import_stmt}: {e}")

        # Recreate inline functions
        recreated_functions = {}
        self.logger.info(f"\nRecreating {len(self.inline_functions)} inline functions...")

        # Create a namespace with common imports and the imported modules
        namespace = {
            '__builtins__': __builtins__,
            'np': None,
            'numpy': None,
            'plt': None,
            'matplotlib': None,
            'pd': None,
            'pandas': None,
        }

        # Try to import common modules into namespace
        for module_name, import_name in [('numpy', 'np'), ('numpy', 'numpy'),
                                        ('matplotlib.pyplot', 'plt'), ('matplotlib', 'matplotlib'),
                                        ('pandas', 'pd'), ('pandas', 'pandas')]:
            try:
                namespace[import_name] = importlib.import_module(module_name)
            except ImportError:
                pass

        # Add imported modules to namespace
        namespace.update(import_namespace)

        # Recreate functions
        for func_key, func_info in self.inline_functions.items():
            try:
                # Execute the function definition
                exec(func_info['source'], namespace)

                # Find the function in the namespace
                func_name = func_info.get('name', func_info.get('original_name'))
                if func_name in namespace:
                    recreated_functions[func_info.get('original_name', func_name)] = namespace[func_name]
                    self.logger.info(f"  ✓ Recreated: {func_info.get('original_name', func_name)}")
                else:
                    # Try to find it by searching for def statements
                    tree = ast.parse(func_info['source'])
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if node.name in namespace:
                                recreated_functions[func_info.get('original_name', node.name)] = namespace[node.name]
                                self.logger.info(f"  ✓ Recreated: {func_info.get('original_name', node.name)}")
                                break

            except Exception as e:
                self.logger.error(f"  ✗ Failed to recreate {func_key}: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        self.logger.info(f"\nEnvironment application complete")
        self.logger.info(f"Recreated {len(recreated_functions)} functions")
        self.logger.info("="*60 + "\n")

        return recreated_functions

    def get_capture_summary(self) -> Dict[str, Any]:
        """Get a summary of what was captured."""
        symbol_summary = self.symbol_tracker.get_summary()

        return {
            'timestamp': datetime.now().isoformat(),
            'python_paths': self.python_paths[:5],  # First 5 paths
            'environment_vars': list(self.env_vars.keys()),
            'required_modules': sorted(list(self.required_modules)),
            'inline_functions': list(self.inline_functions.keys()),
            'symbol_summary': symbol_summary,
            'total_paths': len(self.python_paths),
            'total_inline_functions': len(self.inline_functions),
            'total_required_modules': len(self.required_modules)
        }
