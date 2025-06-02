"""
Symbol tracker for analyzing Python objects and their dependencies.
"""

import sys
import ast
import inspect
import types
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from pathlib import Path
import importlib


class SymbolTracker:
    """Tracks symbols (functions, classes, modules) used by objects."""

    def __init__(self, logger):
        self.logger = logger
        self.captured_symbols = {
            'modules': {},
            'functions': {},
            'classes': {},
            'variables': {},
            'inline_code': {}  # For locally defined functions/lambdas
        }
        self.missing_symbols = {
            'modules': set(),
            'functions': set(),
            'classes': set()
        }

        # Standard library modules that don't need to be captured
        self.stdlib_modules = {
            'builtins', '__builtin__', 'sys', 'os', 'time', 'datetime',
            'collections', 'itertools', 'functools', 'typing', 'types',
            'json', 'pickle', 'pathlib', 're', 'math', 'random',
            'subprocess', 'threading', 'multiprocessing', 'concurrent',
            'warnings', 'logging', 'traceback', 'inspect', 'ast',
            'copy', 'weakref', 'gc', 'importlib', 'pkgutil',
            'dataclasses', 'enum', 'abc', 'contextlib', 'operator'
        }

        # Scientific packages assumed to be installed
        self.scientific_modules = {
            'numpy', 'scipy', 'matplotlib', 'pandas', 'sklearn',
            'tensorflow', 'torch', 'jax', 'numba', 'cython',
            'h5py', 'netCDF4', 'xarray', 'dask', 'joblib',
            'tqdm', 'ipython', 'jupyter', 'notebook'
        }

        # All modules to skip
        self.skip_modules = self.stdlib_modules | self.scientific_modules

    def analyze_object(self, obj: Any, name: str = 'root'):
        """Recursively analyze an object to find all symbols it uses."""
        self.logger.info(f"\nAnalyzing object: {name} (type: {type(obj).__name__})")
        visited = set()
        self._analyze_recursive(obj, name, visited, depth=0)
        self._analyze_dependencies()

    def _analyze_recursive(self, obj: Any, path: str, visited: Set[int], depth: int = 0):
        """Recursively analyze object attributes."""
        # Limit recursion depth
        if depth > 10:
            return

        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        # Skip None, primitives, and some standard types
        if obj is None or isinstance(obj, (int, float, str, bytes, bool)):
            return

        # Analyze based on object type
        if isinstance(obj, type):
            # It's a class
            self._analyze_class(obj, path)

        elif callable(obj):
            # It's a function or method
            self._analyze_callable(obj, path)

        elif isinstance(obj, dict):
            # Analyze dictionary values
            for key, value in obj.items():
                if isinstance(key, str) and not key.startswith('_'):
                    self._analyze_recursive(value, f"{path}['{key}']", visited, depth + 1)

        elif isinstance(obj, (list, tuple, set)):
            # Analyze collection items
            for i, item in enumerate(obj):
                self._analyze_recursive(item, f"{path}[{i}]", visited, depth + 1)

        elif hasattr(obj, '__dict__'):
            # Analyze object attributes
            try:
                for attr_name, attr_value in obj.__dict__.items():
                    if not attr_name.startswith('_'):
                        self._analyze_recursive(attr_value, f"{path}.{attr_name}", visited, depth + 1)
            except Exception as e:
                self.logger.debug(f"Could not analyze __dict__ of {path}: {e}")

        # Also check class of the object
        obj_type = type(obj)
        if obj_type.__module__ not in self.skip_modules:
            self._analyze_class(obj_type, f"{path}.__class__")

    def _analyze_class(self, cls: type, path: str):
        """Analyze a class."""
        module_name = cls.__module__
        class_name = cls.__qualname__

        if module_name in self.skip_modules or module_name == '__main__':
            return

        key = f"{module_name}.{class_name}"
        if key not in self.captured_symbols['classes']:
            self.captured_symbols['classes'][key] = {
                'module': module_name,
                'name': class_name,
                'path': path,
                'file': self._get_module_file(module_name)
            }
            self.logger.debug(f"  Found class: {key}")

            # Track the module
            self._track_module(module_name)

    def _analyze_callable(self, func: callable, path: str):
        """Analyze a callable object (function, method, lambda)."""
        # Get function details
        func_module = getattr(func, '__module__', None)
        func_name = getattr(func, '__name__', '<lambda>')

        if not func_module:
            return

        # Special handling for local functions
        if func_module == '__main__':
            # This is a locally defined function
            self._capture_local_function(func, path)
            return

        if func_module in self.skip_modules:
            return

        # Track imported function
        key = f"{func_module}.{func_name}"
        if key not in self.captured_symbols['functions']:
            self.captured_symbols['functions'][key] = {
                'module': func_module,
                'name': func_name,
                'path': path,
                'file': self._get_module_file(func_module),
                'is_method': inspect.ismethod(func),
                'is_builtin': inspect.isbuiltin(func)
            }
            self.logger.debug(f"  Found function: {key}")

            # Track the module
            self._track_module(func_module)

        # Analyze function's closure and globals
        self._analyze_function_dependencies(func, path)

    def _capture_local_function(self, func: callable, path: str):
        """Capture a locally defined function."""
        func_name = getattr(func, '__name__', '<lambda>')
        key = f"local.{path}.{func_name}"

        try:
            source = inspect.getsource(func)

            self.captured_symbols['inline_code'][key] = {
                'type': 'function',
                'name': func_name,
                'path': path,
                'source': source,
                'is_lambda': func_name == '<lambda>',
                'closure_vars': self._get_closure_vars(func)
            }
            self.logger.info(f"  Captured local function: {func_name} at {path}")

            # Parse the source to find dependencies
            self._analyze_source_dependencies(source, f"{path}.{func_name}")

        except Exception as e:
            self.logger.warning(f"  Could not capture source for {func_name}: {e}")

    def _analyze_function_dependencies(self, func: callable, path: str):
        """Analyze dependencies of a function."""
        # Check closure variables
        if hasattr(func, '__closure__') and func.__closure__:
            for i, cell in enumerate(func.__closure__):
                try:
                    cell_contents = cell.cell_contents
                    self._analyze_recursive(cell_contents, f"{path}.__closure__[{i}]", set(), depth=5)
                except ValueError:
                    pass  # Empty cell

        # Check defaults
        if hasattr(func, '__defaults__') and func.__defaults__:
            for i, default in enumerate(func.__defaults__):
                self._analyze_recursive(default, f"{path}.__defaults__[{i}]", set(), depth=5)

    def _analyze_source_dependencies(self, source: str, context: str):
        """Parse source code to find import dependencies."""
        try:
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        if module_name not in self.skip_modules:
                            self._track_module(module_name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.skip_modules:
                        self._track_module(node.module)

                elif isinstance(node, ast.Name):
                    # Track potential global references
                    if node.id not in {'True', 'False', 'None'}:
                        self.captured_symbols['variables'][f"{context}.{node.id}"] = {
                            'name': node.id,
                            'context': context
                        }

        except Exception as e:
            self.logger.debug(f"Could not parse source for {context}: {e}")

    def _track_module(self, module_name: str):
        """Track a module that needs to be available."""
        if module_name not in self.skip_modules and module_name not in self.captured_symbols['modules']:
            self.captured_symbols['modules'][module_name] = {
                'name': module_name,
                'file': self._get_module_file(module_name),
                'is_package': self._is_package(module_name)
            }
            self.logger.debug(f"  Tracking module: {module_name}")

    def _get_module_file(self, module_name: str) -> Optional[str]:
        """Get the file path for a module."""
        try:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                return getattr(module, '__file__', None)
            else:
                spec = importlib.util.find_spec(module_name)
                if spec and spec.origin:
                    return spec.origin
        except Exception:
            pass
        return None

    def _is_package(self, module_name: str) -> bool:
        """Check if a module is a package."""
        try:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                return hasattr(module, '__path__')
            else:
                spec = importlib.util.find_spec(module_name)
                return spec is not None and spec.submodule_search_locations is not None
        except Exception:
            return False

    def _get_closure_vars(self, func: callable) -> Dict[str, Any]:
        """Extract closure variables from a function."""
        closure_vars = {}

        if hasattr(func, '__code__') and hasattr(func, '__closure__'):
            if func.__closure__:
                for name, cell in zip(func.__code__.co_freevars, func.__closure__):
                    try:
                        closure_vars[name] = {
                            'type': type(cell.cell_contents).__name__,
                            'repr': repr(cell.cell_contents)[:100]
                        }
                    except ValueError:
                        closure_vars[name] = {'type': 'empty_cell'}

        return closure_vars

    def _analyze_dependencies(self):
        """Analyze captured symbols to find missing dependencies."""
        # Check which modules are actually available
        for module_name in list(self.captured_symbols['modules'].keys()):
            try:
                importlib.import_module(module_name)
            except ImportError:
                self.missing_symbols['modules'].add(module_name)
                self.logger.warning(f"  Missing module: {module_name}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of captured symbols."""
        return {
            'modules': len(self.captured_symbols['modules']),
            'functions': len(self.captured_symbols['functions']),
            'classes': len(self.captured_symbols['classes']),
            'inline_code': len(self.captured_symbols['inline_code']),
            'missing_modules': list(self.missing_symbols['modules']),
            'module_list': list(self.captured_symbols['modules'].keys()),
            'inline_functions': [
                info['name'] for info in self.captured_symbols['inline_code'].values()
                if info['type'] == 'function'
            ]
        }

    def get_required_modules(self) -> Set[str]:
        """Get the set of required non-standard modules."""
        required = set()

        for module_info in self.captured_symbols['modules'].values():
            module_name = module_info['name']
            if module_name not in self.skip_modules:
                required.add(module_name)

        return required

    def get_inline_functions(self) -> Dict[str, Dict[str, Any]]:
        """Get all captured inline/local functions."""
        return {
            name: info
            for name, info in self.captured_symbols['inline_code'].items()
            if info['type'] == 'function'
        }
