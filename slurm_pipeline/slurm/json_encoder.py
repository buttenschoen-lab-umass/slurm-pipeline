import json
from dataclasses import is_dataclass, asdict
from datetime import datetime, date, time
from decimal import Decimal
from pathlib import Path
from enum import Enum
from typing import Any
import numpy as np


class DataclassJSONEncoder(json.JSONEncoder):
    """
    A JSON encoder that handles dataclasses and other common Python types
    while preserving standard JSON encoding for basic types.

    Supported types:
    - All standard JSON types (dict, list, str, int, float, bool, None)
    - Dataclasses (converted via asdict)
    - datetime, date, time (ISO format)
    - Decimal (converted to float)
    - Path (converted to string)
    - Enum (uses value)
    - sets (converted to lists)
    - numpy arrays and scalars (if numpy is available)
    """

    def default(self, obj: Any) -> Any:
        """
        Override the default method to handle custom types.

        Args:
            obj: The object to encode

        Returns:
            A JSON-serializable representation of the object

        Raises:
            TypeError: If the object is not JSON serializable
        """
        # Handle dataclasses
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)

        # Handle datetime types
        elif isinstance(obj, (datetime, date, time)):
            return obj.isoformat()

        # Handle Decimal
        elif isinstance(obj, Decimal):
            return float(obj)

        # Handle Path
        elif isinstance(obj, Path):
            return str(obj)

        # Handle Enum
        elif isinstance(obj, Enum):
            return obj.value

        # Handle sets
        elif isinstance(obj, set):
            return list(obj)

        # Handle numpy types (if numpy is available)
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
        except (NameError, AttributeError):
            # numpy is not available or obj is not a numpy type
            pass

        # Handle objects with custom serialization methods
        if hasattr(obj, '__json__'):
            return obj.__json__()
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()

        # Fall back to the parent class default method
        # This will raise TypeError for non-serializable objects
        return super().default(obj)


# Convenience functions that use the custom encoder
def dumps(obj: Any, **kwargs) -> str:
    """
    Serialize obj to a JSON formatted string using DataclassJSONEncoder.

    Args:
        obj: The object to serialize
        **kwargs: Additional arguments passed to json.dumps

    Returns:
        JSON string representation
    """
    kwargs.setdefault('cls', DataclassJSONEncoder)
    return json.dumps(obj, **kwargs)


def dump(obj: Any, fp, **kwargs) -> None:
    """
    Serialize obj as a JSON formatted stream to fp using DataclassJSONEncoder.

    Args:
        obj: The object to serialize
        fp: File-like object to write to
        **kwargs: Additional arguments passed to json.dump
    """
    kwargs.setdefault('cls', DataclassJSONEncoder)
    json.dump(obj, fp, **kwargs)


