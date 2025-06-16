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
            data = asdict(obj)

            # Add class metadata if the class has from_dict method
            obj_class = obj.__class__
            if hasattr(obj_class, 'from_dict'):
                data['__class_module__'] = obj_class.__module__
                data['__class_name__'] = obj_class.__name__

            return data

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
            data = obj.to_dict()

            # Add class metadata if the class has from_dict method
            if hasattr(obj.__class__, 'from_dict'):
                if isinstance(data, dict):
                    data['__class_module__'] = obj.__class__.__module__
                    data['__class_name__'] = obj.__class__.__name__

            return data

        # Handle objects that have __dict__ and from_dict (but aren't dataclasses)
        elif hasattr(obj, '__dict__') and hasattr(obj.__class__, 'from_dict'):
            data = vars(obj).copy()
            data['__class_module__'] = obj.__class__.__module__
            data['__class_name__'] = obj.__class__.__name__
            return data

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


def reconstruct_from_dict(data: dict) -> Any:
    """
    Attempt to reconstruct an object from a dictionary with class metadata.

    Args:
        data: Dictionary potentially containing __class_module__ and __class_name__

    Returns:
        Reconstructed object if successful, otherwise the original dict
    """
    if not isinstance(data, dict):
        return data

    if '__class_module__' in data and '__class_name__' in data:
        module_name = data.pop('__class_module__')
        class_name = data.pop('__class_name__')

        try:
            import importlib
            module = importlib.import_module(module_name)
            obj_class = getattr(module, class_name)

            if hasattr(obj_class, 'from_dict'):
                return obj_class.from_dict(data)
            else:
                # Restore metadata if we can't reconstruct
                data['__class_module__'] = module_name
                data['__class_name__'] = class_name
        except Exception:
            # Restore metadata if reconstruction fails
            data['__class_module__'] = module_name
            data['__class_name__'] = class_name

    return data


def loads(s: str, **kwargs) -> Any:
    """
    Deserialize JSON string, attempting to reconstruct objects with class metadata.

    Args:
        s: JSON string
        **kwargs: Additional arguments passed to json.loads

    Returns:
        Deserialized object
    """
    data = json.loads(s, **kwargs)
    return reconstruct_from_dict(data)


def load(fp, **kwargs) -> Any:
    """
    Deserialize JSON from file, attempting to reconstruct objects with class metadata.

    Args:
        fp: File-like object to read from
        **kwargs: Additional arguments passed to json.load

    Returns:
        Deserialized object
    """
    data = json.load(fp, **kwargs)
    return reconstruct_from_dict(data)
