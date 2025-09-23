from typing import Any
import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization"""

    if isinstance(obj, np.ndarray):
        # If the dtype is object, it might contain other numpy types that need conversion.
        if obj.dtype == "object":
            return [convert_numpy_types(item) for item in obj]
        # For other dtypes, tolist() is efficient and sufficient.
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj
