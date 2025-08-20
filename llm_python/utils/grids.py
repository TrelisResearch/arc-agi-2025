from typing import Any
from llm_python.utils.numpy import convert_numpy_types


def grids_equal(a: Any, b: Any) -> bool:
    a = convert_numpy_types(a)
    b = convert_numpy_types(b)
    return a == b
