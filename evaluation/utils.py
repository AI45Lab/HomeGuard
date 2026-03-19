import contextlib
from functools import partial
import importlib
import os
import sys
from typing import Any, Dict, List


@contextlib.contextmanager
def add_sys_path(path: str | List[str]):
    """A context manager to temporarily add path(s) to the beginning of sys.path.

    This ensures that only paths not already in sys.path are added, and only
    those added paths are removed upon exit, making it safe and idempotent.

    Args:
        path (str | List[str]): A single path or a list of paths to add.
    """
    if isinstance(path, str):
        path = [path]
    
    paths_to_add = [p for p in path if p not in sys.path and os.path.exists(p)]

    for p in paths_to_add[::-1]:
        sys.path.insert(0, p)

    try:
        yield
    finally:
        for p in paths_to_add:
            if p in sys.path:
                sys.path.remove(p)
