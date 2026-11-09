"""Locate the current interpreter's optional native kernel for benchmarks."""

import importlib.machinery
import importlib.util
from pathlib import Path


def library_path():
    """Select an ABI-compatible extension on Unix or Windows."""
    directory = Path(__file__).resolve().parents[1] / "src" / "algroots"
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = directory / ("_native_f4" + suffix)
        if path.is_file():
            return path
    spec = importlib.util.find_spec("algroots._native_f4")
    if spec is not None and spec.origin is not None:
        return Path(spec.origin)
    raise RuntimeError("build the optional algroots native kernel first")


def load_native(path=None):
    """Load a standalone kernel, optionally from an explicit comparison build."""
    path = library_path() if path is None else Path(path)
    spec = importlib.util.spec_from_file_location("_native_f4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load native kernel: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
