#!/usr/bin/env python3

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pfs_obsproc_planning")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
