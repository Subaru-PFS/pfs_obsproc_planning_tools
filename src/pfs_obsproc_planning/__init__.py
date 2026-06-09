#!/usr/bin/env python3

import os
from importlib.metadata import PackageNotFoundError, version

os.environ.setdefault("LOGURU_LEVEL", "INFO")

try:
    __version__ = version("pfs_obsproc_planning")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
