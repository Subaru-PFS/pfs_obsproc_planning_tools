#!/usr/bin/env python3

import logging
import os
from importlib.metadata import PackageNotFoundError, version

os.environ.setdefault("LOGURU_LEVEL", "INFO")

# Keep the external CobraCoach library quiet by default; it emits many INFO-level
# messages during each optimization iteration. Users can opt back in by raising the
# logger level in the process if needed.
for _name in ("cobraCoach", "ics.cobraCharmer", "ics.cobraOps"):
    logging.getLogger(_name).setLevel(logging.WARNING)
    logging.getLogger(_name).propagate = False

try:
    __version__ = version("pfs_obsproc_planning")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
