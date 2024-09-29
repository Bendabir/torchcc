"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import importlib.metadata

from .ccl import ccl2d, ccl3d

__version__ = importlib.metadata.version("torchcc")

__all__ = ["ccl2d", "ccl3d"]
