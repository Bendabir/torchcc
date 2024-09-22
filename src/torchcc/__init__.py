"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import importlib.metadata

from .cc import cc2d, cc3d

__version__ = importlib.metadata.version("torchcc")

__all__ = ["cc2d", "cc3d"]
