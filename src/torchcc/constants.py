"""Copyright (c) 2026 Bendabir."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final


# Number of dimensions of a batch of images [N, H, W]
BATCH_2D_DIM: Final = 3
# Number of dimensions of a batch of volumes [N, H, W, D]
BATCH_3D_DIM: Final = 4
