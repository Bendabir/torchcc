from __future__ import annotations

import re
from typing import Literal

import pytest
import torch

from torchcc import ccl3d


@pytest.mark.parametrize("connectivity", [-1, 0, 1, 2, 4, 8])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_ccl3d_connectivity(connectivity: int, device: torch.device) -> None:
    volumes = torch.zeros((4, 16, 16, 16), device=device, dtype=torch.uint8)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Only 6-connectivity, 18-connectivity and 26-connectivity are supported."
        ),
    ):
        _ = ccl3d(volumes, connectivity=connectivity)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("connectivity", [6, 18, 26])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_ccl3d_not_implemented(
    connectivity: Literal[6, 18, 26],
    device: torch.device,
) -> None:
    volumes = torch.zeros((4, 16, 16, 16), device=device, dtype=torch.uint8)

    with pytest.raises(NotImplementedError):
        _ = ccl3d(volumes, connectivity=connectivity)
