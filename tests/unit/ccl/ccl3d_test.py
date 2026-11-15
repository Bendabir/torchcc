from __future__ import annotations

import re
from typing import Literal

import cc3d
import numpy as np
import pytest
import torch

from torchcc import ccl3d


@pytest.mark.parametrize("connectivity", [6, 18, 26])
def test_ccl3d_cpu_many_components(connectivity: Literal[6, 18, 26]) -> None:
    # Isolated voxels give 512 components : labels must not wrap around at 255.
    batch = np.zeros((2, 16, 16, 16), dtype=np.uint8)
    batch[:, ::2, ::2, ::2] = 1

    with pytest.warns(UserWarning, match="Falling back to ConnectedComponents3D"):
        labels = ccl3d(torch.from_numpy(batch), connectivity=connectivity)

    expected = np.stack([
        cc3d.connected_components(vol, connectivity=connectivity) for vol in batch
    ]).astype(np.int32)

    assert labels.dtype == torch.int32
    np.testing.assert_array_equal(labels.numpy(), expected, strict=True)


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
