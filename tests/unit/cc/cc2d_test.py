from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import pytest
import torch

from torchcc import cc2d


# NOTE : Replace with manually generated tests (so we can also check CPU) ?
@pytest.mark.parametrize(
    "size",
    [
        (1024, 1024),
        (8, 1024, 1024),
        (1024, 2048),
        (8, 1024, 2048),
        (1024, 1023),
        (7, 1024, 1023),
        (1023, 1024),
        (7, 1023, 1024),
        (1023, 1023),
        (7, 1023, 1023),
    ],
    ids=[
        "square",
        "square-batch",
        "rectangle",
        "rectangle-batch",
        "odd-height",
        "odd-height-batch",
        "odd-width",
        "odd-width-batch",
        "odd",
        "odd-batch",
    ],
)
@pytest.mark.parametrize(
    "contiguous",
    [True, False],
    ids=["contiguous", "non-contiguous"],
)
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_cc2d(
    generator: torch.Generator,
    device: torch.device,
    size: tuple[int, ...],
    contiguous: bool,  # noqa: FBT001
    connectivity: Literal[4, 8],
) -> None:
    x = torch.randint(
        low=0,
        high=2,
        size=size,
        generator=generator,
        dtype=torch.uint8,
        device=device,
    )

    if not contiguous:
        x = x.transpose(-2, -1).contiguous().transpose(-2, -1)

    labels = cc2d(x, connectivity=connectivity)
    _x = x.cpu().numpy()
    _labels = labels.cpu().numpy()
    expected = np.zeros(size, dtype=np.uint8)

    # Compare to what OpenCV would produce
    if len(size) == 2:
        _, expected = cv2.connectedComponents(_x, connectivity=connectivity)

    if len(size) == 3:
        for i in range(len(_x)):
            _, expected[i] = cv2.connectedComponents(_x[i], connectivity=connectivity)

    # Use NumPy for better explainability
    np.testing.assert_array_equal(_labels, expected, strict=True)
