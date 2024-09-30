from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
import pytest
import torch

from torchcc import ccl2d


def consecutivize(labels: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    ids: dict[int, int] = {
        id_: i for i, id_ in enumerate(sorted(np.unique(labels).tolist())) if id_ > 0
    }
    new_labels = np.zeros_like(labels)

    for id_, i in ids.items():
        mask = labels == id_
        new_labels[mask] = i

    return new_labels


@pytest.mark.parametrize(
    "path",
    [
        # Randomly selected from glob + random.sample (seed = 1234, k = 10)
        "datasets/2d/mirflickr/im20425.png",
        "datasets/2d/random/granularity/1004709.png",
        "datasets/2d/random/granularity/0703705.png",
        "datasets/2d/random/granularity/0903203.png",
        "datasets/2d/random/granularity/0604302.png",
        "datasets/2d/mirflickr/im5004.png",
        "datasets/2d/random/granularity/0504804.png",
        "datasets/2d/mirflickr/im13926.png",
        "datasets/2d/3dpes/Set_1_ID_05_Camera_1_Seq_1_video0847.png",
        "datasets/2d/random/granularity/1404106.png",
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
    path: str,
    device: torch.device,
    contiguous: bool,  # noqa: FBT001
    connectivity: Literal[4, 8],
) -> None:
    if connectivity == 4:
        pytest.skip("4-connectivity is not yet supported.")

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = torch.from_numpy(image).to(device)

    if not contiguous:
        x = x.transpose(-2, -1).contiguous().transpose(-2, -1)

    labels = ccl2d(x, connectivity=connectivity)

    # Compare to what OpenCV would produce
    _, expected = cv2.connectedComponents(image, connectivity=connectivity)

    # Use NumPy for better explainability
    np.testing.assert_array_equal(
        consecutivize(labels.cpu().numpy()),
        expected,
        strict=True,
    )
