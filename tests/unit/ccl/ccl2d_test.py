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
def test_ccl2d(
    path: str,
    device: torch.device,
    contiguous: bool,  # noqa: FBT001
    connectivity: Literal[4, 8],
) -> None:
    if connectivity == 4:
        pytest.skip("4-connectivity is not yet supported.")

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = torch.from_numpy(image).unsqueeze_(0).to(device)

    if not contiguous:
        x = x.transpose(-2, -1).contiguous().transpose(-2, -1)

    labels = ccl2d(x, connectivity=connectivity)

    # Compare to what OpenCV would produce
    _, expected = cv2.connectedComponents(image, connectivity=connectivity)

    # Use NumPy for better explainability
    np.testing.assert_array_equal(
        consecutivize(labels.cpu().numpy()),
        np.expand_dims(expected, 0),
        strict=True,
    )


@pytest.mark.parametrize(
    "paths",
    [
        # Randomly selected from glob + random.sample (seed = 123456789, k = 30)
        ["datasets/2d/mirflickr/im8735.png"],
        [
            "datasets/2d/mirflickr/im23066.png",
            "datasets/2d/3dpes/Set_1_ID_20_Camera_3_seq_1_video0069.png",
        ],
        [
            "datasets/2d/tobacco800/vss86d00.png",
            "datasets/2d/mirflickr/im22114.png",
            "datasets/2d/3dpes/Set_1_ID_15_Camera_1_Seq_1_video0021.png",
        ],
        [
            "datasets/2d/mirflickr/im19356.png",
            "datasets/2d/mirflickr/im615.png",
            "datasets/2d/mirflickr/im10505.png",
            "datasets/2d/mirflickr/im23635.png",
        ],
        [
            "datasets/2d/random/granularity/1506501.png",
            "datasets/2d/3dpes/Set_1_ID_09_Camera_1_Seq_2_video0359.png",
            "datasets/2d/tobacco800/pqw02f00-page07_4.png",
            "datasets/2d/mirflickr/im16702.png",
            "datasets/2d/random/granularity/1605902.png",
        ],
        [
            "datasets/2d/random/granularity/0800605.png",
            "datasets/2d/random/granularity/1306304.png",
            "datasets/2d/mirflickr/im21636.png",
            "datasets/2d/random/granularity/1603506.png",
            "datasets/2d/random/granularity/1600304.png",
            "datasets/2d/mirflickr/im9456.png",
        ],
        [
            "datasets/2d/mirflickr/im23262.png",
            "datasets/2d/3dpes/Set_1_ID_06_Camera_1_Seq_3_video0810.png",
            "datasets/2d/random/granularity/0202409.png",
            "datasets/2d/3dpes/Set_1_ID_03_Camera_1_Seq_3_video0101.png",
            "datasets/2d/3dpes/Set_1_ID_05_Camera_3_Seq_3_video0137.png",
            "datasets/2d/random/granularity/0405207.png",
            "datasets/2d/3dpes/Set_1_ID_20_Camera_3_seq_2_video0147.png",
            "datasets/2d/3dpes/Set_1_ID_17_Camera_1_Seq_1_video0462.png",
            "datasets/2d/mirflickr/im8725.png",
        ],
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
def test_ccl2d_batch(
    paths: list[str],
    device: torch.device,
    contiguous: bool,  # noqa: FBT001
    connectivity: Literal[4, 8],
) -> None:
    if connectivity == 4:
        pytest.skip("4-connectivity is not yet supported.")

    # Use some padding so batches have the same size.
    # Not really an issue for CCL.
    images: list[npt.NDArray[np.uint8]] = [
        cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths
    ]
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    images = [
        np.pad(img, ((0, h - img.shape[0]), (0, w - img.shape[1]))) for img in images
    ]
    batch = np.stack(images)

    x = torch.from_numpy(batch).to(device)

    if not contiguous:
        x = x.transpose(-2, -1).contiguous().transpose(-2, -1)

    labels = ccl2d(x, connectivity=connectivity)
    expected = np.zeros_like(batch, dtype=np.int32)

    # Compare to what OpenCV would produce
    for i in range(len(batch)):
        _, expected[i] = cv2.connectedComponents(batch[i], connectivity=connectivity)

    # Use NumPy for better explainability
    np.testing.assert_array_equal(
        np.stack([consecutivize(lbl) for lbl in labels.cpu().numpy()]),
        expected,
        strict=True,
    )
