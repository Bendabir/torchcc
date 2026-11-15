from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Literal

import cc3d
import cv2
import numpy as np
import numpy.typing as npt
import pytest
import torch

from torchcc import ccl3d

REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)


def read_volume(folder: str) -> npt.NDArray[np.uint8]:
    slices = []

    for path in sorted(Path(folder).glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        assert image is not None, f"Cannot read {path}."

        slices.append(image > 0)

    return np.stack(slices).astype(np.uint8)


def canonicalize(labels: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    # cc3d numbers components by first voxel, BUF by first 2x2x2 block.
    # Renumber both by first appearance in raster order (0 stays background).
    ids, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    foreground = ids != 0
    mapping = np.zeros(len(ids), dtype=labels.dtype)
    mapping[foreground] = np.argsort(np.argsort(first[foreground])) + 1

    return mapping[inverse].reshape(labels.shape)


def assert_labels(labels: torch.Tensor, batch: npt.NDArray[np.uint8]) -> None:
    expected = np.stack([
        canonicalize(cc3d.connected_components(vol, connectivity=26).astype(np.int32))
        for vol in batch
    ])

    assert labels.dtype == torch.int32
    np.testing.assert_array_equal(
        np.stack([canonicalize(lbl) for lbl in labels.cpu().numpy()]),
        expected,
        strict=True,
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 1, 1), id="1x1x1"),
        pytest.param((1, 1, 1, 37), id="single-column"),
        pytest.param((1, 1, 37, 1), id="single-row"),
        pytest.param((1, 37, 1, 1), id="single-plane-axis"),
        pytest.param((1, 1, 29, 37), id="single-plane"),
        pytest.param((1, 2, 2, 2), id="single-block"),
        pytest.param((1, 37, 2, 2), id="thin-row-column"),
        pytest.param((1, 2, 37, 2), id="thin-plane-column"),
        pytest.param((1, 2, 2, 37), id="thin-plane-row"),
        pytest.param((1, 33, 47, 35), id="odd-odd-odd"),
        pytest.param((1, 34, 46, 36), id="even-even-even"),
        pytest.param((1, 33, 46, 35), id="odd-even-odd"),
        pytest.param((1, 34, 47, 36), id="even-odd-even"),
        # A tile is 32x8 blocks of a plane (2x16x64 voxels), one voxel past it
        pytest.param((1, 3, 16, 64), id="tile-border-plane"),
        pytest.param((1, 2, 17, 64), id="tile-border-row"),
        pytest.param((1, 2, 16, 65), id="tile-border-column"),
        pytest.param((2, 3, 17, 65), id="tile-border-all"),
        pytest.param((5, 17, 19, 23), id="batch-odd"),
        pytest.param((3, 64, 64, 64), id="batch-even"),
    ],
)
@pytest.mark.parametrize("density", [0.1, 0.5, 0.9, 1.0])
@REQUIRES_CUDA
def test_ccl3d_random(
    device: torch.device,
    seed: int,
    shape: tuple[int, int, int, int],
    density: float,
) -> None:
    rng = np.random.default_rng(seed)
    batch = (rng.random(shape) < density).astype(np.uint8)

    labels = ccl3d(torch.from_numpy(batch).to(device))

    assert_labels(labels, batch)


@REQUIRES_CUDA
def test_ccl3d_serpentine(device: torch.device) -> None:
    # One voxel thick path that crosses many tiles in every axis.
    h, w, d = 11, 21, 70
    volume = np.zeros((h, w, d), dtype=np.uint8)
    volume[0::2, 0::2, :] = 1
    volume[0::2, 1::4, -1] = 1
    volume[0::2, 3::4, 0] = 1
    # Planes are linked at alternate corners
    volume[1::4, -1, -1] = 1
    volume[3::4, 0, 0] = 1
    batch = volume[np.newaxis]

    labels = ccl3d(torch.from_numpy(batch).to(device))

    assert_labels(labels, batch)


@REQUIRES_CUDA
def test_ccl3d_diagonals(device: torch.device) -> None:
    # Chains of voxels linked only by a corner or an edge, one per volume.
    # Each one needs one neighbour block : 0, 2, 6, 8 (plane before), 9, 11 (same plane)
    # Even size : a step crosses a block border on all its axes at once.
    size = 40
    steps = np.arange(size)
    back = size - 1 - steps
    middle = np.full(size, size // 2)
    chains = [
        (steps, steps, steps),
        (steps, steps, back),
        (steps, back, steps),
        (steps, back, back),
        (middle, steps, steps),
        (middle, steps, back),
    ]
    batch = np.zeros((len(chains), size, size, size), dtype=np.uint8)

    for i, chain in enumerate(chains):
        batch[i][chain] = 1

    labels = ccl3d(torch.from_numpy(batch).to(device))

    assert_labels(labels, batch)


@REQUIRES_CUDA
def test_ccl3d_neighbour_pairs(device: torch.device) -> None:
    # Two voxels at every position (every block parity and border) and every one
    # of the 26 offsets : each volume must hold exactly one component.
    size = 5
    offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)]
    volumes = []

    for start in itertools.product(range(size), repeat=3):
        for offset in offsets:
            end = tuple(s + o for s, o in zip(start, offset, strict=True))

            if all(0 <= e < size for e in end):
                volume = np.zeros((size, size, size), dtype=np.uint8)
                volume[start] = 1
                volume[end] = 1
                volumes.append(volume)

    labels = ccl3d(torch.from_numpy(np.stack(volumes)).to(device))
    foreground = labels > 0
    lowest = torch.where(foreground, labels, labels.amax()).amin(dim=(1, 2, 3))

    assert torch.equal(foreground.sum(dim=(1, 2, 3)), torch.full_like(lowest, 2))
    assert torch.equal(lowest, labels.amax(dim=(1, 2, 3)))


@pytest.mark.parametrize(
    "folder",
    [
        "datasets/3d/hilbert/n01",
        "datasets/3d/hilbert/n02",
    ],
)
@REQUIRES_CUDA
def test_ccl3d_datasets(device: torch.device, folder: str) -> None:
    batch = read_volume(folder)[np.newaxis]

    labels = ccl3d(torch.from_numpy(batch).to(device))

    assert_labels(labels, batch)


@REQUIRES_CUDA
def test_ccl3d_mitochondria(device: torch.device) -> None:
    # The full volume is 165x768x1024, a crop is enough.
    batch = read_volume("datasets/3d/mitochondria/test_gt")[
        np.newaxis,
        :49,
        300:557,
        400:913,
    ]

    labels = ccl3d(torch.from_numpy(np.ascontiguousarray(batch)).to(device))

    assert_labels(labels, batch)


@REQUIRES_CUDA
def test_ccl3d_large_batch(device: torch.device, seed: int) -> None:
    # More volumes than the grid can hold at once (65535), so the batch runs in chunks.
    size = 65_535 + 7
    checked = 16
    rng = np.random.default_rng(seed)
    batch = (rng.random((size, 3, 3, 5)) < 0.5).astype(np.uint8)

    labels = ccl3d(torch.from_numpy(batch).to(device))[-checked:]

    assert_labels(labels, batch[-checked:])


@pytest.mark.parametrize(
    "order",
    [(0, 1, 3, 2), (0, 3, 2, 1), (0, 2, 1, 3)],
    ids=["columns", "planes", "rows"],
)
@REQUIRES_CUDA
def test_ccl3d_non_contiguous(
    device: torch.device,
    seed: int,
    order: tuple[int, int, int, int],
) -> None:
    rng = np.random.default_rng(seed)
    data = (rng.random((3, 19, 23, 29)) < 0.5).astype(np.uint8)
    x = torch.from_numpy(data).to(device).permute(order)

    labels = ccl3d(x)

    assert_labels(labels, np.ascontiguousarray(data.transpose(order)))


@REQUIRES_CUDA
def test_ccl3d_channel_slice(device: torch.device, seed: int) -> None:
    # Typical model output [N, C, H, W, D]: the slice has a contiguous last dimension
    # but non-contiguous outer strides.
    rng = np.random.default_rng(seed)
    masks = (rng.random((4, 3, 19, 23, 29)) < 0.5).astype(np.uint8)
    channel = 1

    labels = ccl3d(torch.from_numpy(masks).to(device)[:, channel])

    assert_labels(labels, np.ascontiguousarray(masks[:, channel]))


@pytest.mark.parametrize(
    ("planes", "rows", "cols"),
    [
        pytest.param(slice(None), slice(None), slice(3, 30), id="columns"),
        pytest.param(slice(None), slice(5, 20), slice(None), id="rows"),
        pytest.param(slice(2, 17), slice(None), slice(None), id="planes"),
        pytest.param(slice(2, 17), slice(5, 20), slice(3, 30), id="all"),
    ],
)
@REQUIRES_CUDA
def test_ccl3d_crop(
    device: torch.device,
    seed: int,
    planes: slice,
    rows: slice,
    cols: slice,
) -> None:
    # A crop keeps the last dimension contiguous, but the other strides change.
    rng = np.random.default_rng(seed)
    masks = (rng.random((3, 19, 23, 37)) < 0.5).astype(np.uint8)

    labels = ccl3d(torch.from_numpy(masks).to(device)[:, planes, rows, cols])

    assert_labels(labels, np.ascontiguousarray(masks[:, planes, rows, cols]))


@pytest.mark.parametrize(
    "shape",
    [(0, 8, 8, 8), (2, 0, 5, 5), (2, 5, 0, 5), (2, 5, 5, 0)],
    ids=["no-volume", "no-plane", "no-row", "no-column"],
)
@REQUIRES_CUDA
def test_ccl3d_empty(device: torch.device, shape: tuple[int, int, int, int]) -> None:
    labels = ccl3d(torch.zeros(shape, device=device, dtype=torch.uint8))

    assert labels.shape == shape
    assert labels.dtype == torch.int32


@REQUIRES_CUDA
def test_ccl3d_labels_per_volume(device: torch.device, seed: int) -> None:
    # Labels restart in each volume : the same volume gives the same labels.
    rng = np.random.default_rng(seed)
    volume = torch.from_numpy((rng.random((1, 17, 19, 23)) < 0.5).astype(np.uint8))

    # expand on the GPU : the batch stride is 0, no copy
    labels = ccl3d(volume.to(device).expand(3, -1, -1, -1))

    assert torch.equal(labels[0], labels[1])
    assert torch.equal(labels[0], labels[2])


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
@REQUIRES_CUDA
def test_ccl3d_connectivity(connectivity: int, device: torch.device) -> None:
    volumes = torch.zeros((4, 16, 16, 16), device=device, dtype=torch.uint8)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Only 6-connectivity, 18-connectivity and 26-connectivity are supported."
        ),
    ):
        _ = ccl3d(volumes, connectivity=connectivity)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("connectivity", [6, 18])
@REQUIRES_CUDA
def test_ccl3d_not_implemented(
    connectivity: Literal[6, 18],
    device: torch.device,
) -> None:
    volumes = torch.zeros((4, 16, 16, 16), device=device, dtype=torch.uint8)

    with pytest.raises(
        NotImplementedError,
        match=re.escape(f"3D CCL {connectivity}-connectivity is not yet supported."),
    ):
        _ = ccl3d(volumes, connectivity=connectivity)


@pytest.mark.parametrize(
    "shape",
    [(1,), (32, 32), (4, 32, 32), (2, 4, 8, 32, 32)],
    ids=["dim-1", "dim-2", "dim-3", "dim-5"],
)
@REQUIRES_CUDA
def test_ccl3d_input_shape(device: torch.device, shape: tuple[int, ...]) -> None:
    volumes = torch.zeros(shape, device=device, dtype=torch.uint8)

    with pytest.raises(
        ValueError,
        match=re.escape("Input must be a batch of volumes [N, H, W, D]."),
    ):
        ccl3d(volumes)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint16,
        torch.int8,
        torch.int32,
        torch.int64,
        torch.float32,
        torch.bool,
    ],
    ids=["uint16", "int8", "int32", "int64", "float32", "bool"],
)
@REQUIRES_CUDA
def test_ccl3d_input_type(device: torch.device, dtype: torch.dtype) -> None:
    volumes = torch.zeros((2, 8, 8, 8), device=device, dtype=dtype)

    with pytest.raises(TypeError, match=re.escape("Input must be uint8 data.")):
        ccl3d(volumes)


@REQUIRES_CUDA
def test_ccl3d_max_size(device: torch.device) -> None:
    # One volume with more than INT32_MAX voxels (1291^3).
    # Expanding a single voxel needs no memory, the check runs before any copy.
    size = 1291
    volumes = torch.zeros((1, 1, 1, 1), device=device, dtype=torch.uint8)

    with pytest.raises(
        ValueError,
        match=re.escape("Volumes are too big and will cause overflow on labels."),
    ):
        ccl3d(volumes.expand(1, size, size, size))
