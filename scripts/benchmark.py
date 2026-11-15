"""Copyright (c) 2026 Bendabir.

Benchmark CPU vs GPU Connected Components Labeling on the YACCLAB datasets.

Run `just data` first, then : `uv run --no-sync python scripts/benchmark.py` (2D)
or `uv run --no-sync python scripts/benchmark.py ccl3d` (3D).
"""

from __future__ import annotations

import logging
import math
import statistics
import time
import warnings
from collections import Counter

# Cyclopts resolves the type hints at runtime.
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import cv2
import cyclopts
import numpy as np
import torch
from cyclopts.types import ExistingDirectory
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from rich.table import Table

import torchcc

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Final


LOGGER = logging.getLogger("benchmark")

MS: Final = 1e3
# Min warmup time per measure, so the GPU clocks ramp up after an idle phase.
WARMUP_SECONDS: Final = 0.25
# Min timed duration per measure, so short calls get enough samples for a stable median.
MEASURE_SECONDS: Final = 0.5
# Max images read per dataset to find the most common shape (datasets are huge).
IMAGE_SCAN_LIMIT: Final = 512
# Check images are tiny edge cases, not representative.
SKIPPED_IMAGE_DATASETS: Final = {"check"}
# Max volumes read per 3D dataset to find the most common shape (a volume is many PNGs).
VOLUME_SCAN_LIMIT: Final = 16


def read_image(path: Path) -> np.ndarray:
    """Read a PNG as a binary uint8 image.

    Parameters
    ----------
    path : Path
        Path to the PNG file.

    Returns
    -------
    np.ndarray
        Binary image [H, W], with 1 for the foreground.

    Raises
    ------
    ValueError
        If the image cannot be read.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image : {path}")

    return (image > 0).astype(np.uint8)


def load_batch(dataset: Path, count: int) -> torch.Tensor:
    """Load images sharing the most common shape, as a batch.

    Parameters
    ----------
    dataset : Path
        Dataset directory.
    count : int
        Batch size. Images are recycled when there are not enough of them.

    Returns
    -------
    torch.Tensor
        Batch of images [N, H, W].
    """
    images = [read_image(p) for p in sorted(dataset.rglob("*.png"))[:IMAGE_SCAN_LIMIT]]
    shape = Counter(image.shape for image in images).most_common(1)[0][0]
    images = [image for image in images if image.shape == shape]

    # Timings do not need unique data.
    return torch.from_numpy(np.stack([images[i % len(images)] for i in range(count)]))


def measure(fn: Callable[[], object], *, warmup: int, repeats: int) -> float:
    """Measure the median time of a call.

    Parameters
    ----------
    fn : Callable[[], object]
        Function to time.
    warmup : int
        Min number of calls before timing (it lasts WARMUP_SECONDS at least).
    repeats : int
        Min number of timed calls (they last MEASURE_SECONDS at least).

    Returns
    -------
    float
        Median time in milliseconds.
    """
    # Warm up for a minimum time too : after a few seconds idle (e.g. the CPU rows),
    # the GPU clocks need time to ramp up. A count alone gave rows up to 5x slower.
    calls = 0
    end = time.perf_counter() + WARMUP_SECONDS

    while (calls < warmup) or (time.perf_counter() < end):
        fn()
        torch.cuda.synchronize()
        calls += 1

    # Same for the timed calls : 10 calls of 2 ms are too short to be stable,
    # one slow burst moved the median by up to 2.5x.
    times = []
    end = time.perf_counter() + MEASURE_SECONDS

    while (len(times) < repeats) or (time.perf_counter() < end):
        torch.cuda.synchronize()

        start = time.perf_counter()

        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return statistics.median(times) * MS


def read_volume(folder: Path) -> np.ndarray:
    """Read a folder of PNG slices as a binary uint8 volume.

    Parameters
    ----------
    folder : Path
        Folder holding one PNG per slice.

    Returns
    -------
    np.ndarray
        Binary volume [slices, H, W], with 1 for the foreground.
    """
    return np.stack([read_image(p) for p in sorted(folder.glob("*.png"))])


def find_volume_datasets(root: Path) -> list[Path]:
    """Find the 3D datasets, i.e. the folders whose subfolders are volumes.

    Parameters
    ----------
    root : Path
        Directory to search (e.g. `datasets/3d`).

    Returns
    -------
    list[Path]
        Dataset directories (e.g. `hilbert` or `random3D/granularity`).
    """
    datasets = []

    for folder in sorted(d for d in root.iterdir() if d.is_dir()):
        subfolders = [d for d in folder.iterdir() if d.is_dir()]

        if not subfolders:
            continue

        # Volumes are folders of PNG slices, otherwise go one level down
        if any(subfolders[0].glob("*.png")):
            datasets.append(folder)
        else:
            datasets.extend(find_volume_datasets(folder))

    return datasets


def load_volumes(dataset: Path, count: int) -> torch.Tensor:
    """Load volumes sharing the largest shape, as a batch.

    Some datasets mix sizes in equal numbers (e.g. 8^3 to 256^3), so the most
    common shape is not meaningful. The largest one is the useful case to time.

    Parameters
    ----------
    dataset : Path
        Dataset directory, with one subfolder of PNG slices per volume.
    count : int
        Batch size. Volumes are recycled when there are not enough of them.

    Returns
    -------
    torch.Tensor
        Batch of volumes [N, slices, H, W].
    """
    folders = sorted(d for d in dataset.iterdir() if d.is_dir())
    # Sample across the whole dataset : folders are often sorted by density.
    step = max(1, len(folders) // VOLUME_SCAN_LIMIT)
    volumes = [read_volume(f) for f in folders[::step][:VOLUME_SCAN_LIMIT]]
    shape = max((volume.shape for volume in volumes), key=math.prod)
    volumes = [volume for volume in volumes if volume.shape == shape]

    # Timings do not need unique data.
    return torch.from_numpy(np.stack([volumes[i % len(volumes)] for i in range(count)]))


def run(
    title: str,
    datasets: Sequence[tuple[str, Callable[[], torch.Tensor]]],
    label: Callable[[torch.Tensor], torch.Tensor],
    *,
    batch_sizes: Sequence[int],
    warmup: int,
    repeats: int,
    device: str,
) -> None:
    """Time CPU, GPU and GPU end-to-end labeling on each dataset, then print a table.

    Parameters
    ----------
    title : str
        Table title. The GPU name is appended.
    datasets : Sequence[tuple[str, Callable[[], torch.Tensor]]]
        Dataset names, with a function that loads the largest batch.
    label : Callable[[torch.Tensor], torch.Tensor]
        Labeling function to time.
    batch_sizes : Sequence[int]
        Batch sizes to benchmark.
    warmup : int
        Min number of calls before timing.
    repeats : int
        Min number of timed calls.
    device : str
        GPU to use.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler()],
    )
    # The CPU fallback warns on each call.
    warnings.filterwarnings("ignore", message="Falling back to")

    gpu = torch.device(device)
    gpu_name = torch.cuda.get_device_name(gpu)
    opts = {"warmup": warmup, "repeats": repeats}
    table = Table(title=f"{title}, {gpu_name} (median, ms)")

    LOGGER.info("GPU : %s", gpu_name)

    for column in ("dataset", "batch", "shape", "CPU", "GPU", "GPU e2e", "speedup"):
        table.add_column(column)

    with Progress() as progress:
        task = progress.add_task("Benchmark", total=len(datasets) * len(batch_sizes))

        for name, load in datasets:
            LOGGER.info("Loading %s", name)

            batch = load()
            shape = "x".join(map(str, batch.shape[1:]))

            LOGGER.info("%s : %d samples of shape %s", name, len(batch), shape)

            for size in batch_sizes:
                x = batch[:size]
                x_gpu = x.to(gpu)
                cpu = measure(lambda x=x: label(x), **opts)
                on_gpu = measure(lambda x=x_gpu: label(x), **opts)
                e2e = measure(lambda x=x: label(x.to(gpu)).cpu(), **opts)
                progress.advance(task)
                table.add_row(
                    name,
                    str(size),
                    shape,
                    f"{cpu:.3f}",
                    f"{on_gpu:.3f}",
                    f"{e2e:.3f}",
                    f"{cpu / on_gpu:.1f}x",
                )

    Console().print(table)


APP: Final = cyclopts.App()


@APP.default
def ccl2d(
    *,
    datasets_dir: ExistingDirectory = Path("datasets"),
    batch_sizes: Annotated[
        Sequence[int],
        cyclopts.Parameter(consume_multiple=True, negative=""),
    ] = (1, 8, 32, 64),
    warmup: int = 3,
    repeats: int = 10,
    device: str = "cuda:0",
    connectivity: Literal[4, 8] = 8,
) -> None:
    """Benchmark CPU vs GPU 2D Connected Components Labeling.

    Run `just data` first.

    Parameters
    ----------
    datasets_dir : Path
        Directory holding the `2d` datasets.
    batch_sizes : Sequence[int]
        Batch sizes to benchmark.
    warmup : int
        Min number of calls before timing.
    repeats : int
        Min number of timed calls.
    device : str
        GPU to use.
    connectivity : Literal[4, 8]
        Connectivity for the connected components labeling.
    """
    count = max(batch_sizes)
    datasets = [
        (d.name, lambda d=d: load_batch(d, count))
        for d in sorted((datasets_dir / "2d").iterdir())
        if d.is_dir() and d.name not in SKIPPED_IMAGE_DATASETS
    ]

    run(
        f"ccl2d, {connectivity}-connectivity",
        datasets,
        lambda x: torchcc.ccl2d(x, connectivity=connectivity),
        batch_sizes=batch_sizes,
        warmup=warmup,
        repeats=repeats,
        device=device,
    )


@APP.command
def ccl3d(
    *,
    datasets_dir: ExistingDirectory = Path("datasets"),
    batch_sizes: Annotated[
        Sequence[int],
        cyclopts.Parameter(consume_multiple=True, negative=""),
    ] = (1, 4, 8),
    warmup: int = 3,
    repeats: int = 10,
    device: str = "cuda:0",
    connectivity: Literal[26] = 26,
) -> None:
    """Benchmark CPU vs GPU 3D Connected Components Labeling.

    Run `just data` first. CPU runs ConnectedComponents3D.

    Parameters
    ----------
    datasets_dir : Path
        Directory holding the `3d` datasets.
    batch_sizes : Sequence[int]
        Batch sizes to benchmark. Large volumes need a lot of GPU memory
        (about 5 B per voxel).
    warmup : int
        Min number of calls before timing.
    repeats : int
        Min number of timed calls.
    device : str
        GPU to use.
    connectivity : Literal[26]
        Connectivity for the connected components labeling.
    """
    root = datasets_dir / "3d"
    count = max(batch_sizes)
    datasets = [
        (str(d.relative_to(root)), lambda d=d: load_volumes(d, count))
        for d in find_volume_datasets(root)
    ]

    run(
        f"ccl3d, {connectivity}-connectivity",
        datasets,
        lambda x: torchcc.ccl3d(x, connectivity=connectivity),
        batch_sizes=batch_sizes,
        warmup=warmup,
        repeats=repeats,
        device=device,
    )


if __name__ == "__main__":
    APP()
