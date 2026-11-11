"""Copyright (c) 2026 Bendabir.

Benchmark CPU vs GPU 2D Connected Components Labeling (4-connectivity is not yet
supported on GPU, so 8-connectivity is used) on the YACCLAB datasets.

Run `just data` first, then : `uv run --no-sync python scripts/benchmark.py`.
"""

from __future__ import annotations

import logging
import statistics
import time
import warnings
from collections import Counter

# Cyclopts resolves the type hints at runtime.
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

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
CONNECTIVITY: Final = 8
# Max images read per dataset to find the most common shape (datasets are huge).
SCAN_LIMIT: Final = 512
# Check images are tiny edge cases, not representative.
SKIPPED_DATASETS: Final = {"check"}


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
    images = [read_image(p) for p in sorted(dataset.rglob("*.png"))[:SCAN_LIMIT]]
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
        Number of calls before timing.
    repeats : int
        Number of timed calls.

    Returns
    -------
    float
        Median time in milliseconds.
    """
    for _ in range(warmup):
        fn()

    times = []

    for _ in range(repeats):
        torch.cuda.synchronize()

        start = time.perf_counter()

        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return statistics.median(times) * MS


def main(
    *,
    datasets_dir: ExistingDirectory = Path("datasets"),
    batch_sizes: Annotated[
        Sequence[int],
        cyclopts.Parameter(consume_multiple=True, negative=""),
    ] = (1, 8, 32, 64),
    warmup: int = 3,
    repeats: int = 10,
    device: str = "cuda:0",
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
        Number of calls before timing.
    repeats : int
        Number of timed calls.
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
    table = Table(title=f"ccl2d, {CONNECTIVITY}-connectivity, {gpu_name} (median, ms)")

    LOGGER.info("GPU : %s", gpu_name)

    for column in ("dataset", "batch", "shape", "CPU", "GPU", "GPU e2e", "speedup"):
        table.add_column(column)

    datasets = [
        d
        for d in sorted((datasets_dir / "2d").iterdir())
        if d.is_dir() and d.name not in SKIPPED_DATASETS
    ]

    with Progress() as progress:
        task = progress.add_task("Benchmark", total=len(datasets) * len(batch_sizes))

        for dataset in datasets:
            LOGGER.info("Loading %s", dataset.name)

            batch = load_batch(dataset, max(batch_sizes))
            shape = "x".join(map(str, batch.shape[1:]))

            LOGGER.info("%s : %d samples of shape %s", dataset.name, len(batch), shape)

            for size in batch_sizes:
                x = batch[:size]
                x_gpu = x.to(gpu)
                cpu = measure(
                    lambda x=x: torchcc.ccl2d(x, connectivity=CONNECTIVITY),
                    **opts,
                )
                on_gpu = measure(
                    lambda x=x_gpu: torchcc.ccl2d(x, connectivity=CONNECTIVITY),
                    **opts,
                )
                e2e = measure(
                    lambda x=x: torchcc.ccl2d(
                        x.to(gpu),
                        connectivity=CONNECTIVITY,
                    ).cpu(),
                    **opts,
                )
                progress.advance(task)
                table.add_row(
                    dataset.name,
                    str(size),
                    shape,
                    f"{cpu:.3f}",
                    f"{on_gpu:.3f}",
                    f"{e2e:.3f}",
                    f"{cpu / on_gpu:.1f}x",
                )

    Console().print(table)


if __name__ == "__main__":
    cyclopts.run(main)
