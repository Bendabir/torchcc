"""Copyright (c) 2024 Bendabir."""

# mypy: allow-untyped-calls
from __future__ import annotations

import itertools as it
import random
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict, TypeVar, final

import cv2
import cyclopts
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from tqdm import tqdm

from torchcc.ccl import ccl2d

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

app = cyclopts.App(version_flags=None)
app_2d = cyclopts.App(name="2d", version_flags=None)
app_3d = cyclopts.App(name="3d", version_flags=None)

app.command(app_2d)
app.command(app_3d)


def _get_devices() -> list[str]:
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))

    return devices


@final
class _Results(TypedDict):
    path: list[Path]
    density: list[float]
    size: list[int]
    time: list[float]
    device: list[str]
    batch_size: list[int]
    connectivity: list[int]


T = TypeVar("T")


def _batchify(iterable: Iterable[T], *, n: int = 1) -> Iterator[list[T]]:
    iterator = iter(iterable)

    while batch := list(it.islice(iterator, n)):
        yield batch


@app_2d.default
def _2d(  # noqa: PLR0914
    *,
    connectivity: Literal[4, 8] = 8,
    device: Annotated[
        Literal["cpu", "cuda:0"],
        cyclopts.Parameter(help="Device to run the benchmark on."),
    ] = "cpu",
    seed: int = 1234,
    input_: Annotated[Path, cyclopts.Parameter(name="input")] = Path("datasets/2d"),
    output: Path = Path("benchmarks/2d.csv"),
    k: int = 10_000,
    batch_size: int = 16,
) -> None:
    """Run a 2D benchmark to assess performances."""
    output.parent.mkdir(parents=True, exist_ok=True)

    if device == "cpu":
        warnings.warn(
            "Running on CPU. Using batches of 1 as there is no parallelism.",
            UserWarning,
            stacklevel=2,
        )

        batch_size = 1

    generator = random.Random(seed)  # noqa: S311
    paths = list(input_.rglob("**/*.png"))
    paths = generator.sample(paths, k=k)

    results: _Results = {
        "density": [],
        "path": [],
        "size": [],
        "time": [],
        "device": [],
        "batch_size": [],
        "connectivity": [],
    }

    with tqdm(unit="image") as bar:
        for ps in _batchify(paths, n=batch_size):
            # Use some padding so batches have the same size.
            # Not really an issue for CCL.
            images: list[npt.NDArray[np.uint8]] = [
                cv2.imread(p.as_posix(), cv2.IMREAD_GRAYSCALE) for p in ps
            ]
            sizes = [np.size(img) for img in images]
            densities = [
                np.sum(images[i] > 0).item() / sizes[i] for i in range(len(images))
            ]
            h = max(img.shape[0] for img in images)
            w = max(img.shape[1] for img in images)
            images = [
                np.pad(img, ((0, h - img.shape[0]), (0, w - img.shape[1])))
                for img in images
            ]
            batch = np.stack(images)

            if device == "cpu":
                cpu_start = time.perf_counter()
                _ = ccl2d(torch.from_numpy(batch), connectivity=connectivity)
                cpu_end = time.perf_counter()
                duration = cpu_end - cpu_start
            else:
                gpu_batch = torch.from_numpy(batch).to(device)
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)

                gpu_start.record()

                _ = ccl2d(gpu_batch, connectivity=connectivity)

                gpu_end.record()
                torch.cuda.synchronize()

                duration = 1_000 * gpu_start.elapsed_time(gpu_end)

            length = len(ps)

            bar.update(length)

            results["path"].extend(ps)
            results["density"].extend(densities)
            results["size"].extend(sizes)
            results["time"].extend([duration / length] * length)
            results["batch_size"].extend([batch_size] * length)
            results["device"].extend([device] * length)
            results["connectivity"].extend([connectivity] * length)

    df = pd.DataFrame(results)

    df.to_csv(output, index=False)


@app_3d.default
def _3d(connectivity: Literal[6, 18, 26]) -> None:
    """Run a 3D benchmark to assess performances."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
