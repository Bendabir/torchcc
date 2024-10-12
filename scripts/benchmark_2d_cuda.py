"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import itertools as it
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, TypeVar, final

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from tqdm import tqdm

from torchcc import ccl2d

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

T = TypeVar("T")


def _batchify(iterable: Iterable[T], *, n: int = 1) -> Iterator[list[T]]:
    iterator = iter(iterable)

    while batch := list(it.islice(iterator, n)):
        yield batch


@final
class _Results(TypedDict):
    path: list[Path]
    density: list[float]
    size: list[int]
    time: list[float]
    device: list[str]
    batch_size: list[int]
    connectivity: list[int]
    implementation: list[str]


SEED = 1234
INPUT = Path("datasets/2d")
BS = 8
OUTPUT = Path(f"benchmarks/2d-cuda-{BS}.csv")
K = 50_000
DEVICE = "cuda"

generator = random.Random(SEED)  # noqa: S311
paths = list(INPUT.rglob("**/*.png"))
paths = generator.sample(paths, k=K)

results: _Results = {
    "density": [],
    "path": [],
    "size": [],
    "time": [],
    "device": [],
    "batch_size": [],
    "connectivity": [],
    "implementation": [],
}

with tqdm(unit="image", total=len(paths)) as bar:
    for ps in _batchify(paths, n=BS):
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
        tensor = torch.from_numpy(batch).to(DEVICE)
        start = time.perf_counter()

        with torch.no_grad():
            _ = ccl2d(tensor, connectivity=8)
            torch.cuda.synchronize()

        end = time.perf_counter()
        duration = end - start
        length = len(ps)

        results["path"].extend(ps)
        results["density"].extend(densities)
        results["size"].extend(sizes)
        results["time"].extend([duration / length] * length)
        results["batch_size"].extend([BS] * length)
        results["device"].extend([DEVICE] * length)
        results["connectivity"].extend([8] * length)
        results["implementation"].extend(["torchcc"] * length)

        bar.update(length)

pd.DataFrame(results).to_csv(OUTPUT, index=False)
