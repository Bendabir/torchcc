"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import TypedDict, final

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torchcc import ccl2d


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
OUTPUT = Path("benchmarks/2d-cpu-1.csv")
K = 50_000

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

for p in tqdm(paths, unit="image"):
    img = cv2.imread(p.as_posix(), cv2.IMREAD_GRAYSCALE)
    tensor = torch.from_numpy(img).unsqueeze_(0)
    start = time.perf_counter()
    _ = ccl2d(tensor, connectivity=8)
    end = time.perf_counter()
    duration = end - start

    size = np.size(img)
    density = np.sum(img > 0).item() / size

    results["path"].append(p)
    results["density"].append(density)
    results["size"].append(size)
    results["time"].append(duration)
    results["batch_size"].append(1)
    results["device"].append("cpu")
    results["connectivity"].append(8)
    results["implementation"].append("opencv")

pd.DataFrame(results).to_csv(OUTPUT, index=False)
