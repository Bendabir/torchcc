"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import re
from typing import Annotated

import bs4
import cyclopts
import httpx
import pandas as pd

_WHEEL_PATTERN = re.compile(
    # Identify the versions using CUDA so we can easily know CUDA, Python and Torch
    # versions (along with the platform).
    # From PEP 491 : https://peps.python.org/pep-0491/
    # {distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
    r"^torch-(\d+\.\d+\.\d+)\+(cu\d+)-([a-z0-9]+)-[a-z0-9]+-([a-z_0-9\.]+)\.whl$"
)
_CUDA_PATTERN = re.compile(r"^cu(\d+)(\d)$")
_PYTHON_PATTERN = re.compile(r"^cp(\d)(\d+)$")


def _fix_cuda_version(version: str) -> str:
    s = _CUDA_PATTERN.search(version)

    if not s:
        raise ValueError("Couldn't extract CUDA version.")

    return f"{s.group(1)}.{s.group(2)}"


def _fix_python_version(version: str) -> str:
    s = _PYTHON_PATTERN.search(version)

    if not s:
        raise ValueError("Couldn't extract Python version.")

    return f"{s.group(1)}.{s.group(2)}"


def _version(v: str) -> tuple[int, ...]:
    return tuple(map(int, v.split(".")))


def _get_versions(url: str) -> pd.DataFrame:
    r = httpx.get(url, follow_redirects=True)

    r.raise_for_status()

    soup = bs4.BeautifulSoup(r.content, features="html.parser")
    links = soup.find_all("a")
    wheels = [a.text.lower().strip() for a in links]
    versions: dict[str, list[str]] = {
        "torch": [],
        "cuda": [],
        "python": [],
        "platform": [],
    }

    for w in wheels:
        if s := _WHEEL_PATTERN.search(w):
            for i, k in enumerate(versions.keys(), start=1):
                versions[k].append(s.group(i))

    dataframe = pd.DataFrame(versions)
    dataframe["cuda"] = dataframe["cuda"].apply(_fix_cuda_version)
    dataframe["python"] = dataframe["python"].apply(_fix_python_version)

    return dataframe


app = cyclopts.App(version_flags=None)


@app.default
def main(
    *,
    url: Annotated[
        str,
        cyclopts.Parameter(help="URL of PyTorch wheel files."),
    ] = "https://download.pytorch.org/whl/torch",
    path: Annotated[
        str,
        cyclopts.Parameter(help="Destination of the matrix (as a CSV file)."),
    ] = "versions.csv",
    min_torch: Annotated[
        str,
        cyclopts.Parameter(
            help="Minimum PyTorch version to save. Older versions are trashed."
        ),
    ] = "1.7.1",
    min_cuda: Annotated[
        str,
        cyclopts.Parameter(
            help="Minimum CUDA version to save. Older versions are trashed."
        ),
    ] = "9.2",
    min_python: Annotated[
        str,
        cyclopts.Parameter(
            help="Minimum Python version to save. Older versions are trashed."
        ),
    ] = "3.9",
) -> None:
    """Extract the compatibility matrix for CUDA, Torch and Python."""
    versions = _get_versions(url)

    torch_versions = versions["torch"].apply(_version)
    cuda_versions = versions["cuda"].apply(_version)
    python_versions = versions["python"].apply(_version)

    selected = torch_versions >= _version(min_torch)
    selected &= cuda_versions >= _version(min_cuda)
    selected &= python_versions >= _version(min_python)

    versions[selected].sort_values(
        ["torch", "cuda", "python"],
        key=lambda s: s.apply(_version),
    ).to_csv(path, index=False)


if __name__ == "__main__":
    app()
