"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import cyclopts
import pandas as pd


def _version(v: str) -> tuple[int, ...]:
    return tuple(map(int, v.split(".")))


app = cyclopts.App(version_flags=None)


@app.default
def main(
    input_path: Annotated[
        Path,
        cyclopts.Parameter(
            help="Path to the versions file built with the versions.py script."
        ),
    ] = Path("versions.csv"),
    output_path: Annotated[
        str,
        cyclopts.Parameter(help="Destination path to save the compatibility matrix."),
    ] = "matrix.csv",
) -> None:
    """Select PyTorch versions to build for each available CUDA x Python pair.

    It basically selects the most recent version of PyTorch available for each pair.
    """
    versions = pd.read_csv(
        input_path,
        usecols=["torch", "cuda", "python"],
        dtype={"torch": "str", "cuda": "str", "python": "str"},
    )
    versions = versions.drop_duplicates()
    versions = versions.sort_values(
        ["cuda", "python", "torch"],
        key=lambda s: s.apply(_version),
    )
    versions = versions.drop_duplicates(["cuda", "python"], keep="last")

    versions.to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
