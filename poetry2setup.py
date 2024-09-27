"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

from pathlib import Path

from poetry.core.factory import Factory
from poetry.core.masonry.builders.sdist import SdistBuilder

if __name__ == "__main__":
    factory = Factory()
    builder = SdistBuilder(factory.create_poetry(Path.cwd()))
    content = builder.build_setup()

    with Path("setup.py").open("wb") as file:
        file.write(content)
