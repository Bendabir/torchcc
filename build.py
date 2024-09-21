"""Copyright (c) 2024 Bendabir."""

# mypy: allow-untyped-calls
from __future__ import annotations

import glob
from typing import Any

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def build(setup_kwargs: dict[str, Any]) -> None:
    """Add specifications to build the CUDA extension."""
    setup_kwargs.update(
        {
            "ext_modules": [
                CUDAExtension(
                    # NOTE : Couldn't find a way to build the lib to torchcc._C
                    #        (or something else) and then import it properly,
                    #        so the lib file is a top level.
                    name="_libtorchcc",
                    sources=[
                        *glob.glob("csrc/*.cpp"),
                        *glob.glob("csrc/*.cu"),
                    ],
                )
            ],
            "cmdclass": {
                "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
            },
            "zip_safe": False,
        }
    )
