"""Copyright (c) 2024 Bendabir."""

# mypy: allow-untyped-calls
from __future__ import annotations

import glob
import os
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
                        *glob.glob("csrc/**/*.cpp", recursive=True),
                        *glob.glob("csrc/**/*.cu", recursive=True),
                    ],
                    include_dirs=[
                        # NOTE : Need to provide the full path.
                        #        Otherwise the compiler doesn't find our header files.
                        #        Other libs (Python, Torch, CUDA, etc.)
                        #        are automatically provided.
                        os.path.abspath("include"),
                    ],
                    extra_compile_args={
                        "nvcc": [
                            "-D__STRICT_ANSI__",
                            "-DCUDA_HAS_FP16=1",
                            "-D__CUDA_NO_HALF_OPERATORS__",
                            "-D__CUDA_NO_HALF_CONVERSIONS__",
                            "-D__CUDA_NO_HALF2_OPERATORS__",
                        ],
                    },
                )
            ],
            "cmdclass": {
                "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
            },
            "zip_safe": False,
        }
    )
