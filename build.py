"""Copyright (c) 2024 Bendabir."""

# mypy: allow-untyped-calls
from __future__ import annotations

import glob
import os
from typing import Any

import torch
import torch.version
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define some default architectures to build for (basically all)
# Couldn't find a better way to do this
TORCH_CUDA_ARCH_LIST = "TORCH_CUDA_ARCH_LIST"

if TORCH_CUDA_ARCH_LIST not in os.environ:
    archs: list[str] = []

    if torch.version.cuda is None:
        raise RuntimeError("Couldn't infer CUDA version.")

    version = tuple(map(int, torch.version.cuda.split(".")))

    # More details :
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
    # https://en.wikipedia.org/wiki/CUDA
    # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    if version <= (10, 2):
        archs.extend(("3.0", "3.5", "3.7"))

    if version <= (11, 8):
        archs.extend(("5.0", "5.2", "5.3"))

    if version >= (8, 0):
        archs.extend(("6.0", "6.1", "6.2"))

    if version >= (9, 0):
        archs.extend(("7.0", "7.2"))

    if version >= (10, 0):
        archs.append("7.5")

    if version >= (11, 1):
        archs.extend(("8.0", "8.6", "8.7"))

    if version >= (11, 8):
        archs.append("8.9")

    if version >= (12, 0):
        archs.extend(("9.0", "9.0a"))

    if version >= (12, 6):
        archs.append("10.0")

    os.environ[TORCH_CUDA_ARCH_LIST] = " ".join(archs)


def build(setup_kwargs: dict[str, Any]) -> None:
    """Add specifications to build the CUDA extension."""
    setup_kwargs.update(
        {
            "ext_modules": [
                CUDAExtension(
                    name="torchcc._cuda",
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
