"""Copyright (c) 2024 Bendabir."""

# mypy: allow-untyped-calls
from __future__ import annotations

import glob
import os

import torch
import torch.version
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define some default architectures to build for (basically all)
# Couldn't find a better way to do this
TORCH_CUDA_ARCH_LIST = "TORCH_CUDA_ARCH_LIST"
DEBUG_MODE = "DEBUG_MODE"

# Restrict to the PyTorch stable ABI introduced in 2.10, whatever PyTorch version
# is used to build. Format : (major << 56) | (minor << 48)
TORCH_TARGET_VERSION = "-DTORCH_TARGET_VERSION=0x020a000000000000"

if TORCH_CUDA_ARCH_LIST not in os.environ:
    if torch.version.cuda is None:
        raise RuntimeError("Couldn't infer CUDA version.")

    version = tuple(map(int, torch.version.cuda.split(".")))

    # More details :
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
    # https://en.wikipedia.org/wiki/CUDA
    # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    # shortcut: assumes CUDA >= 12.6, add version checks to support older CUDA.
    # TODO : Add support for newer architectures
    archs = [
        "6.0",
        "6.1",
        "6.2",
        "7.0",
        "7.2",
        "7.5",
        "8.0",
        "8.6",
        "8.9",
        "9.0",
        "9.0a",
        "10.0",
    ]

    # CUDA 13 dropped Maxwell, Pascal and Volta
    if version >= (13, 0):
        archs = archs[archs.index("7.5") :]

    # For forward-compatibility
    # See : https://pytorch.org/docs/stable/cpp_extension.html
    archs[-1] = f"{archs[-1]}+PTX"

    os.environ[TORCH_CUDA_ARCH_LIST] = " ".join(archs)


def _nvcc_extra_compile_args(*, debug_mode: bool) -> list[str]:
    extra_compile_args = [
        "-D__STRICT_ANSI__",
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        TORCH_TARGET_VERSION,
        "-O0" if debug_mode else "-O3",
    ]

    if debug_mode:
        extra_compile_args.append("-g")

    return extra_compile_args


def _cxx_extra_compile_args(*, debug_mode: bool) -> list[str]:
    extra_compile_args = [
        TORCH_TARGET_VERSION,
        "-O0" if debug_mode else "-O3",
    ]

    if debug_mode:
        extra_compile_args.append("-g")

    return extra_compile_args


def _extra_link_args(*, debug_mode: bool) -> list[str]:
    if debug_mode:
        return ["-O0", "-g"]

    return []


def _include_dirs() -> list[str]:
    return [
        # NOTE : Need to provide the full path.
        #        Otherwise the compiler doesn't find our header files.
        #        Other libs (Python, Torch, CUDA, etc.)
        #        are automatically provided.
        os.path.abspath("include"),
    ]


debug_mode = os.getenv(DEBUG_MODE, "false").strip().lower() == "true"

# Project metadata lives in pyproject.toml
setup(
    ext_modules=[
        CUDAExtension(
            name="torchcc._C",
            sources=[
                *glob.glob("csrc/**/*.cpp", recursive=True),
                *glob.glob("csrc/**/*.cu", recursive=True),
            ],
            include_dirs=_include_dirs(),
            extra_compile_args={
                "cxx": _cxx_extra_compile_args(debug_mode=debug_mode),
                "nvcc": _nvcc_extra_compile_args(debug_mode=debug_mode),
            },
            extra_link_args=_extra_link_args(debug_mode=debug_mode),
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    zip_safe=False,
)
