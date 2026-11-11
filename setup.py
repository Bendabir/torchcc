"""Copyright (c) 2024 Bendabir."""

# mypy: allow-untyped-calls
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch.version
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if TYPE_CHECKING:
    from typing import Final

# Define some default architectures to build for (basically all)
# Couldn't find a better way to do this
TORCH_CUDA_ARCH_LIST: Final = "TORCH_CUDA_ARCH_LIST"
DEBUG_MODE: Final = "DEBUG_MODE"

# Restrict to the PyTorch stable ABI introduced in 2.10, whatever PyTorch version
# is used to build. Format : (major << 56) | (minor << 48)
TORCH_TARGET_MAJOR: Final = 2
TORCH_TARGET_MINOR: Final = 10
TORCH_TARGET_VERSION: Final = hex(
    (TORCH_TARGET_MAJOR << 56) | (TORCH_TARGET_MINOR << 48)
)

# Compute capabilities to build for, per CUDA toolkit (major, minor).
# The last entry carries PTX for forward-compatibility (never on "9.0a",
# arch-specific PTX isn't forward-compatible).
# See : https://pytorch.org/docs/stable/cpp_extension.html
# More details :
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
# CUDA 12.8 added Blackwell (10.0, 12.0).
# CUDA 13.0 dropped Maxwell, Pascal and Volta.
CUDA_ARCHS: Final[dict[tuple[int, int], list[str]]] = {
    (12, 6): [
        "5.0",
        "5.2",
        "5.3",
        "6.0",
        "6.1",
        "6.2",
        "7.0",
        "7.2",
        "7.5",
        "8.0",
        "8.6",
        "8.7",
        "8.9",
        "9.0a",
        "9.0+PTX",
    ],
    (12, 8): [
        "5.0",
        "5.2",
        "5.3",
        "6.0",
        "6.1",
        "6.2",
        "7.0",
        "7.2",
        "7.5",
        "8.0",
        "8.6",
        "8.7",
        "8.9",
        "9.0a",
        "9.0",
        "10.0",
        "12.0+PTX",
    ],
    (13, 0): [
        "7.5",
        "8.0",
        "8.6",
        "8.7",
        "8.9",
        "9.0a",
        "9.0",
        "10.0",
        "10.3",
        "11.0",
        "12.0",
        "12.1+PTX",
    ],
    (13, 2): [
        "7.5",
        "8.0",
        "8.6",
        "8.7",
        "8.9",
        "9.0a",
        "9.0",
        "10.0",
        "10.3",
        "11.0",
        "12.0",
        "12.1+PTX",
    ],
    (13, 4): [
        "7.5",
        "8.0",
        "8.6",
        "8.7",
        "8.9",
        "9.0a",
        "9.0",
        "10.0",
        "10.3",
        "10.7",
        "11.0",
        "12.0",
        "12.1+PTX",
    ],
}

if TORCH_CUDA_ARCH_LIST not in os.environ:
    if torch.version.cuda is None:
        raise RuntimeError("Couldn't infer CUDA version.")

    major, minor = tuple(map(int, torch.version.cuda.split(".")[:2]))
    version = (major, minor)

    if version not in CUDA_ARCHS:
        raise RuntimeError(f"No CUDA architectures defined for CUDA {version}.")

    os.environ[TORCH_CUDA_ARCH_LIST] = " ".join(CUDA_ARCHS[version])


def _nvcc_extra_compile_args(*, debug_mode: bool) -> list[str]:
    extra_compile_args = [
        "-D__STRICT_ANSI__",
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION}",
        "-O0" if debug_mode else "-O3",
    ]

    if debug_mode:
        extra_compile_args.append("-g")

    return extra_compile_args


def _cxx_extra_compile_args(*, debug_mode: bool) -> list[str]:
    extra_compile_args = [
        f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION}",
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
        str(Path("include").resolve()),
    ]


debug_mode = os.getenv(DEBUG_MODE, "false").strip().lower() == "true"

csrc = Path("csrc")

# Project metadata lives in pyproject.toml
setup(
    ext_modules=[
        CUDAExtension(
            name="torchcc._C",
            sources=[
                *(p.as_posix() for p in csrc.rglob("*.cpp")),
                *(p.as_posix() for p in csrc.rglob("*.cu")),
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
