"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from torchcc import _cpu, _cuda

if TYPE_CHECKING:
    import torch


def cc2d(x: torch.Tensor, *, connectivity: Literal[4, 8] = 8) -> torch.Tensor:
    """Run Connected Components Labeling on 2D images (or batches).

    Note
    ----
    On CPU, computation is delegated to OpenCV if installed.

    Parameters
    ----------
    x : torch.Tensor
        Data to perform CCL on.
        It must be an image (H, W) or a batch of images (N, H, W).
        Only uint8 data is supported.
    connectivity : {4, 8}, optional
        Define how to perform CCL.
        With 4-connectivity, pixels are considered connected if they share an edge.
        With 8-connectivity, pixels are considered connected if they share at least
        a vertex.
        Default is 8.

    Returns
    -------
    torch.Tensor
        The labeled Connected Components.
    """
    if x.is_cuda:
        return _cuda.cc2d(x, connectivity)

    return _cpu.cc2d(x, connectivity)  # pragma: no cover


def cc3d(x: torch.Tensor, connectivity: Literal[6, 18, 26] = 26) -> torch.Tensor:
    """Run Connected Components Labeling on 3D volumes (or batches).

    Note
    ----
    3D CCL on CPU is not yet supported.

    Parameters
    ----------
    x : torch.Tensor
        Data to perform CCL on.
        It must be an image (H, W, D) or a batch of images (N, H, W, D).
        Only uint8 data is supported.
    connectivity : {6, 18, 26}, optional
        Define how to perform CCL.
        With 6-connectivity, pixels are considered connected if they share a side.
        With 18-connectivity, pixels are considered connected if they share at least
        an edge.
        With 26-connectivity, pixels are considered connected if they share at least
        a vertex.
        Default is 26.

    Returns
    -------
    torch.Tensor
        The labeled Connected Components.
    """
    if x.is_cuda:
        return _cuda.cc3d(x, connectivity)

    return _cpu.cc3d(x, connectivity)  # pragma: no cover
