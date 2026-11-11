"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

from typing import Literal

import torch

# Importing the extension registers the torch.ops.torchcc.* operators.
from torchcc import _C, _cpu  # noqa: F401


def ccl2d(x: torch.Tensor, *, connectivity: Literal[4, 8] = 8) -> torch.Tensor:
    """Run Connected Components Labeling on 2D batches of images.

    Note
    ----
    On CPU, computation is delegated to OpenCV if installed.
    On GPU, contiguous data is required.

    Parameters
    ----------
    x : torch.Tensor
        Data to perform CCL on.
        It must be a batch of images (N, H, W).
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
        # The stable ABI only raises RuntimeError, so validate here
        # to raise meaningful exceptions.
        if connectivity not in {4, 8}:
            raise ValueError("Only 4-connectivity and 8-connectivity are supported.")

        if connectivity == 4:  # noqa: PLR2004
            raise NotImplementedError("2D CCL 4-connectivity is not yet supported.")

        if x.ndim != 3:  # noqa: PLR2004
            raise ValueError("Input must be a batch of images [N, H, W].")

        if x.dtype != torch.uint8:
            raise TypeError("Input must be uint8 data.")

        if x.numel() > torch.iinfo(torch.int32).max:
            raise ValueError(
                "Provided input is too big and will cause overflow on labels."
            )

        # TODO : Fix contiguous requirement.
        return torch.ops.torchcc.ccl2d(x.contiguous(), connectivity)

    return _cpu.ccl2d(x, connectivity)  # pragma: no cover


def ccl3d(x: torch.Tensor, connectivity: Literal[6, 18, 26] = 26) -> torch.Tensor:
    """Run Connected Components Labeling on 3D batches of volumes.

    Note
    ----
    3D CCL on CPU is not yet supported.

    Parameters
    ----------
    x : torch.Tensor
        Data to perform CCL on.
        It must be a batch of volumes (N, H, W, D).
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
        if connectivity not in {6, 18, 26}:
            raise ValueError(
                "Only 6-connectivity, 18-connectivity and 26-connectivity "
                "are supported."
            )

        # TODO : Call torch.ops.torchcc.ccl3d once it is implemented.
        raise NotImplementedError(
            f"3D CCL {connectivity}-connectivity is not yet supported."
        )

    return _cpu.ccl3d(x, connectivity)  # pragma: no cover
