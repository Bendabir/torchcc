"""Copyright (c) 2024 Bendabir."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import torch

try:
    import cv2
except ImportError:
    CV2_INSTALLED = False
else:
    CV2_INSTALLED = True


def cc2d(x: torch.Tensor, connectivity: Literal[4, 8]) -> torch.Tensor:
    """Run Connected Components Labeling on 2D images (or batches) on CPU.

    Note
    ----
    This is basically a wrapper on OpenCV for convenience.

    Parameters
    ----------
    x : torch.Tensor
        Data to perform CCL on.
        It must be an image (H, W) or a batch of images (N, H, W).
        Only uint8 data is supported.
    connectivity : {4, 8}
        Define how to perform CCL.
        With 4-connectivity, pixels are considered connected if they share an edge.
        With 8-connectivity, pixels are considered connected if they share at least
        a vertex.

    Returns
    -------
    torch.Tensor
        The labeled Connected Components.

    Raises
    ------
    RuntimeError
        If the data is not CPU and OpenCV is not installed.
    ValueError
        If the provided data is not an image or a batch of images.
    """
    if CV2_INSTALLED:
        warnings.warn(
            "Falling back to OpenCV for CCL on CPU. "
            "This will (highly) affect performances.",
            UserWarning,
            stacklevel=2,
        )

        if x.ndim == 2:  # noqa: PLR2004
            _, labels = cv2.connectedComponents(x.numpy(), connectivity=connectivity)

            return torch.from_numpy(labels)

        if x.ndim == 3:  # noqa: PLR2004
            data = x.numpy()
            labels = np.zeros(data.shape, dtype=np.uint8)

            for i in range(len(data)):
                _, cc = cv2.connectedComponents(data[i], connectivity=connectivity)

                labels[i] = cc

            return torch.from_numpy(labels)

        raise ValueError("Input must be an image [H, W] or a batch [N, H, W].")

    raise RuntimeError(
        "2D CCL on CPU is not natively supported. "
        'Install OpenCV for CPU support : pip install "opencv-python-headless>=4,<5".'
    )


def cc3d(x: torch.Tensor, connectivity: Literal[6, 26]) -> torch.Tensor:
    """Run Connected Components Labeling on 3D volumes (or batches) on CPU.

    Note
    ----
    This is not supported at the moment.

    Parameters
    ----------
    x : torch.Tensor
        Data to perform CCL on.
        It must be an image (H, W, D) or a batch of images (N, H, W, D).
        Only uint8 data is supported.
    connectivity : {6, 26}
        Define how to perform CCL.
        With 6-connectivity, pixels are considered connected if they share a side.
        With 26-connectivity, pixels are considered connected if they share at least
        a vertex.

    Returns
    -------
    torch.Tensor
        The labeled Connected Components.
    """
    raise NotImplementedError("3D CCL on CPU is not supported.")
