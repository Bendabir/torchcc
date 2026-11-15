// Copyright (c) 2024 Bendabir.
#ifndef TORCHCC_H
#define TORCHCC_H

// NOTE : Only use headers from torch/csrc/stable and torch/headeronly
//        to keep the binary compatible across PyTorch versions (stable ABI).
#include <torch/csrc/stable/tensor.h>

namespace torchcc
{
    // Label the connected components of a batch of images [N, H, W] (CUDA, uint8).
    // connectivity is 4 or 8. Returns int32 labels of the same shape (0 is background).
    // Labels are not consecutive.
    torch::stable::Tensor ccl2d(const torch::stable::Tensor &input, const int64_t connectivity);
    // Label the connected components of a batch of volumes [N, H, W, D] (CUDA, uint8).
    // Only 26-connectivity is implemented (6 and 18 raise). Returns int32 labels of the same shape
    // (0 is background). Labels are unique per volume, not consecutive.
    torch::stable::Tensor ccl3d(const torch::stable::Tensor &input, const int64_t connectivity);
}

#endif // TORCHCC_H
