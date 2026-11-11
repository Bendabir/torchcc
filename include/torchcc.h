// Copyright (c) 2024 Bendabir.
#ifndef TORCHCC_H
#define TORCHCC_H

// NOTE : Only use headers from torch/csrc/stable and torch/headeronly
//        to keep the binary compatible across PyTorch versions (stable ABI).
#include <torch/csrc/stable/tensor.h>

namespace torchcc
{
    torch::stable::Tensor ccl2d(const torch::stable::Tensor &input, const int64_t connectivity);
    torch::stable::Tensor ccl3d(const torch::stable::Tensor &input, const int64_t connectivity);
}

#endif // TORCHCC_H
