// Copyright (c) 2024 Bendabir.
#ifndef TORCHCC_H
#define TORCHCC_H

#include <torch/torch.h>

namespace torchcc
{
    torch::Tensor ccl2d(const torch::Tensor &input, const uint8_t connectivity);
    torch::Tensor ccl3d(const torch::Tensor &input, const uint8_t connectivity);
}

#endif // TORCHCC_H
