#ifndef TORCHCC_H
#define TORCHCC_H

#include <torch/torch.h>

namespace torchcc
{
    torch::Tensor cc2d(const torch::Tensor &input, const uint8_t connectivity);
    torch::Tensor cc3d(const torch::Tensor &input, const uint8_t connectivity);
}

#endif // TORCHCC_H
