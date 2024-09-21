#ifndef TORCHCC_H
#define TORCHCC_H

#include <torch/torch.h>

namespace torchcc
{
    // Just for first build tests and PoC
    // FIXME : Remove this
    torch::Tensor identity(const torch::Tensor &tensor);
}

#endif // TORCHCC_H
