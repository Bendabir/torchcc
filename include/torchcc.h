#include <torch/torch.h>

namespace torchcc
{
    // Just for first build tests and PoC
    // FIXME : Remove this
    torch::Tensor identity(const torch::Tensor &tensor);
}
