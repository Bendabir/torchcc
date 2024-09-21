#include <torch/torch.h>
#include <stdexcept>
#include <torchcc.h>

namespace torchcc
{
    // Just for first build tests and PoC
    // FIXME : Remove this
    torch::Tensor identity(const torch::Tensor &tensor)
    {
        return tensor;
    }
}
