#include <torch/torch.h>
#include <stdexcept>
#include <torchcc.h>

namespace torchcc
{
    torch::Tensor cc2d(const torch::Tensor &input, const uint8_t connectivity)
    {
        return torch::zeros_like(input, torch::TensorOptions(torch::kUInt8));
    }

    torch::Tensor cc3d(const torch::Tensor &input, const uint8_t connectivity)
    {
        return torch::zeros_like(input, torch::TensorOptions(torch::kUInt8));
    }
}
