#include <torch/torch.h>
#include <stdexcept>
#include <torchcc.h>

namespace torchcc
{
    torch::Tensor cc2d(const torch::Tensor &input, const uint8_t connectivity)
    {
        TORCH_CHECK_VALUE(
            (connectivity == 4) || (connectivity == 8),
            "Only 4-connectivity and 8-connectivity are supported.");

        const size_t ndim = input.ndimension();

        TORCH_CHECK_VALUE(input.is_cuda(), "Input must be a CUDA tensor.");
        TORCH_CHECK_VALUE((ndim == 2) || (ndim == 3), "Input must be an image [H, W] or a batch [N, H, W].");
        TORCH_CHECK_TYPE(input.scalar_type() == torch::kUInt8, "Input must be uint8 data.");

        return torch::zeros_like(input); // TODO
    }

    torch::Tensor cc3d(const torch::Tensor &input, const uint8_t connectivity)
    {
        TORCH_CHECK_VALUE(
            (connectivity == 6) || (connectivity == 26),
            "Only 6-connectivity and 26-connectivity are supported.");

        const size_t ndim = input.ndimension();

        TORCH_CHECK_VALUE(input.is_cuda(), "Input must be a CUDA tensor.");
        TORCH_CHECK_VALUE((ndim == 3) || (ndim == 4), "Input must be a volume [H, W, D] or a batch [N, H, W, D].");
        TORCH_CHECK_TYPE(input.scalar_type() == torch::kUInt8, "Input must be uint8 data.");

        return torch::zeros_like(input); // TODO
    }
}
