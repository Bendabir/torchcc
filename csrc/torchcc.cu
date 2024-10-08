// Copyright (c) 2024 Bendabir.
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <stdexcept>
#include <torchcc.h>
#include <buf.h>

#define BUF_2D_BLOCK_ROWS 16
#define BUF_2D_BLOCK_COLS 16
#define BUF_2D_BLOCK_DEPTH 4 // 4 * 16 * 16 = 1024 (threads in a block)

namespace torchcc
{
    torch::Tensor ccl2d(const torch::Tensor &x, const uint8_t connectivity)
    {
        TORCH_CHECK_VALUE(
            (connectivity == 4) || (connectivity == 8),
            "Only 4-connectivity and 8-connectivity are supported.");
        TORCH_CHECK_NOT_IMPLEMENTED(connectivity != 4, "2D CCL 4-connectivity is not yet supported.");

        const size_t ndim = x.ndimension();

        TORCH_CHECK_VALUE(x.is_cuda(), "Input must be a CUDA tensor.");
        TORCH_CHECK_VALUE(ndim == 3, "Input must be a batch of images [N, H, W].");
        TORCH_CHECK_TYPE(x.scalar_type() == torch::kUInt8, "Input must be uint8 data.");

        const uint32_t n = x.size(0);
        const uint32_t w = x.size(-1);
        const uint32_t h = x.size(-2);

        // As each block of 2x2 pixels will take the raster index of its top-left pixel,
        // we need to ensure we don't overflow
        TORCH_CHECK_VALUE(w * h * n <= INT32_MAX, "Provided input is too big and will cause overflow on labels.");

        const torch::TensorOptions options = torch::TensorOptions(torch::kInt32);
        torch::Tensor labels = torch::zeros_like(x, options);

        const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        // NOTE : There are new const_data_ptr and mutable_data_ptr accessors
        //        but keep legacy ones for backward compatibility.
        //        Especially because TORCH_VERSION_* definition were introduced with Torch 1.8
        // NOTE : The current implementation of the BUF algorithm requires contiguous data
        //        but we could probably deal with non-contiguous data if thinking a bit.
        const uint8_t *const x_ptr = x.data_ptr<uint8_t>();
        int32_t *const labels_ptr = labels.data_ptr<int32_t>();

        if (connectivity == 8)
        {
            const dim3 grid = dim3(
                // Working on block of 2x2 pixels
                ((w + 1) / 2 + BUF_2D_BLOCK_COLS - 1) / BUF_2D_BLOCK_COLS,
                ((h + 1) / 2 + BUF_2D_BLOCK_ROWS - 1) / BUF_2D_BLOCK_ROWS,
                // on each image
                ((n + 1) + BUF_2D_BLOCK_DEPTH - 1) / BUF_2D_BLOCK_DEPTH);
            const dim3 block = dim3(BUF_2D_BLOCK_COLS, BUF_2D_BLOCK_ROWS, BUF_2D_BLOCK_DEPTH);

            // Start with BUF algorithm because it's easier to implement
            // but we should move to BKE at some point because it's more efficient.
            // We need to split the algorithm in multiple kernels
            // as we need to store intermediate results to global memory for synchronization.
            buf::ccl2d::init<<<grid, block, 0, stream>>>(x_ptr, labels_ptr, w, h, n);
            buf::ccl2d::merge<<<grid, block, 0, stream>>>(x_ptr, labels_ptr, w, h, n);
            buf::ccl2d::compress<<<grid, block, 0, stream>>>(labels_ptr, w, h, n);
            buf::ccl2d::finalize<<<grid, block, 0, stream>>>(x_ptr, labels_ptr, w, h, n);
        }

        return labels;
    }

    torch::Tensor ccl3d(const torch::Tensor &x, const uint8_t connectivity)
    {
        TORCH_CHECK_VALUE(
            (connectivity == 6) || (connectivity == 18) || (connectivity == 26),
            "Only 6-connectivity, 18-connectivity and 26-connectivity are supported.");
        TORCH_CHECK_NOT_IMPLEMENTED(connectivity != 6, "3D CCL 6-connectivity is not yet supported.");
        TORCH_CHECK_NOT_IMPLEMENTED(connectivity != 18, "3D CCL 18-connectivity is not yet supported.");
        TORCH_CHECK_NOT_IMPLEMENTED(connectivity != 26, "3D CCL 26-connectivity is not yet supported.");

        const size_t ndim = x.ndimension();

        TORCH_CHECK_VALUE(x.is_cuda(), "Input must be a CUDA tensor.");
        TORCH_CHECK_VALUE(ndim == 4, "Input must be a batch of volumes [N, H, W, D].");
        TORCH_CHECK_TYPE(x.scalar_type() == torch::kUInt8, "Input must be uint8 data.");

        const torch::TensorOptions options = torch::TensorOptions(torch::kInt32);

        return torch::zeros_like(x, options); // TODO
    }
}
