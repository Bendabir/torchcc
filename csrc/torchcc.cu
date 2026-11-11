// Copyright (c) 2024 Bendabir.
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <torchcc.h>
#include <buf.h>

// Use constants so it's typed (instead of using text substitution)
constexpr uint32_t BUF_2D_BLOCK_COLS = 32; // a warp reads 32 consecutive 2x2 blocks of a row
constexpr uint32_t BUF_2D_BLOCK_ROWS = 8;
constexpr uint32_t MAX_GRID_Z = 65535;

namespace torchcc
{
    torch::stable::Tensor ccl2d(const torch::stable::Tensor &x, const int64_t connectivity)
    {
        STD_TORCH_CHECK(
            (connectivity == 4) || (connectivity == 8),
            "Only 4-connectivity and 8-connectivity are supported.");
        STD_TORCH_CHECK(connectivity != 4, "2D CCL 4-connectivity is not yet supported.");

        const size_t ndim = x.dim();

        STD_TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor.");
        STD_TORCH_CHECK(ndim == 3, "Input must be a batch of images [N, H, W].");
        STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Byte, "Input must be uint8 data.");

        // TODO : Dedicated kernel for non-contiguous inputs ?
        STD_TORCH_CHECK(x.is_contiguous(), "Input must be contiguous.");

        // As each block of 2x2 pixels will take the raster index of its top-left pixel,
        // we need to ensure we don't overflow (int64_t, so the check itself cannot overflow)
        STD_TORCH_CHECK(x.numel() <= INT32_MAX, "Provided input is too big and will cause overflow on labels.");

        const uint32_t n = x.size(0);
        const uint32_t w = x.size(-1);
        const uint32_t h = x.size(-2);

        // Run on the device of the input
        const torch::stable::accelerator::DeviceGuard guard(x.get_device_index());

        // No zero fill : finalize writes every pixel (background included)
        torch::stable::Tensor labels = torch::stable::new_empty(x, x.sizes(), torch::headeronly::ScalarType::Int);

        // A grid dimension of 0 fails the launch
        if (x.numel() == 0)
        {
            return labels;
        }

        void *stream_ptr = nullptr;
        TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(x.get_device_index(), &stream_ptr));
        const cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

        // NOTE : The current implementation of the BUF algorithm requires contiguous data
        //        but we could probably deal with non-contiguous data if thinking a bit.
        const uint8_t *const x_ptr = x.const_data_ptr<uint8_t>();
        int32_t *const labels_ptr = labels.mutable_data_ptr<int32_t>();

        if (connectivity == 8)
        {
            // One thread per 2x2 block, in tiles of 32x8 blocks : warps are coalesced on a row,
            // tiles keep vertical neighbours in the same thread block. One image per grid z.
            const uint32_t image = h * w;
            const dim3 block = dim3(BUF_2D_BLOCK_COLS, BUF_2D_BLOCK_ROWS);

            // Start with BUF algorithm because it's easier to implement
            // but we should move to BKE at some point because it's more efficient.
            // We need to split the algorithm in multiple kernels
            // as we need to store intermediate results to global memory for synchronization.
            // gridDim.z is limited, so launch the batch in chunks of images.
            for (uint32_t first = 0; first < n; first += MAX_GRID_Z)
            {
                const uint32_t count = std::min(n - first, MAX_GRID_Z);
                const dim3 grid = dim3(
                    ((w + 1) / 2 + BUF_2D_BLOCK_COLS - 1) / BUF_2D_BLOCK_COLS,
                    ((h + 1) / 2 + BUF_2D_BLOCK_ROWS - 1) / BUF_2D_BLOCK_ROWS,
                    count);
                const uint8_t *const chunk_x = x_ptr + static_cast<int64_t>(first) * image;
                int32_t *const chunk_labels = labels_ptr + static_cast<int64_t>(first) * image;

                buf::ccl2d::init<<<grid, block, 0, stream>>>(chunk_x, chunk_labels, w, h, count);
                STD_CUDA_KERNEL_LAUNCH_CHECK();
                buf::ccl2d::merge<<<grid, block, 0, stream>>>(chunk_x, chunk_labels, w, h, count);
                STD_CUDA_KERNEL_LAUNCH_CHECK();
                buf::ccl2d::compress<<<grid, block, 0, stream>>>(chunk_labels, w, h, count);
                STD_CUDA_KERNEL_LAUNCH_CHECK();
                buf::ccl2d::finalize<<<grid, block, 0, stream>>>(chunk_x, chunk_labels, w, h, count);
                STD_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }

        return labels;
    }

    torch::stable::Tensor ccl3d(const torch::stable::Tensor &x, const int64_t connectivity)
    {
        STD_TORCH_CHECK(
            (connectivity == 6) || (connectivity == 18) || (connectivity == 26),
            "Only 6-connectivity, 18-connectivity and 26-connectivity are supported.");
        STD_TORCH_CHECK(connectivity != 6, "3D CCL 6-connectivity is not yet supported.");
        STD_TORCH_CHECK(connectivity != 18, "3D CCL 18-connectivity is not yet supported.");
        STD_TORCH_CHECK(connectivity != 26, "3D CCL 26-connectivity is not yet supported.");

        const size_t ndim = x.dim();

        STD_TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor.");
        STD_TORCH_CHECK(ndim == 4, "Input must be a batch of volumes [N, H, W, D].");
        STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Byte, "Input must be uint8 data.");

        return torch::stable::new_zeros(x, x.sizes(), torch::headeronly::ScalarType::Int); // TODO
    }
}
