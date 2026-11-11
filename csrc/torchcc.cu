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


        // As each block of 2x2 pixels will take the raster index (in its image) of its top-left pixel,
        // we need to ensure we don't overflow (int64_t, so the check itself cannot overflow)
        STD_TORCH_CHECK(
            (x.size(-2) * x.size(-1)) <= INT32_MAX,
            "Images are too big and will cause overflow on labels."
        );

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

        // Kernels read the image with its batch and row strides, so a slice of a bigger tensor
        // (e.g. x[:, c] from [N, C, H, W]) needs no copy. Only the pixels of a row must be contiguous.
        const torch::stable::Tensor img = x.stride(2) == 1 ? x : torch::stable::contiguous(x);
        const int64_t sn = img.stride(0);
        const int64_t sh = img.stride(1);
        const uint8_t *const x_ptr = img.const_data_ptr<uint8_t>();
        int32_t *const labels_ptr = labels.mutable_data_ptr<int32_t>();

        if (connectivity == 8)
        {
            // Union-find runs on a compact buffer with one label per 2x2 block (about the size of the input).
            // The paper stores block labels in the output (top-left pixel of each block) to avoid cudaMalloc.
            // The caching allocator makes this allocation cheap, and the dense layout uses whole 32 B sectors
            // (the sparse one used 1 int out of 4) with a 4x smaller working set for find.
            // Measured on sm_86 only : 17-27% faster at batch 32, neutral at batch 1.
            // Cost : about +1 B per pixel, so peak memory goes from 5 B to 6 B per pixel (+20%).
            // If a compute capability (or a memory-bound use case) needs the in-place layout back,
            // the sparse version is in commit 5960911.
            const uint32_t bw = (w + 1) / 2;
            const uint32_t bh = (h + 1) / 2;
            torch::stable::Tensor blocks = torch::stable::new_empty(
                x,
                {static_cast<int64_t>(n), static_cast<int64_t>(bh), static_cast<int64_t>(bw)},
                torch::headeronly::ScalarType::Int);
            int32_t *const blocks_ptr = blocks.mutable_data_ptr<int32_t>();

            // One thread per 2x2 block, in tiles of 32x8 blocks : warps are coalesced on a row,
            // localMerge joins the blocks of a tile in shared memory, borderMerge only the links across tiles,
            // finalize finds the roots. One image per grid z.
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
                    (bw + BUF_2D_BLOCK_COLS - 1) / BUF_2D_BLOCK_COLS,
                    (bh + BUF_2D_BLOCK_ROWS - 1) / BUF_2D_BLOCK_ROWS,
                    count);
                const uint8_t *const chunk_x = x_ptr + first * sn;
                int32_t *const chunk_blocks = blocks_ptr + static_cast<int64_t>(first) * bh * bw;
                int32_t *const chunk_labels = labels_ptr + static_cast<int64_t>(first) * image;

                buf::ccl2d::localMerge<<<grid, block, 0, stream>>>(chunk_x, sn, sh, chunk_blocks, w, h, count);
                STD_CUDA_KERNEL_LAUNCH_CHECK();
                const dim3 border_grid = dim3((grid.x + BUF_2D_BORDER_TILES - 1) / BUF_2D_BORDER_TILES, grid.y, grid.z);
                buf::ccl2d::borderMerge<<<border_grid, dim3(BUF_2D_BORDER_THREADS, BUF_2D_BORDER_TILES), 0, stream>>>(chunk_x, sn, sh, chunk_blocks, w, h, count);
                STD_CUDA_KERNEL_LAUNCH_CHECK();
                buf::ccl2d::finalize<<<grid, block, 0, stream>>>(chunk_x, sn, sh, chunk_blocks, chunk_labels, w, h, count);
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

        return torch::stable::new_zeros(x, x.sizes(), torch::headeronly::ScalarType::Int);
    }
}
