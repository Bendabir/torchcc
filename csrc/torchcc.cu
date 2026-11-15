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
#include <ha4.h>

namespace
{
    // Use constants so it's typed (instead of using text substitution)
    constexpr uint32_t MAX_GRID_Z = 65535; // Hardware limit of gridDim.z (one image per z)

    // Check that x is a CUDA uint8 tensor with ndim dimensions.
    void checkBatch(const torch::stable::Tensor &x, const int64_t ndim, const char *const shape_message)
    {
        STD_TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor.");
        STD_TORCH_CHECK(x.dim() == ndim, shape_message);
        STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Byte, "Input must be uint8 data.");
    }

    // What the 2D launchers need. The caller keeps the labels alive.
    struct Batch2D
    {
        torch::stable::Tensor img; // row pixels are contiguous, batch and row strides are free
        int32_t *labels;           // dense [n, h, w]
        uint32_t n;          // images in the batch
        uint32_t h;          // image height
        uint32_t w;          // image width
        cudaStream_t stream; // current stream of the input device
    };

    // HA4 : one warp per row, one block per strip of 4 rows. stripLabeling merges the runs
    // inside each strip, borderMerge the runs across strips, relabel finds the roots.
    // Union-find runs in the labels, so no extra buffer.
    void launchHA4(const Batch2D &b)
    {
        const uint8_t *const img_ptr = b.img.const_data_ptr<uint8_t>();
        const int64_t sn = b.img.stride(0);
        const int64_t sh = b.img.stride(1);
        const dim3 block = dim3(HA4_2D_BLOCK_COLS, HA4_2D_BLOCK_ROWS);
        // grid.x goes to 2^31 - 1 and h * w <= INT32_MAX, so no limit on h or w here
        const uint32_t strips = (b.h + HA4_2D_BLOCK_ROWS - 1) / HA4_2D_BLOCK_ROWS;
        const uint32_t chunks = (b.w + HA4_2D_BLOCK_COLS - 1) / HA4_2D_BLOCK_COLS;
        // borderMerge : one warp per border row, so a block covers 4 strips
        const uint32_t strip_groups = (strips + HA4_2D_BLOCK_ROWS - 1) / HA4_2D_BLOCK_ROWS;

        // gridDim.z is limited, so launch the batch in chunks of images (one image per grid z)
        for (uint32_t first = 0; first < b.n; first += MAX_GRID_Z)
        {
            const uint32_t count = std::min(b.n - first, MAX_GRID_Z);
            const uint8_t *const img = img_ptr + first * sn;
            int32_t *const labels = b.labels + static_cast<int64_t>(first) * b.h * b.w;

            ha4::ccl2d::stripLabeling<<<dim3(strips, 1, count), block, 0, b.stream>>>(img, sn, sh, labels, b.w, b.h, count);
            STD_CUDA_KERNEL_LAUNCH_CHECK();
            ha4::ccl2d::borderMerge<<<dim3(strip_groups * chunks, 1, count), block, 0, b.stream>>>(img, sn, sh, labels, b.w, b.h, count);
            STD_CUDA_KERNEL_LAUNCH_CHECK();
            ha4::ccl2d::relabel<<<dim3(strips * chunks, 1, count), block, 0, b.stream>>>(img, sn, sh, labels, b.w, b.h, count);
            STD_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    // Start with BUF algorithm because it's easier to implement
    // but we should move to BKE at some point because it's more efficient.
    // One thread per 2x2 block, in tiles of 32x8 blocks : warps are coalesced on a row,
    // localMerge joins the blocks of a tile in shared memory, borderMerge only the links across tiles,
    // finalize finds the roots.
    void launchBUF(const Batch2D &b)
    {
        const uint8_t *const img_ptr = b.img.const_data_ptr<uint8_t>();
        const int64_t sn = b.img.stride(0);
        const int64_t sh = b.img.stride(1);

        // Union-find runs on a compact buffer with one label per 2x2 block (about the size of the input).
        // The paper stores block labels in the output (top-left pixel of each block) to avoid cudaMalloc.
        // The caching allocator makes this allocation cheap, and the dense layout uses whole 32 B sectors
        // (the sparse one used 1 int out of 4) with a 4x smaller working set for find.
        // Measured on sm_86 only : 17-27% faster at batch 32, neutral at batch 1.
        // Cost : about +1 B per pixel, so peak memory goes from 5 B to 6 B per pixel (+20%).
        // If a compute capability (or a memory-bound use case) needs the in-place layout back,
        // the sparse version is in commit 5960911.
        const uint32_t bw = (b.w + 1) / 2;
        const uint32_t bh = (b.h + 1) / 2;
        torch::stable::Tensor blocks = torch::stable::new_empty(
            b.img,
            {static_cast<int64_t>(b.n), static_cast<int64_t>(bh), static_cast<int64_t>(bw)},
            torch::headeronly::ScalarType::Int);
        int32_t *const blocks_ptr = blocks.mutable_data_ptr<int32_t>();

        const dim3 block = dim3(BUF_2D_BLOCK_COLS, BUF_2D_BLOCK_ROWS);
        const dim3 border_block = dim3(BUF_2D_BORDER_THREADS, BUF_2D_BORDER_TILES);
        const uint32_t tiles_x = (bw + BUF_2D_BLOCK_COLS - 1) / BUF_2D_BLOCK_COLS;
        const uint32_t tiles_y = (bh + BUF_2D_BLOCK_ROWS - 1) / BUF_2D_BLOCK_ROWS;

        // gridDim.z is limited, so launch the batch in chunks of images (one image per grid z)
        for (uint32_t first = 0; first < b.n; first += MAX_GRID_Z)
        {
            const uint32_t count = std::min(b.n - first, MAX_GRID_Z);
            const uint8_t *const img = img_ptr + first * sn;
            int32_t *const labels = b.labels + static_cast<int64_t>(first) * b.h * b.w;
            int32_t *const chunk_blocks = blocks_ptr + static_cast<int64_t>(first) * bh * bw;
            const dim3 grid = dim3(tiles_x, tiles_y, count);
            const dim3 border_grid = dim3((tiles_x + BUF_2D_BORDER_TILES - 1) / BUF_2D_BORDER_TILES, tiles_y, count);

            buf::ccl2d::localMerge<<<grid, block, 0, b.stream>>>(img, sn, sh, chunk_blocks, b.w, b.h, count);
            STD_CUDA_KERNEL_LAUNCH_CHECK();
            buf::ccl2d::borderMerge<<<border_grid, border_block, 0, b.stream>>>(
                img,
                sn,
                sh,
                chunk_blocks,
                b.w,
                b.h,
                count);
            STD_CUDA_KERNEL_LAUNCH_CHECK();
            buf::ccl2d::finalize<<<grid, block, 0, b.stream>>>(img, sn, sh, chunk_blocks, labels, b.w, b.h, count);
            STD_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
}

namespace torchcc
{
    // Host entry point of 2D CCL. Checks the input, allocates the int32 labels [N, H, W]
    // and launches HA4 (4-connectivity) or BUF (8-connectivity) on the current stream.
    // Asynchronous. Empty inputs return empty labels.
    torch::stable::Tensor ccl2d(const torch::stable::Tensor &x, const int64_t connectivity)
    {
        STD_TORCH_CHECK(
            (connectivity == 4) || (connectivity == 8),
            "Only 4-connectivity and 8-connectivity are supported.");
        checkBatch(x, 3, "Input must be a batch of images [N, H, W].");

        // Labels are raster indices in an image (0-based for BUF, 1-based for HA4, so up to h * w),
        // we need to ensure we don't overflow (int64_t, so the check itself cannot overflow)
        STD_TORCH_CHECK(
            (x.size(-2) * x.size(-1)) <= INT32_MAX,
            "Images are too big and will cause overflow on labels."
        );

        // Run on the device of the input
        const torch::stable::accelerator::DeviceGuard guard(x.get_device_index());

        // No zero fill : finalize (BUF) and relabel (HA4) write every pixel (background included)
        torch::stable::Tensor labels = torch::stable::new_empty(x, x.sizes(), torch::headeronly::ScalarType::Int);

        // A grid dimension of 0 fails the launch
        if (x.numel() == 0)
        {
            return labels;
        }

        void *stream_ptr = nullptr;
        TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(x.get_device_index(), &stream_ptr));

        // Kernels read the image with its batch and row strides, so a slice of a bigger tensor
        // (e.g. x[:, c] from [N, C, H, W]) needs no copy. Only the pixels of a row must be contiguous.
        const Batch2D batch = {
            x.stride(2) == 1 ? x : torch::stable::contiguous(x),
            labels.mutable_data_ptr<int32_t>(),
            static_cast<uint32_t>(x.size(0)),
            static_cast<uint32_t>(x.size(1)),
            static_cast<uint32_t>(x.size(2)),
            static_cast<cudaStream_t>(stream_ptr),
        };

        // We need to split the algorithms in multiple kernels
        // as we need to store intermediate results to global memory for synchronization.
        if (connectivity == 4)
        {
            launchHA4(batch);
        }
        else
        {
            launchBUF(batch);
        }

        return labels;
    }

    // Host entry point of 3D CCL. Not implemented yet : every valid connectivity (6, 18, 26)
    // is rejected, so the zero-filled return at the end is unreachable for now.
    torch::stable::Tensor ccl3d(const torch::stable::Tensor &x, const int64_t connectivity)
    {
        STD_TORCH_CHECK(
            (connectivity == 6) || (connectivity == 18) || (connectivity == 26),
            "Only 6-connectivity, 18-connectivity and 26-connectivity are supported.");
        STD_TORCH_CHECK(connectivity != 6, "3D CCL 6-connectivity is not yet supported.");
        STD_TORCH_CHECK(connectivity != 18, "3D CCL 18-connectivity is not yet supported.");
        STD_TORCH_CHECK(connectivity != 26, "3D CCL 26-connectivity is not yet supported.");
        checkBatch(x, 4, "Input must be a batch of volumes [N, H, W, D].");

        return torch::stable::new_zeros(x, x.sizes(), torch::headeronly::ScalarType::Int);
    }
}
