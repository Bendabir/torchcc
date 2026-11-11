// Copyright (c) 2024 Bendabir.
#ifndef BUF_H
#define BUF_H

#include <cuda.h>

// One thread per 2x2 block, in tiles of 32x8 blocks : warps are coalesced on a row,
// and the kernels run a union-find in shared memory per tile.
constexpr uint32_t BUF_2D_BLOCK_COLS = 32;
constexpr uint32_t BUF_2D_BLOCK_ROWS = 8; // 16 decreased performances by around 5-15%
// Blocks on the top row, left column and right column of a tile
constexpr uint32_t BUF_2D_BORDER_THREADS = BUF_2D_BLOCK_COLS + 2 * (BUF_2D_BLOCK_ROWS - 1);
// Tiles per thread block in borderMerge (one tile per block would cap occupancy at 16 small blocks per SM)
constexpr uint32_t BUF_2D_BORDER_TILES = 4;

namespace buf
{
    namespace ccl2d
    {
        __global__ void localMerge(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
        __global__ void borderMerge(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
        __global__ void finalize(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_blocks,
            int32_t *__restrict__ const g_batch_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
    }
}

#endif // BUF_H
