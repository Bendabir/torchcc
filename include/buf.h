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

// One thread per 2x2x2 block. A thread block is a tile of X x Y x Z blocks (columns, rows, planes)
// with BUF_3D_THREADS threads. X is up to BUF_3D_BLOCK_X so warps are coalesced on a row,
// the host shrinks X (then Y) when the volume is thin, so no thread is idle on small D or W.
// Measured on sm_86 : 32x8x1 is 9-16% faster than 3D tiles (16x4x4, 32x4x2, 64x2x2) on usual volumes.
constexpr uint32_t BUF_3D_THREADS = 256;
constexpr uint32_t BUF_3D_BLOCK_X = 32;
constexpr uint32_t BUF_3D_BLOCK_Y = 8;

namespace buf
{
    namespace ccl2d
    {
        // Join the blocks inside each tile with a union-find in shared memory (see csrc/buf.cu).
        __global__ void localMerge(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
        // Join the blocks across tile borders in global memory (see csrc/buf.cu).
        __global__ void borderMerge(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
        // Find the root of each block and write the pixel labels (see csrc/buf.cu).
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

    namespace ccl3d
    {
        // Make each block its own root (see csrc/buf.cu).
        __global__ void init(
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t d,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
        // Join the connected blocks in global memory (see csrc/buf.cu).
        __global__ void merge(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            const int64_t sw,
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t d,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
        // Find the root of each block and write the voxel labels (see csrc/buf.cu).
        __global__ void finalize(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            const int64_t sw,
            int32_t *__restrict__ const g_batch_blocks,
            int32_t *__restrict__ const g_batch_labels,
            const uint32_t d,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
    }
}

#endif // BUF_H
