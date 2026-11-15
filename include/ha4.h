// Copyright (c) 2026 Bendabir.
#ifndef HA4_H
#define HA4_H

#include <cuda.h>

// One warp per image row, one thread per pixel, in strips of 4 rows (one thread block per strip).
// Each warp loops over its row in chunks of 32 pixels.
constexpr uint32_t HA4_2D_BLOCK_COLS = 32; // warp size, bitmasks are 32-bit
constexpr uint32_t HA4_2D_BLOCK_ROWS = 4;  // paper

namespace ha4
{
    namespace ccl2d
    {
        // Merge the runs inside each strip of 4 rows (see csrc/ha4.cu).
        __global__ void stripLabeling(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
        // Merge the runs across the borders between strips (see csrc/ha4.cu).
        __global__ void borderMerge(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
        // Replace each label by its root (see csrc/ha4.cu).
        __global__ void relabel(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
    }
}

#endif // HA4_H
