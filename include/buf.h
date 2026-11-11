// Copyright (c) 2024 Bendabir.
#ifndef BUF_H
#define BUF_H

#include <cuda.h>

namespace buf
{
    namespace ccl2d
    {
        __global__ void init(
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t bw,
            const uint32_t bh,
            const uint32_t n);
        __global__ void merge(
            const uint8_t *__restrict__ const g_batch_img,
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
        __global__ void compress(
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t bw,
            const uint32_t bh,
            const uint32_t n);
        __global__ void finalize(
            const uint8_t *__restrict__ const g_batch_img,
            const int32_t *__restrict__ const g_batch_blocks,
            int32_t *__restrict__ const g_batch_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
    }
}

#endif // BUF_H
