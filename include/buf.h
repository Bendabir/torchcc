// Copyright (c) 2024 Bendabir.
#ifndef BUF_H
#define BUF_H

#include <cuda.h>

namespace buf
{
    namespace ccl2d
    {
        __global__ void init(
            const uint8_t *const g_img,
            int32_t *const g_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
        __global__ void merge(
            const uint8_t *const g_img,
            int32_t *const g_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
        __global__ void compress(int32_t *const g_labels, const uint32_t w, const uint32_t h, const uint32_t n);
        __global__ void finalize(
            const uint8_t *const g_img,
            int32_t *const g_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n);
    }
}

#endif // BUF_H
