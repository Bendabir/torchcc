// Copyright (c) 2026 Bendabir.
#ifndef UF_H
#define UF_H

#include <cuda.h>

// Union-find on a label array, shared by the CCL kernels.
// setup.py builds without -rdc, so device functions must live in a header to be shared.
// BASE is the label of index 0 : the node with label l is stored at g_labels[l - BASE],
// and it is a root when g_labels[l - BASE] == l (0 for BUF, 1 for HA4).
namespace uf
{
    template <uint32_t BASE>
    __device__ __forceinline__ uint32_t find(const int32_t *const g_labels, const uint32_t label)
    {
        uint32_t n = label;

        while (g_labels[n - BASE] != n)
        {
            n = g_labels[n - BASE];
        }

        return n;
    }

    template <uint32_t BASE>
    __device__ __forceinline__ uint32_t findAndCompress(int32_t *const g_labels, const uint32_t label)
    {
        uint32_t n = label;

        while (g_labels[n - BASE] != n)
        {
            n = g_labels[n - BASE];
            g_labels[label - BASE] = n;
        }

        return n;
    }

    template <uint32_t BASE>
    __device__ __forceinline__ void computeUnion(int32_t *const g_labels, uint32_t a, uint32_t b)
    {
        bool done = false;

        do
        {
            a = find<BASE>(g_labels, a);
            b = find<BASE>(g_labels, b);

            if (a < b)
            {
                const int32_t old = atomicMin(g_labels + b - BASE, a);
                done = (old == b);
                b = old;
            }
            else if (b < a)
            {
                const int32_t old = atomicMin(g_labels + a - BASE, b);
                done = (old == a);
                a = old;
            }
            else
            {
                done = true;
            }
        } while (!done);
    }
}

#endif // UF_H
