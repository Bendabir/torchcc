// Copyright (c) 2024 Bendabir.
#include <buf.h>
#include <cuda.h>

#define BLOCK_ROWS 16
#define BLOCK_COLS 16
#define BITMASK_3x3 0x0777
#define BITMASK_3x4R 0xEEEE // all but col 0
#define BITMASK_2x4L 0x3333 // only col O and 1
#define BITMASK_3x4L 0x7777 // all but col 3
#define BITMASK_4x3B 0xFFF0 // all but row 0
#define BITMASK_4x2T 0x00FF // only row O and 1
#define BITMASK_4x3T 0x0FFF // all but row 3

// The 3x3 bitmask correspond to the neigbours of the top-left internal pixel
// (centered on the X block).
//     1 1 1 0    . . . .
//     1 1 1 0    . X X .
//     1 1 1 0    . X X .
//     0 0 0 0    . . . .
//
// When shifted 1 left
//     0 1 1 1
//     0 1 1 1
//     0 1 1 1
//     0 0 0 0
//
// When shifted 4 left
//     0 0 0 0
//     1 1 1 0
//     1 1 1 0
//     1 1 1 0
//
// To determine connectivity, we only need to check for pixels 0, 1, 2, 3, 4 and 8
// around the surrounding X block.
//     0 1 2 3
//     4 X X .
//     8 X X .
//     . . . .
//
// LSB is pixel 0, MSB is pixel 15.
//
// Blocks are organised as follow
//     P P Q Q R R
//     P P Q Q R R
//     S S X X
//     S S X X

namespace buf
{
    namespace ccl2d
    {
        // Only use it with unsigned numeric types
        template <typename T>
        __device__ __forceinline__ uint8_t hasBit(T bitmap, uint8_t pos)
        {
            return (bitmap >> pos) & 1;
        }

        // Returns the root index of the UFTree
        // (the identiﬁer of the subset that contains a)
        __device__ uint32_t find(const int32_t *const g_buffer, uint32_t n)
        {
            while (g_buffer[n] != n)
            {
                n = g_buffer[n];
            }

            return n;
        }

        __device__ uint32_t findAndCompress(int32_t *const g_buffer, uint32_t n)
        {
            const uint32_t id = n;

            while (g_buffer[n] != n)
            {
                n = g_buffer[n];
                g_buffer[id] = n;
            }

            return n;
        }

        // Merges the UFTrees of a and b, linking one root to the other
        // (joins the subsets containing a and b)
        __device__ void computeUnion(int32_t *const g_buffer, uint32_t a, uint32_t b)
        {
            bool done = false;

            do
            {

                a = find(g_buffer, a);
                b = find(g_buffer, b);

                if (a < b)
                {
                    int32_t old = atomicMin(g_buffer + b, a);
                    done = (old == b);
                    b = old;
                }
                else if (b < a)
                {
                    int32_t old = atomicMin(g_buffer + a, b);
                    done = (old == a);
                    a = old;
                }
                else
                {
                    done = true;
                }

            } while (!done);
        }

        // --- KERNELS ---
        __global__ void init(const uint8_t *const g_img, int32_t *const g_labels, const uint32_t w, const uint32_t h)
        {
            const uint32_t row = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
            const uint32_t col = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
            const uint32_t index = row * w + col;

            // Assign each block to the raster index of the top-left pixel
            if ((row >= h) || (col >= w))
            {
                return;
            }

            g_labels[index] = index;
        }

        __global__ void merge(const uint8_t *const g_img, int32_t *const g_labels, const uint32_t w, const uint32_t h)
        {
            const uint32_t row = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
            const uint32_t col = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
            const uint32_t index = row * w + col; // Basically pixel 5

            if ((row >= h) || (col >= w))
            {
                return;
            }

            uint16_t bitset = 0;

            // TODO : Use a buffer in register memory (or shared memory)
            //        to load the pixels of interest at once from global memory

            // First, check the pixels of the block
            // No check on the bottom-right pixel as it's never responsible for connections between blocks
            if (g_img[index]) // top-left
            {
                bitset |= BITMASK_3x3;
            }

            if ((row <= h) && g_img[index + 1]) // top-right, checking we don't overflow
            {
                bitset |= BITMASK_3x3 << 1;
            }

            if ((col <= w) && g_img[index + w]) // bottom-left, checking we don't overflow
            {
                bitset |= BITMASK_3x3 << 4;
            }

            // Check the different edge cases
            if (col == 0) // no left-pixel, droping 1st column of the bitset
            {
                bitset &= BITMASK_3x4R;
            }

            if (col > w) // no right-pixel, but blocks are 2-pixels wide, dropping the 2 last columns of the bitset
            {
                bitset &= BITMASK_2x4L;
            }
            else if (col + 1 > w) // almost on the right edge, dropping the last column of the bitset
            {
                bitset &= BITMASK_3x4L;
            }

            if (row == 0) // likewise, no top-pixel
            {
                bitset &= BITMASK_4x3B;
            }

            if (row > h) // no bottom-pixel, but 2-pixels large, similarly dropping the last 2 rows of the bitset
            {
                bitset &= BITMASK_4x2T;
            }
            else if (row + 1 > h) // again, almost on the bottom edge, dropping the last row of the bitset
            {
                bitset &= BITMASK_4x3T;
            }

            // We can now check for neighbour blocks
            if (!bitset)
            {
                return;
            }

            // If we have a top-left pixel (5) and a top-left pixel (0)
            // i.e. is pixel 0 connected to block X
            if (hasBit(bitset, 0) && g_img[index - w - 1])
            {
                // Merge block X with block P (top-left)
                computeUnion(g_labels, index, index - 2 * w - 2); // above, left
            }

            // Check if we have pixels in the bottom of the top block
            // i.e. are pixels 1 or 2 connected to block X
            if ((hasBit(bitset, 1) && g_img[index - w]) || (hasBit(bitset, 2) && g_img[index - w + 1]))
            {
                // Merge block X with block Q
                computeUnion(g_labels, index, index - 2 * w); // above
            }

            // Check if we have a top-right pixel and a top-right diagonal pixel
            // i.e. is pixel 3 connected to block X
            if (hasBit(bitset, 3) && g_img[index - w + 2])
            {
                // Merge block X with block R
                computeUnion(g_labels, index, index - 2 * w + 2); // above, right
            }

            // Check if we have pixels in the right of the left block
            // i.e. are pixels 4 or 8 connected to block X
            if ((hasBit(bitset, 4) && g_img[index - 1]) || (hasBit(bitset, 8) && g_img[index + w - 1]))
            {
                // Merge block X with block S
                computeUnion(g_labels, index, index - 2); // left
            }
        }

        __global__ void compress(int32_t *const g_labels, const uint32_t w, const uint32_t h)
        {
            const uint32_t row = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
            const uint32_t col = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
            const uint32_t index = row * w + col;

            if ((row >= h) || (col >= w))
            {
                return;
            }

            findAndCompress(g_labels, index);
        }

        __global__ void finalize(
            const uint8_t *const g_img,
            int32_t *const g_labels,
            const uint32_t w,
            const uint32_t h)
        {
            const uint32_t row = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
            const uint32_t col = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
            const uint32_t index = row * w + col;

            if ((row >= h) || (col >= w))
            {
                return;
            }

            uint32_t label = g_labels[index] + 1;

            // Pseudo-algorithm of the paper uses multiplication to avoid checks
            // but I suppose that checks are more efficient
            // Apply the labels of the different blocks uniformaly
            // Accounting for edges
            g_labels[index] = g_img[index] ? label : 0;

            if (col + 1 < w)
            {
                g_labels[index + 1] = g_img[index + 1] ? label : 0;

                if (row + 1 < h)
                {
                    g_labels[index + w + 1] = g_img[index + w + 1] ? label : 0;
                }
            }

            if (row + 1 < h)
            {
                g_labels[index + w] = g_img[index + w] ? label : 0;
            }
        }
    }
}
