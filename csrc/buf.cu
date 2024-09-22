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
    namespace cc2d
    {
        // Only use it with unsigned numeric types
        template <typename T>
        __device__ __forceinline__ uint8_t hasBit(T d_bitmap, uint8_t d_pos)
        {
            return (d_bitmap >> d_pos) & 1;
        }

        // Returns the root index of the UFTree
        // (the identiﬁer of the subset that contains a)
        __device__ uint32_t find(const int32_t *const s_buffer, uint32_t d_n)
        {
            while (s_buffer[d_n] != d_n)
            {
                d_n = s_buffer[d_n];
            }

            return d_n;
        }

        __device__ uint32_t findAndCompress(int32_t *const s_buffer, uint32_t d_n)
        {
            const uint32_t id = d_n;

            while (s_buffer[d_n] != d_n)
            {
                d_n = s_buffer[d_n];
                s_buffer[id] = d_n;
            }

            return d_n;
        }

        // Merges the UFTrees of a and b, linking one root to the other
        // (joins the subsets containing a and b)
        __device__ void computeUnion(int32_t *const s_buffer, uint32_t d_a, uint32_t d_b)
        {
            bool done = false;

            do
            {

                d_a = find(s_buffer, d_a);
                d_b = find(s_buffer, d_b);

                if (d_a < d_b)
                {
                    int32_t old = atomicMin(s_buffer + d_b, d_a);
                    done = (old == d_b);
                    d_b = old;
                }
                else if (d_b < d_a)
                {
                    int32_t old = atomicMin(s_buffer + d_a, d_b);
                    done = (old == d_a);
                    d_a = old;
                }
                else
                {
                    done = true;
                }

            } while (!done);
        }

        // --- KERNELS ---
        __global__ void init(const uint8_t *const img, int32_t *const labels, const uint32_t w, const uint32_t h)
        {
            const uint32_t row = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
            const uint32_t col = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
            const uint32_t index = row * w + col;

            // Assign each block to the raster index of the top-left pixel
            if ((row < h) && (col < w))
            {
                labels[index] = index;
            }
        }

        __global__ void merge(const uint8_t *const img, int32_t *const labels, const uint32_t w, const uint32_t h)
        {
            const uint32_t row = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
            const uint32_t col = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
            const uint32_t index = row * w + col; // Basically pixel 5

            // Avoid nesting too much
            if ((row >= h) || (col >= w))
            {
                return;
            }

            uint16_t bitset = 0;

            // First, check the pixels of the block
            // No check on the bottom-right pixel as it's never responsible for connections between blocks
            if (img[index]) // top-left
            {
                bitset |= BITMASK_3x3;
            }

            if ((row <= h) && img[index + 1]) // top-right, checking we don't overflow
            {
                bitset |= BITMASK_3x3 << 1;
            }

            if ((col <= w) && img[index + w]) // bottom-left, checking we don't overflow
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
            if (hasBit(bitset, 0) && img[index - w - 1])
            {
                // Merge block X with block P (top-left)
                computeUnion(labels, index, index - 2 * w - 2); // above, left
            }

            // Check if we have pixels in the bottom of the top block
            // i.e. are pixels 1 or 2 connected to block X
            if ((hasBit(bitset, 1) && img[index - w]) || (hasBit(bitset, 2) && img[index - w + 1]))
            {
                // Merge block X with block Q
                computeUnion(labels, index, index - 2 * w); // above
            }

            // Check if we have a top-right pixel and a top-right diagonal pixel
            // i.e. is pixel 3 connected to block X
            if (hasBit(bitset, 3) && img[index - w + 2])
            {
                // Merge block X with block R
                computeUnion(labels, index, index - 2 * w + 2); // above, right
            }

            // Check if we have pixels in the right of the left block
            // i.e. are pixels 4 or 8 connected to block X
            if ((hasBit(bitset, 4) && img[index - 1]) || (hasBit(bitset, 8) && img[index + w - 1]))
            {
                // Merge block X with block S
                computeUnion(labels, index, index - 2); // left
            }
        }

        __global__ void compress(int32_t *const labels, const uint32_t w, const uint32_t h)
        {
            const uint32_t row = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
            const uint32_t col = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
            const uint32_t index = row * w + col;

            if ((row < h) && (col < w))
            {
                findAndCompress(labels, index);
            }
        }

        __global__ void finalize(const uint8_t *const img, int32_t *const labels, const uint32_t w, const uint32_t h)
        {
            const uint32_t row = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
            const uint32_t col = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
            const uint32_t index = row * w + col;

            // Early stop
            if ((row >= h) || (col >= w))
            {
                return;
            }

            uint32_t label = labels[index] + 1;

            // Pseudo-algorithm of the paper uses multiplication to avoid checks
            // but I suppose that checks are more efficient
            // Apply the labels of the different blocks uniformaly
            // Accounting for edges
            labels[index] = img[index] ? label : 0;

            if (col + 1 < w)
            {
                labels[index + 1] = img[index + 1] ? label : 0;

                if (row + 1 < h)
                {
                    labels[index + w + 1] = img[index + w + 1] ? label : 0;
                }
            }

            if (row + 1 < h)
            {
                labels[index + w] = img[index + w] ? label : 0;
            }
        }

        // NOTE : Perhaps we can fuse kernels ?
    }
}
