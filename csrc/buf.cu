// Copyright (c) 2024 Bendabir.
#include <buf.h>
#include <cuda.h>

// TODO : Adapt this to switch to BKE which is SotA (thus a bit more efficient)

// Use constants so it's typed (instead of using text substitution)
constexpr uint16_t BITMASK_3x3 = 0x0777;
constexpr uint16_t BITMASK_3x4R = 0xEEEE; // all but col 0
constexpr uint16_t BITMASK_2x4L = 0x3333; // only col O and 1
constexpr uint16_t BITMASK_3x4L = 0x7777; // all but col 3
constexpr uint16_t BITMASK_4x3B = 0xFFF0; // all but row 0
constexpr uint16_t BITMASK_4x2T = 0x00FF; // only row O and 1
constexpr uint16_t BITMASK_4x3T = 0x0FFF; // all but row 3

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
//
// As all images in a batch are independant, it's pretty straight forward to work on batches on the Z dimension

namespace buf
{
    namespace ccl2d
    {
        __device__ __forceinline__ uint8_t hasBit(const uint16_t mask, const uint8_t pos)
        {
            return (mask >> pos) & 1;
        }

        __device__ uint32_t find(const int32_t *const g_labels, const uint32_t index)
        {
            uint32_t n = index;

            while (g_labels[n] != n)
            {
                n = g_labels[n];
            }

            return n;
        }

        __device__ uint32_t findAndCompress(int32_t *const g_labels, const uint32_t index)
        {
            uint32_t n = index;

            while (g_labels[n] != n)
            {
                n = g_labels[n];
                g_labels[index] = n;
            }

            return n;
        }

        __device__ void computeUnion(int32_t *const g_labels, uint32_t a, uint32_t b)
        {
            bool done = false;

            do
            {
                a = find(g_labels, a);
                b = find(g_labels, b);

                if (a < b)
                {
                    const int32_t old = atomicMin(g_labels + b, a);
                    done = (old == b);
                    b = old;
                }
                else if (b < a)
                {
                    const int32_t old = atomicMin(g_labels + a, b);
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
        // Block labels live in a compact buffer [N, bh, bw] with bh = ceil(h / 2) and bw = ceil(w / 2),
        // one int per 2x2 block, so label reads and writes are coalesced.
        // Each block takes its own block index (in its image) as initial label.
        __global__ void init(int32_t *__restrict__ const g_batch_blocks, const uint32_t bw, const uint32_t bh, const uint32_t n)
        {
            const uint32_t brow = blockIdx.y * blockDim.y + threadIdx.y;
            const uint32_t bcol = blockIdx.x * blockDim.x + threadIdx.x;
            const uint32_t depth = blockIdx.z;

            if ((brow >= bh) || (bcol >= bw) || (depth >= n))
            {
                return;
            }

            const uint32_t block = brow * bw + bcol;

            g_batch_blocks[static_cast<int64_t>(depth) * bh * bw + block] = block;
        }

        __global__ void merge(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n)
        {
            const uint32_t bw = (w + 1) / 2;
            const uint32_t bh = (h + 1) / 2;
            const uint32_t brow = blockIdx.y * blockDim.y + threadIdx.y;
            const uint32_t bcol = blockIdx.x * blockDim.x + threadIdx.x;
            const uint32_t depth = blockIdx.z;

            if ((brow >= bh) || (bcol >= bw) || (depth >= n))
            {
                return;
            }

            const uint32_t row = 2 * brow;
            const uint32_t col = 2 * bcol;
            const uint32_t block = brow * bw + bcol;
            // Top-left pixel of the block, basically pixel 5 (see paper). Neighbours are at +/- 1 and +/- sh.
            const uint8_t *const g_img = g_batch_img + depth * sn + row * sh + col;
            int32_t *const g_blocks = g_batch_blocks + static_cast<int64_t>(depth) * bh * bw;

            // NOTE : As we're only using bit 0, 1, 2, 3, 4 and 8, this could technically fits on a byte
            uint16_t mask = 0;

            // NOTE : This might be suboptimal.
            //        Perhaps we should copy all the image data we need in the thread at once (if possible).
            //        So we work either from the registers or shared memory.

            // First, check the pixels of the block, so we build a "pixels-to-check" mask for foreground pixels.
            // No check on the bottom-right pixel as it's never responsible for connections between blocks.
            if (g_img[0]) // top-left pixel
            {
                mask |= BITMASK_3x3;
            }

            if ((col + 1 < w) && g_img[1]) // top-right pixel
            {
                mask |= BITMASK_3x3 << 1;
            }

            if ((row + 1 < h) && g_img[sh]) // bottom-left pixel
            {
                mask |= BITMASK_3x3 << 4;
            }

            // Check the different edge cases
            if (col == 0) // no left-pixel, droping 1st column of the mask
            {
                mask &= BITMASK_3x4R;
            }

            // no right-pixel, but blocks are 2-pixels wide, dropping the 2 last columns of the mask
            // because we don't need to check them as they don't exist
            if (col + 1 >= w)
            {
                mask &= BITMASK_2x4L;
            }
            else if (col + 2 >= w) // no column on the right side of the block (2 apart from the index)
            {
                mask &= BITMASK_3x4L;
            }

            if (row == 0) // likewise, no top-pixel
            {
                mask &= BITMASK_4x3B;
            }

            // same for the bottom-pixel, it doesn't exist, so we don't need to check it nor the bottom of the block
            if (row + 1 >= h)
            {
                mask &= BITMASK_4x2T;
            }
            else if (row + 2 >= h) // no row on the bottom side of the block (2 apart from the index)
            {
                mask &= BITMASK_4x3T;
            }

            // We can now check for neighbour blocks
            if (!mask)
            {
                return;
            }

            // If we have a top-left pixel (5) and a top-left pixel (0)
            // i.e. is pixel 0 connected to block X
            if (hasBit(mask, 0) && g_img[-sh - 1])
            {
                // Merge block X with block P (top-left)
                computeUnion(g_blocks, block, block - bw - 1); // above, left
            }

            // Check if we have pixels in the bottom of the top block
            // i.e. are pixels 1 or 2 connected to block X
            if ((hasBit(mask, 1) && g_img[-sh]) || (hasBit(mask, 2) && g_img[-sh + 1]))
            {
                // Merge block X with block Q
                computeUnion(g_blocks, block, block - bw); // above
            }

            // Check if we have a top-right pixel and a top-right diagonal pixel
            // i.e. is pixel 3 connected to block X
            if (hasBit(mask, 3) && g_img[-sh + 2])
            {
                // Merge block X with block R
                computeUnion(g_blocks, block, block - bw + 1); // above, right
            }

            // Check if we have pixels in the right of the left block
            // i.e. are pixels 4 or 8 connected to block X
            if ((hasBit(mask, 4) && g_img[-1]) || (hasBit(mask, 8) && g_img[sh - 1]))
            {
                // Merge block X with block S
                computeUnion(g_blocks, block, block - 1); // left
            }
        }

        // One thread per 2x2 block. Each warp still writes 64 consecutive ints per pixel row (L2 merges them).
        // Measured on sm_86 : about 1.2x the copy bound (x.to(int32)), which is the floor here as the block
        // buffer is read too. One thread per pixel was 5-15% slower (4x more threads, 1-byte loads).
        __global__ void finalize(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_blocks,
            int32_t *__restrict__ const g_batch_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n)
        {
            const uint32_t bw = (w + 1) / 2;
            const uint32_t bh = (h + 1) / 2;
            const uint32_t brow = blockIdx.y * blockDim.y + threadIdx.y;
            const uint32_t bcol = blockIdx.x * blockDim.x + threadIdx.x;
            const uint32_t depth = blockIdx.z;

            if ((brow >= bh) || (bcol >= bw) || (depth >= n))
            {
                return;
            }

            const uint32_t row = 2 * brow;
            const uint32_t col = 2 * bcol;
            const uint32_t index = row * w + col;
            const uint8_t *const g_img = g_batch_img + depth * sn + row * sh + col;
            int32_t *const g_labels = g_batch_labels + static_cast<int64_t>(depth) * h * w;
            // Fused compress : the forest is final after merge, so each thread finds its root here.
            const uint32_t label = findAndCompress(g_batch_blocks + static_cast<int64_t>(depth) * bh * bw, brow * bw + bcol) + 1;

            // Pseudo-algorithm of the paper uses multiplication to avoid checks
            // but I suppose that checks are more efficient
            // Apply the labels of the different blocks uniformaly
            // Accounting for edges
            g_labels[index] = g_img[0] ? label : 0; // top-left

            if (col + 1 < w)
            {
                g_labels[index + 1] = g_img[1] ? label : 0; // top-right
            }

            if (row + 1 < h)
            {
                g_labels[index + w] = g_img[sh] ? label : 0; // bottom-left
            }

            if ((col + 1 < w) && (row + 1 < h))
            {
                g_labels[index + w + 1] = g_img[sh + 1] ? label : 0; // bottom-right
            }
        }
    }
}
