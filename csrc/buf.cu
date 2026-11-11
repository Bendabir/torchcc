// Copyright (c) 2024 Bendabir.
#include <buf.h>
#include <cuda.h>

// TODO : Adapt this to switch to BKE which is SotA (thus a bit more efficient)

// Use constants so it's typed (instead of using text substitution)
// TODO : Move these masks to buf::ccl2d for consistency ?
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

        constexpr uint8_t LINK_P = 1; // top-left block
        constexpr uint8_t LINK_Q = 2; // top block
        constexpr uint8_t LINK_R = 4; // top-right block
        constexpr uint8_t LINK_S = 8; // left block

        // Which neighbour blocks (P, Q, R, S) are connected to the block whose top-left pixel is g_img.
        __device__ __forceinline__ uint8_t links(
            const uint8_t *const g_img,
            const int64_t sh,
            const uint32_t row,
            const uint32_t col,
            const uint32_t w,
            const uint32_t h)
        {
            // NOTE : As we're only using bit 0, 1, 2, 3, 4 and 8, this could technically fits on a byte
            uint16_t mask = 0;

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

            uint8_t result = 0;

            if (!mask)
            {
                return result;
            }

            // Pixel 0 connects block P
            if (hasBit(mask, 0) && g_img[-sh - 1])
            {
                result |= LINK_P;
            }

            // Pixels 1 or 2 connect block Q
            if ((hasBit(mask, 1) && g_img[-sh]) || (hasBit(mask, 2) && g_img[-sh + 1]))
            {
                result |= LINK_Q;
            }

            // Pixel 3 connects block R
            if (hasBit(mask, 3) && g_img[-sh + 2])
            {
                result |= LINK_R;
            }

            // Pixels 4 or 8 connect block S
            if ((hasBit(mask, 4) && g_img[-1]) || (hasBit(mask, 8) && g_img[sh - 1]))
            {
                result |= LINK_S;
            }

            return result;
        }

        // --- KERNELS ---
        // Block labels live in a compact buffer [N, bh, bw] with bh = ceil(h / 2) and bw = ceil(w / 2),
        // one int per 2x2 block, so label reads and writes are coalesced.
        // Each thread block is a tile of BUF_2D_BLOCK_COLS x BUF_2D_BLOCK_ROWS blocks.

        // Init and union-find inside the tile, in shared memory (most links never leave a tile), so it's pretty fast.
        // Each block then gets the global index of its tile root, so every tile is a flat forest.
        __global__ void localMerge(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n)
        {
            __shared__ int32_t s_labels[BUF_2D_BLOCK_COLS * BUF_2D_BLOCK_ROWS];

            const uint32_t bw = (w + 1) / 2;
            const uint32_t bh = (h + 1) / 2;
            const uint32_t tx = threadIdx.x;
            const uint32_t ty = threadIdx.y;
            const uint32_t local = ty * BUF_2D_BLOCK_COLS + tx;
            const uint32_t brow = blockIdx.y * BUF_2D_BLOCK_ROWS + ty;
            const uint32_t bcol = blockIdx.x * BUF_2D_BLOCK_COLS + tx;
            const uint32_t depth = blockIdx.z;
            // No early return before the barriers : threads outside the image only keep their own label
            const bool inside = (brow < bh) && (bcol < bw) && (depth < n);
            const uint32_t row = 2 * brow;
            const uint32_t col = 2 * bcol;

            s_labels[local] = local;

            __syncthreads();

            const uint8_t link = inside ? links(g_batch_img + depth * sn + row * sh + col, sh, row, col, w, h) : 0;

            if ((link & LINK_P) && (tx > 0) && (ty > 0))
            {
                computeUnion(s_labels, local, local - BUF_2D_BLOCK_COLS - 1);
            }

            if ((link & LINK_Q) && (ty > 0))
            {
                computeUnion(s_labels, local, local - BUF_2D_BLOCK_COLS);
            }

            if ((link & LINK_R) && (ty > 0) && (tx + 1 < BUF_2D_BLOCK_COLS))
            {
                computeUnion(s_labels, local, local - BUF_2D_BLOCK_COLS + 1);
            }

            if ((link & LINK_S) && (tx > 0))
            {
                computeUnion(s_labels, local, local - 1);
            }

            __syncthreads();

            if (!inside)
            {
                return;
            }

            // Local and global raster orders match, so the root index is never above the block index
            const uint32_t root = find(s_labels, local);
            const uint32_t root_row = blockIdx.y * BUF_2D_BLOCK_ROWS + root / BUF_2D_BLOCK_COLS;
            const uint32_t root_col = blockIdx.x * BUF_2D_BLOCK_COLS + root % BUF_2D_BLOCK_COLS;

            g_batch_blocks[static_cast<int64_t>(depth) * bh * bw + brow * bw + bcol] = root_row * bw + root_col;
        }

        // Global union-find, only for the links that cross a tile border (localMerge did the others).
        // One thread per border block only (interior blocks have all their neighbours P, Q, R, S
        // in the same tile). threadIdx.y picks one of BUF_2D_BORDER_TILES tiles in a row, threadIdx.x is t :
        //     t = 0 .. 31      top row (P, Q, R in the tile above)
        //     t = 32 .. 38     left column (P, S in the tile on the left)
        //     t = 39 .. 45     right column (R in the tile above right)
        __global__ void borderMerge(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n)
        {
            const uint32_t t = threadIdx.x;
            uint32_t tx = 0;
            uint32_t ty = 0;

            if (t < BUF_2D_BLOCK_COLS)
            {
                tx = t;
            }
            else if (t < BUF_2D_BLOCK_COLS + BUF_2D_BLOCK_ROWS - 1)
            {
                ty = t - BUF_2D_BLOCK_COLS + 1;
            }
            else if (t < BUF_2D_BORDER_THREADS)
            {
                tx = BUF_2D_BLOCK_COLS - 1;
                ty = t - BUF_2D_BLOCK_COLS - BUF_2D_BLOCK_ROWS + 2;
            }
            else
            {
                return;
            }

            const uint32_t bw = (w + 1) / 2;
            const uint32_t bh = (h + 1) / 2;
            const uint32_t brow = blockIdx.y * BUF_2D_BLOCK_ROWS + ty;
            const uint32_t bcol = (blockIdx.x * BUF_2D_BORDER_TILES + threadIdx.y) * BUF_2D_BLOCK_COLS + tx;
            const uint32_t depth = blockIdx.z;

            if ((brow >= bh) || (bcol >= bw) || (depth >= n))
            {
                return;
            }

            const uint32_t row = 2 * brow;
            const uint32_t col = 2 * bcol;
            const uint32_t block = brow * bw + bcol;
            int32_t *const g_blocks = g_batch_blocks + static_cast<int64_t>(depth) * bh * bw;
            const uint8_t link = links(g_batch_img + depth * sn + row * sh + col, sh, row, col, w, h);

            if ((link & LINK_P) && ((tx == 0) || (ty == 0)))
            {
                computeUnion(g_blocks, block, block - bw - 1);
            }

            if ((link & LINK_Q) && (ty == 0))
            {
                computeUnion(g_blocks, block, block - bw);
            }

            if ((link & LINK_R) && ((ty == 0) || (tx + 1 == BUF_2D_BLOCK_COLS)))
            {
                computeUnion(g_blocks, block, block - bw + 1);
            }

            if ((link & LINK_S) && (tx == 0))
            {
                computeUnion(g_blocks, block, block - 1);
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
