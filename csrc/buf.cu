// Copyright (c) 2024 Bendabir.
#include <buf.h>
#include <uf.h>
#include <cuda.h>

// TODO : Adapt this to switch to BKE which is SotA (thus a bit more efficient) ? More complex to implement though.
namespace buf
{
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
    namespace ccl2d
    {
        // Use constants so it's typed (instead of using text substitution)
        constexpr uint16_t BITMASK_3x3 = 0x0777;
        constexpr uint16_t BITMASK_3x4R = 0xEEEE; // all but col 0
        constexpr uint16_t BITMASK_2x4L = 0x3333; // only col O and 1
        constexpr uint16_t BITMASK_3x4L = 0x7777; // all but col 3
        constexpr uint16_t BITMASK_4x3B = 0xFFF0; // all but row 0
        constexpr uint16_t BITMASK_4x2T = 0x00FF; // only row O and 1
        constexpr uint16_t BITMASK_4x3T = 0x0FFF; // all but row 3

        // Bit pos of mask (0 or 1).
        __device__ __forceinline__ uint8_t hasBit(const uint16_t mask, const uint8_t pos)
        {
            return (mask >> pos) & 1;
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
            // Tiles are flattened on grid x (gridDim.y is limited to 65535)
            const uint32_t tiles_x = (bw + BUF_2D_BLOCK_COLS - 1) / BUF_2D_BLOCK_COLS;
            const uint32_t tile_x = blockIdx.x % tiles_x;
            const uint32_t tile_y = blockIdx.x / tiles_x;
            const uint32_t tx = threadIdx.x;
            const uint32_t ty = threadIdx.y;
            const uint32_t local = ty * BUF_2D_BLOCK_COLS + tx;
            const uint32_t brow = tile_y * BUF_2D_BLOCK_ROWS + ty;
            const uint32_t bcol = tile_x * BUF_2D_BLOCK_COLS + tx;
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
                uf::computeUnion<0>(s_labels, local, local - BUF_2D_BLOCK_COLS - 1);
            }

            if ((link & LINK_Q) && (ty > 0))
            {
                uf::computeUnion<0>(s_labels, local, local - BUF_2D_BLOCK_COLS);
            }

            if ((link & LINK_R) && (ty > 0) && (tx + 1 < BUF_2D_BLOCK_COLS))
            {
                uf::computeUnion<0>(s_labels, local, local - BUF_2D_BLOCK_COLS + 1);
            }

            if ((link & LINK_S) && (tx > 0))
            {
                uf::computeUnion<0>(s_labels, local, local - 1);
            }

            __syncthreads();

            if (!inside)
            {
                return;
            }

            // Local and global raster orders match, so the root index is never above the block index
            const uint32_t root = uf::find<0>(s_labels, local);
            const uint32_t root_row = tile_y * BUF_2D_BLOCK_ROWS + root / BUF_2D_BLOCK_COLS;
            const uint32_t root_col = tile_x * BUF_2D_BLOCK_COLS + root % BUF_2D_BLOCK_COLS;

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
            // Groups of tiles are flattened on grid x (gridDim.y is limited to 65535)
            const uint32_t tiles_x = (bw + BUF_2D_BLOCK_COLS - 1) / BUF_2D_BLOCK_COLS;
            const uint32_t groups_x = (tiles_x + BUF_2D_BORDER_TILES - 1) / BUF_2D_BORDER_TILES;
            const uint32_t group_x = blockIdx.x % groups_x;
            const uint32_t tile_y = blockIdx.x / groups_x;
            const uint32_t brow = tile_y * BUF_2D_BLOCK_ROWS + ty;
            const uint32_t bcol = (group_x * BUF_2D_BORDER_TILES + threadIdx.y) * BUF_2D_BLOCK_COLS + tx;
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
                uf::computeUnion<0>(g_blocks, block, block - bw - 1);
            }

            if ((link & LINK_Q) && (ty == 0))
            {
                uf::computeUnion<0>(g_blocks, block, block - bw);
            }

            if ((link & LINK_R) && ((ty == 0) || (tx + 1 == BUF_2D_BLOCK_COLS)))
            {
                uf::computeUnion<0>(g_blocks, block, block - bw + 1);
            }

            if ((link & LINK_S) && (tx == 0))
            {
                uf::computeUnion<0>(g_blocks, block, block - 1);
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
            // Tiles are flattened on grid x (gridDim.y is limited to 65535)
            const uint32_t tiles_x = (bw + blockDim.x - 1) / blockDim.x;
            const uint32_t brow = (blockIdx.x / tiles_x) * blockDim.y + threadIdx.y;
            const uint32_t bcol = (blockIdx.x % tiles_x) * blockDim.x + threadIdx.x;
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
            const uint32_t label = uf::findAndCompress<0>(g_batch_blocks + static_cast<int64_t>(depth) * bh * bw, brow * bw + bcol) + 1;

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

    namespace ccl3d
    {
        // Blocks are 2x2x2 voxels. In 26-connectivity, all foreground voxels of a block share a label.
        // Axes : plane p on H, row r on W, column c on D (contiguous).
        // The bitset covers the 4x4x4 voxels around a block X, bit = pz * 16 + py * 4 + px,
        // with offsets -1..2 from the block origin on each axis :
        //     pz=0 (p-1)     pz=1 (p)          pz=2 (p+1)        pz=3 (p+2)
        //      0  1  2  3    16 17 18 19       32 33 34 35       never used
        //      4  5  6  7    20 [21][22] 23    36 [37][38] 39
        //      8  9 10 11    24 [25][26] 27    40 [41][42] 43
        //     12 13 14 15    28 29 30 31       44 45 46 47
        // LSB is bit 0.
        constexpr uint64_t BITMASK_3x3x3 = 0x0000'0777'0777'0777; // neighbours of the voxel at offset (0, 0, 0)
        constexpr uint64_t BITMASK_X_NO_LEFT = 0xEEEE'EEEE'EEEE'EEEE; // all but px = 0
        constexpr uint64_t BITMASK_X_2 = 0x3333'3333'3333'3333;       // only px = 0 and 1
        constexpr uint64_t BITMASK_X_3 = 0x7777'7777'7777'7777;       // all but px = 3
        constexpr uint64_t BITMASK_Y_NO_TOP = 0xFFF0'FFF0'FFF0'FFF0;  // all but py = 0
        constexpr uint64_t BITMASK_Y_2 = 0x00FF'00FF'00FF'00FF;       // only py = 0 and 1
        constexpr uint64_t BITMASK_Y_3 = 0x0FFF'0FFF'0FFF'0FFF;       // all but py = 3
        constexpr uint64_t BITMASK_Z_NO_PREV = 0xFFFF'FFFF'FFFF'0000; // all but pz = 0
        constexpr uint64_t BITMASK_Z_2 = 0x0000'0000'FFFF'FFFF;       // only pz = 0 and 1

        // The 13 neighbour blocks before X in raster order. k is also the raster index in the 3x3x3
        // block neighbourhood, so the offset is (k / 9 - 1, k / 3 % 3 - 1, k % 3 - 1) :
        //     plane p-1 (dz=-1)    plane p (dz=0)
        //      0  1  2              9 10 11
        //      3  4  5             12  X
        //      6  7  8
        constexpr uint32_t NEIGHBOURS = 13;
        // __device__ : device code cannot index a host constexpr array with a runtime index
        __device__ constexpr uint64_t BITS[NEIGHBOURS] = {
            0x0000'0000'0000'0001, // 0 : 0
            0x0000'0000'0000'0006, // 1 : 1, 2
            0x0000'0000'0000'0008, // 2 : 3
            0x0000'0000'0000'0110, // 3 : 4, 8
            0x0000'0000'0000'0660, // 4 : 5, 6, 9, 10
            0x0000'0000'0000'0880, // 5 : 7, 11
            0x0000'0000'0000'1000, // 6 : 12
            0x0000'0000'0000'6000, // 7 : 13, 14
            0x0000'0000'0000'8000, // 8 : 15
            0x0000'0001'0001'0000, // 9 : 16, 32
            0x0000'0006'0006'0000, // 10 : 17, 18, 33, 34
            0x0000'0008'0008'0000, // 11 : 19, 35
            0x0000'0110'0110'0000, // 12 : 20, 24, 36, 40
        };

        __device__ __forceinline__ int32_t offsetZ(const uint32_t k) { return static_cast<int32_t>(k / 9) - 1; }
        __device__ __forceinline__ int32_t offsetY(const uint32_t k) { return static_cast<int32_t>(k / 3 % 3) - 1; }
        __device__ __forceinline__ int32_t offsetX(const uint32_t k) { return static_cast<int32_t>(k % 3) - 1; }

        // Bits of the voxels of neighbour k. Each external voxel belongs to exactly one neighbour.
        __device__ __forceinline__ uint64_t neighbourBits(const uint32_t k)
        {
            return BITS[k];
        }

        // Which neighbour blocks (bit k) are connected to the block whose origin voxel is g_img.
        __device__ __forceinline__ uint16_t links(
            const uint8_t *const g_img,
            const int64_t sh,
            const int64_t sw,
            const uint32_t p,
            const uint32_t r,
            const uint32_t c,
            const uint32_t d,
            const uint32_t w,
            const uint32_t h)
        {
            const bool has_col = c + 1 < d;
            const bool has_row = r + 1 < w;
            const bool has_plane = p + 1 < h;
            uint64_t mask = 0;

            // Voxels of the block. The voxel at (1, 1, 1) only touches blocks after X, so it is skipped.
            if (g_img[0])
            {
                mask |= BITMASK_3x3x3;
            }

            if (has_col && g_img[1])
            {
                mask |= BITMASK_3x3x3 << 1;
            }

            if (has_row && g_img[sw])
            {
                mask |= BITMASK_3x3x3 << 4;
            }

            if (has_row && has_col && g_img[sw + 1])
            {
                mask |= BITMASK_3x3x3 << 5;
            }

            if (has_plane && g_img[sh])
            {
                mask |= BITMASK_3x3x3 << 16;
            }

            if (has_plane && has_col && g_img[sh + 1])
            {
                mask |= BITMASK_3x3x3 << 17;
            }

            if (has_plane && has_row && g_img[sh + sw])
            {
                mask |= BITMASK_3x3x3 << 20;
            }

            // Drop the voxels outside the volume
            if (c == 0)
            {
                mask &= BITMASK_X_NO_LEFT;
            }

            if (!has_col)
            {
                mask &= BITMASK_X_2;
            }
            else if (c + 2 >= d)
            {
                mask &= BITMASK_X_3;
            }

            if (r == 0)
            {
                mask &= BITMASK_Y_NO_TOP;
            }

            if (!has_row)
            {
                mask &= BITMASK_Y_2;
            }
            else if (r + 2 >= w)
            {
                mask &= BITMASK_Y_3;
            }

            if (p == 0)
            {
                mask &= BITMASK_Z_NO_PREV;
            }

            // No mask for p + 2 >= h : plane pz = 3 is never read
            if (!has_plane)
            {
                mask &= BITMASK_Z_2;
            }

            uint16_t result = 0;

            if (!mask)
            {
                return result;
            }

#pragma unroll
            for (uint32_t k = 0; k < NEIGHBOURS; ++k)
            {
                uint64_t bits = mask & neighbourBits(k);

                // One foreground voxel is enough
                while (bits)
                {
                    const int32_t b = __ffsll(static_cast<long long>(bits)) - 1;
                    const int64_t offset = (b / 16 - 1) * sh + (b / 4 % 4 - 1) * sw + (b % 4 - 1);

                    if (g_img[offset])
                    {
                        result |= 1 << k;
                        break;
                    }

                    bits &= bits - 1;
                }
            }

            return result;
        }

        // Block coordinates (x on columns, y on rows, z on planes) of the thread.
        // Tiles of BUF_3D_BLOCK_X x BUF_3D_BLOCK_Y blocks are flattened on grid x
        // (gridDim.y and gridDim.z are limited to 65535).
        __device__ __forceinline__ uint3 blockCoords(const uint32_t bd, const uint32_t bw)
        {
            const uint32_t tiles_x = (bd + BUF_3D_BLOCK_X - 1) / BUF_3D_BLOCK_X;
            const uint32_t tiles_y = (bw + BUF_3D_BLOCK_Y - 1) / BUF_3D_BLOCK_Y;

            return make_uint3(
                (blockIdx.x % tiles_x) * BUF_3D_BLOCK_X + threadIdx.x,
                (blockIdx.x / tiles_x % tiles_y) * BUF_3D_BLOCK_Y + threadIdx.y,
                blockIdx.x / tiles_x / tiles_y);
        }

        // --- KERNELS ---
        // Block labels live in a compact buffer [N, bh, bw, bd] (ceil of half sizes), one int per 2x2x2 block.
        // One thread per block. Unlike 2D, there is no union-find in shared memory first : links() is the
        // main cost, and a local merge in shared memory needs it twice (once per merge kernel).
        // Measured on sm_86 : the global merge alone is 15-25% faster than a local merge in tiles of
        // 16x4x4 or 32x4x2 blocks followed by a merge of the tile borders.

        // Each block is its own root.
        __global__ void init(
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t d,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n)
        {
            const uint32_t bd = (d + 1) / 2;
            const uint32_t bw = (w + 1) / 2;
            const uint32_t bh = (h + 1) / 2;
            const uint3 block = blockCoords(bd, bw);
            const uint32_t depth = blockIdx.z;

            if ((block.x >= bd) || (block.y >= bw) || (block.z >= bh) || (depth >= n))
            {
                return;
            }

            const uint32_t index = (block.z * bw + block.y) * bd + block.x;

            g_batch_blocks[static_cast<int64_t>(depth) * bh * bw * bd + index] = index;
        }

        // Global union-find with the 13 neighbours before each block.
        __global__ void merge(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            const int64_t sw,
            int32_t *__restrict__ const g_batch_blocks,
            const uint32_t d,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n)
        {
            const uint32_t bd = (d + 1) / 2;
            const uint32_t bw = (w + 1) / 2;
            const uint32_t bh = (h + 1) / 2;
            const uint3 block = blockCoords(bd, bw);
            const uint32_t depth = blockIdx.z;

            if ((block.x >= bd) || (block.y >= bw) || (block.z >= bh) || (depth >= n))
            {
                return;
            }

            const uint32_t p = 2 * block.z;
            const uint32_t r = 2 * block.y;
            const uint32_t c = 2 * block.x;
            const uint16_t link = links(g_batch_img + depth * sn + p * sh + r * sw + c, sh, sw, p, r, c, d, w, h);
            const uint32_t index = (block.z * bw + block.y) * bd + block.x;
            int32_t *const g_blocks = g_batch_blocks + static_cast<int64_t>(depth) * bh * bw * bd;

#pragma unroll
            for (uint32_t k = 0; k < NEIGHBOURS; ++k)
            {
                // links drops the neighbours outside the volume, so the neighbour index is valid
                if ((link >> k) & 1)
                {
                    const int64_t offset = (offsetZ(k) * static_cast<int64_t>(bw) + offsetY(k)) * bd + offsetX(k);
                    uf::computeUnion<0>(g_blocks, index, static_cast<uint32_t>(index + offset));
                }
            }
        }

        // One thread per 2x2x2 block : find the root, then write the labels of up to 8 voxels.
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
            const uint32_t n)
        {
            const uint32_t bd = (d + 1) / 2;
            const uint32_t bw = (w + 1) / 2;
            const uint32_t bh = (h + 1) / 2;
            const uint3 block = blockCoords(bd, bw);
            const uint32_t depth = blockIdx.z;

            if ((block.x >= bd) || (block.y >= bw) || (block.z >= bh) || (depth >= n))
            {
                return;
            }

            const uint32_t p = 2 * block.z;
            const uint32_t r = 2 * block.y;
            const uint32_t c = 2 * block.x;
            const uint8_t *const g_img = g_batch_img + depth * sn + p * sh + r * sw + c;
            int32_t *const g_labels = g_batch_labels + static_cast<int64_t>(depth) * h * w * d + (static_cast<int64_t>(p) * w + r) * d + c;
            // Fused compress : the forest is final after merge, so each thread finds its root here
            const int32_t label = uf::findAndCompress<0>(
                g_batch_blocks + static_cast<int64_t>(depth) * bh * bw * bd,
                (block.z * bw + block.y) * bd + block.x) + 1;

#pragma unroll
            for (uint32_t dz = 0; dz < 2; ++dz)
            {
#pragma unroll
                for (uint32_t dy = 0; dy < 2; ++dy)
                {
#pragma unroll
                    for (uint32_t dx = 0; dx < 2; ++dx)
                    {
                        if ((p + dz < h) && (r + dy < w) && (c + dx < d))
                        {
                            g_labels[(static_cast<int64_t>(dz) * w + dy) * d + dx] = g_img[dz * sh + dy * sw + dx] ? label : 0;
                        }
                    }
                }
            }
        }
    }
}
