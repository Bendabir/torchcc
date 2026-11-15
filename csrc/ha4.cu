// Copyright (c) 2026 Bendabir.
#include <ha4.h>
#include <uf.h>
#include <cuda.h>

// HA4 (Hennequin et al., DASIP 2018) : 4-connectivity CCL on runs of pixels ("segments").
// The image is split in strips of HA4_2D_BLOCK_ROWS rows, one thread block per strip,
// one warp per row. Each warp loops over its row in chunks of 32 pixels (one pixel per lane).
//
//     +----------------------------+
//     | strip 0  rows 0..3         |   warp ty handles row y = strip * 4 + ty
//     +============================+ <-- border row (y = 4) : borderMerge
//     | strip 1  rows 4..7         |
//     +============================+ <-- border row (y = 8)
//     | strip 2  ...               |
//     +----------------------------+
//
// The lanes share their pixels with a ballot, so each lane knows where its run starts :
//
//     lane tx          0 1 2 3 4 5 6 7 ...
//     pixel            0 1 1 1 0 0 1 0
//     start distance   - 0 1 2 - - 0 -      (0 = run start in this chunk)
//
// Union-find lives in the output labels L (int32, [N, H, W]). Labels are 1-based raster indices :
// the node at raster index k has label k + 1 and L[k] holds its parent label (root : L[k] == k + 1,
// background : 0). The final label is then the root label itself, so relabel can write it in place
// (it is a valid path compression for a concurrent find).
//
// Nodes are the run starts and the foreground pixels of lane 0 (they store the label of their true
// run start, which can be in a previous chunk). Other pixels are only written by relabel.
//
// As all images in a batch are independent, each image gets its own blockIdx.z.

namespace ha4
{
    namespace ccl2d
    {
        // Lane mask of the warp intrinsics (ballot, shfl) : all 32 lanes take part.
        // So every lane of the warp must reach each intrinsic call, also lanes outside the image.
        constexpr uint32_t FULL_MASK = 0xFFFFFFFF;

        // Number of set pixels on the left of lane tx in its run (0 for a run start).
        // Lane 0 always gets 0 (a shift by 32 is undefined behaviour).
        __device__ __forceinline__ uint32_t startDistance(const uint32_t pixels, const uint32_t tx)
        {
            return tx == 0 ? 0 : __clz(~(pixels << (32 - tx)));
        }

        // Distance to its run start for the lane 0 of the next chunk : length of the run that touches lane 31,
        // added to the previous distance when the whole chunk is set.
        __device__ __forceinline__ uint32_t nextDistance(const uint32_t pixels, const uint32_t distance)
        {
            const uint32_t d = __clz(~pixels);

            return d == HA4_2D_BLOCK_COLS ? distance + d : d;
        }

        // --- KERNELS ---

        // Label each strip on its own. Each warp inits the nodes of its row, then merges its runs
        // with the runs of the row above (same strip, exchanged as bitmasks in shared memory).
        // The run of lane 0 can come from a previous chunk, so each row carries its distance to the start.
        //
        //     row y-1   . 1 1 1 1 . .      union(start(y), start(y-1)) only where one of the two
        //     row y     . . . 1 1 1 .      pixels is a run start : one union per pair of runs
        __global__ void stripLabeling(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n)
        {
            // Bitmask of each row for the current chunk. Warps cannot read the registers of other warps,
            // so warp ty publishes its row here for the warp of row y + 1.
            // Double buffer : one barrier per chunk, so a warp can write chunk i + 1
            // while the warp below still reads chunk i. A buffer is written again only at chunk i + 2,
            // after the barrier of chunk i + 1, which no warp passes before it has read chunk i.
            __shared__ uint32_t s_pixels[2][HA4_2D_BLOCK_ROWS];

            const uint32_t tx = threadIdx.x;
            const uint32_t ty = threadIdx.y;
            const uint32_t y = blockIdx.x * HA4_2D_BLOCK_ROWS + ty;
            const uint32_t depth = blockIdx.z;
            // No early return before the barriers : rows outside the image only have background pixels
            const bool inside = (y < h) && (depth < n);
            // Rows past h get the image start : they never read it, but a pointer past the end is UB
            const uint8_t *const g_img = inside ? g_batch_img + depth * sn + y * sh : g_batch_img;
            int32_t *const g_labels = g_batch_labels + static_cast<int64_t>(depth) * h * w;
            const uint32_t line = y * w;
            // Distance from lane 0 of the current chunk to the start of its run, for row y and row y - 1.
            // A run can span several chunks, so the distance is carried from chunk to chunk.
            // Only lane 0 uses it (the other lanes find their run start in the chunk).
            uint32_t distance_y = 0;
            uint32_t distance_y1 = 0;
            uint32_t parity = 0;

            // Loop bound is the same for all lanes, so they all stay active for the ballot and the barrier
            // NOTE : What about unrolling the loop over i to improve instruction-level parallelism (if possible) ?
            for (uint32_t i = 0; i < w; i += HA4_2D_BLOCK_COLS)
            {
                // Lane tx reads pixel x of row y (32 consecutive bytes per warp). k is its raster index.
                // The ballot gives the 32 pixels of the chunk to all lanes : bit tx = pixel of lane tx.
                const uint32_t x = i + tx;
                const bool p = inside && (x < w) && g_img[x];
                const uint32_t pixels = __ballot_sync(FULL_MASK, p);
                uint32_t s_y = startDistance(pixels, tx);
                const uint32_t k = line + x;

                if (p && (s_y == 0))
                {
                    g_labels[k] = k - (tx == 0 ? distance_y : 0) + 1;
                }

                if (tx == 0)
                {
                    s_pixels[parity][ty] = pixels;
                }

                // Row y - 1 is written by another warp : wait until all warps have published their bitmask
                __syncthreads();

                const uint32_t pixels_y1 = ty > 0 ? s_pixels[parity][ty - 1] : 0;
                const bool p_y1 = (pixels_y1 >> tx) & 1;
                uint32_t s_y1 = startDistance(pixels_y1, tx);

                if (tx == 0)
                {
                    s_y = distance_y;
                    s_y1 = distance_y1;
                }

                if (p && p_y1 && ((s_y == 0) || (s_y1 == 0)))
                {
                    uf::computeUnion<1>(g_labels, k - s_y + 1, k - w - s_y1 + 1);
                }

                distance_y = nextDistance(pixels, distance_y);
                distance_y1 = nextDistance(pixels_y1, distance_y1);
                parity ^= 1;
            }
        }

        // Merge the runs across the strip borders (first row of each strip but the first one).
        // One warp per (border row, chunk of 32 pixels) : warp ty of block (group, chunk) handles
        // the border row of strip group * 4 + ty. Lane 0 counts as a run start : it is a node,
        // and the union of a run that comes from a previous chunk is just redundant.
        __global__ void borderMerge(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n)
        {
            const uint32_t tx = threadIdx.x;
            const uint32_t chunks = (w + HA4_2D_BLOCK_COLS - 1) / HA4_2D_BLOCK_COLS;
            const uint32_t strip = (blockIdx.x / chunks) * HA4_2D_BLOCK_ROWS + threadIdx.y;
            const uint32_t x = (blockIdx.x % chunks) * HA4_2D_BLOCK_COLS + tx;
            const uint32_t y = strip * HA4_2D_BLOCK_ROWS;
            const uint32_t depth = blockIdx.z;

            // Same for the whole warp (no barrier in this kernel)
            if ((y == 0) || (y >= h) || (depth >= n))
            {
                return;
            }

            const uint8_t *const g_img = g_batch_img + depth * sn + y * sh;
            int32_t *const g_labels = g_batch_labels + static_cast<int64_t>(depth) * h * w;
            const bool p = (x < w) && g_img[x];
            const bool p_y1 = (x < w) && g_img[static_cast<int64_t>(x) - sh];
            const uint32_t s_y = startDistance(__ballot_sync(FULL_MASK, p), tx);
            const uint32_t s_y1 = startDistance(__ballot_sync(FULL_MASK, p_y1), tx);
            const uint32_t k = y * w + x;

            if (p && p_y1 && ((s_y == 0) || (s_y1 == 0)))
            {
                uf::computeUnion<1>(g_labels, k - s_y + 1, k - w - s_y1 + 1);
            }
        }

        // One warp per (row, chunk of 32 pixels). Each node finds its root, its run gets it with a shuffle,
        // then every lane writes its pixel (background included), so 32 consecutive int32 per warp.
        __global__ void relabel(
            const uint8_t *__restrict__ const g_batch_img,
            const int64_t sn,
            const int64_t sh,
            int32_t *__restrict__ const g_batch_labels,
            const uint32_t w,
            const uint32_t h,
            const uint32_t n)
        {
            const uint32_t tx = threadIdx.x;
            const uint32_t chunks = (w + HA4_2D_BLOCK_COLS - 1) / HA4_2D_BLOCK_COLS;
            const uint32_t y = (blockIdx.x / chunks) * HA4_2D_BLOCK_ROWS + threadIdx.y;
            const uint32_t x = (blockIdx.x % chunks) * HA4_2D_BLOCK_COLS + tx;
            const uint32_t depth = blockIdx.z;

            // Same for the whole warp (no barrier in this kernel)
            if ((y >= h) || (depth >= n))
            {
                return;
            }

            const uint8_t *const g_img = g_batch_img + depth * sn + y * sh;
            int32_t *const g_labels = g_batch_labels + static_cast<int64_t>(depth) * h * w;
            const bool p = (x < w) && g_img[x];
            const uint32_t s = startDistance(__ballot_sync(FULL_MASK, p), tx);
            const uint32_t k = y * w + x;
            uint32_t root = 0;

            if (p && (s == 0))
            {
                root = uf::find<1>(g_labels, k + 1);
            }

            // Labels are 1-based, so the root label is the final label
            const uint32_t label = __shfl_sync(FULL_MASK, root, tx - s);

            if (x < w)
            {
                g_labels[k] = p ? label : 0;
            }
        }
    }
}
