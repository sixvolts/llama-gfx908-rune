#include "argsort.cuh"
#include "top-k.cuh"

#include <cstdlib>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

// Picks the digit of the k-th key: the highest bin whose count of keys at or above it (inclusive suffix sum) reaches
// the remaining rank, bin 0 if none; the new rank subtracts the keys strictly above that bin. The serial version (one
// thread walking the bins down from the top) left select at 11-18 us per launch in decode (4 per top-k call, 14 calls
// per verify step on GLM-5.3-Flash). The scan version computes the same integers with wave suffix scans and an integer
// max (identical state, so identical selections). GGML_CUDA_TOPK_SCAN=0 keeps the serial walk.
template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift,
        bool scan) {
    constexpr int NBINS = 1 << RADIX_BITS;
    static_assert(BLOCK_SIZE == NBINS, "one thread per bin");

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }

    if (scan) {
        constexpr int warp_size = ggml_cuda_get_physical_warp_size();
        constexpr int nwarps    = NBINS / warp_size;
        __shared__ int warp_total[nwarps];
        __shared__ int chosen;
        const int lane = tid % warp_size;
        const int warp = tid / warp_size;

        // inclusive suffix sum within the wave (sum over this lane and the lanes above it)
        int sfx = count;
#pragma unroll
        for (int off = 1; off < warp_size; off <<= 1) {
            const int v = __shfl_down(sfx, off, warp_size);
            if (lane + off < warp_size) {
                sfx += v;
            }
        }
        if (lane == 0) {
            warp_total[warp] = sfx;
        }
        if (tid == 0) {
            chosen = 0;
        }
        __syncthreads();
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            if (w > warp) {
                sfx += warp_total[w];
            }
        }
        const top_k_radix_state state0 = states[row];
        if (tid > 0 && sfx >= state0.rank) {
            atomicMax(&chosen, tid);   // integer max: order-independent
        }
        __syncthreads();
        if (tid == chosen) {
            top_k_radix_state state = state0;
            state.rank -= sfx - count;  // keys strictly above the chosen bin
            state.prefix |= (uint32_t) tid << shift;
            state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
            states[row] = state;
        }
        return;
    }

    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

// The entries above the k-th key, placed deterministically: an atomic slot counter wrote them in arrival order, so
// the selected SET was reproducible but its ORDER was not, and every consumer that accumulates over the list (the
// DSA sparse attention / indexer on GLM-5.3-Flash) rounded differently on every run (KL 0.0094 between identical
// perplexity runs). Pass 1 counts each block's entries; pass 2 gives each block the sum of the earlier blocks'
// counts as its base and writes its entries in its own column order (warp ballots, as top_k_gather_equal).
template<int BLOCK_SIZE>
static __global__ void top_k_radix_count_greater(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_counts,
        int ncols,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    const uint32_t prefix = states[row].prefix;
    __shared__ int total;

    if (tid == 0) {
        total = 0;
    }
    __syncthreads();

    int count = 0;
    for (int col = row_block * BLOCK_SIZE + tid; col < ncols; col += blocks_per_row * BLOCK_SIZE) {
        count += top_k_float_to_ordered(row_src[col]) > prefix;
    }
    atomicAdd(&total, count);   // integer: order-independent
    __syncthreads();

    if (tid == 0) {
        block_counts[blockIdx.x] = total;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        const int * __restrict__ block_counts,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp = tid / warpSize;
    const int nwarps = BLOCK_SIZE / warpSize;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    const uint32_t prefix = states[row].prefix;
    __shared__ int warp_counts[32];

    int count = 0;
    for (int b = 0; b < row_block; ++b) {
        count += block_counts[row * blocks_per_row + b];
    }

    for (int base = row_block * BLOCK_SIZE; base < ncols; base += blocks_per_row * BLOCK_SIZE) {
        const int col = base + tid;
        const bool greater = col < ncols && top_k_float_to_ordered(row_src[col]) > prefix;
        const unsigned long long mask = __ballot(greater);
        if (lane == 0) {
            warp_counts[warp] = __popcll(mask);
        }
        __syncthreads();
        int before = count;
        for (int w = 0; w < nwarps; ++w) {
            if (w < warp) {
                before += warp_counts[w];
            }
            count += warp_counts[w];
        }
        const unsigned long long lane_mask = (1ULL << lane) - 1;
        if (greater) {
            row_dst[before + __popcll(mask & lane_mask)] = col;
        }
        __syncthreads();
    }
}

// halo-hybrid (after pwilkin/llama.cpp 408ea2e1a): the entries EQUAL to the k-th key, taken in column order.
// The previous gather appended them by atomic arrival order, so when the k-th key is tied (GLM-5.3-Flash's
// indexer scores whole runs of pools at exactly 0 after the ReLU) the selected set differed from run to run
// and from the CPU's first-index choice; the DSA selection was not reproducible across runs.
template<int BLOCK_SIZE>
static __device__ void top_k_gather_equal(
        const float * src, int * dst, int ncols, uint32_t threshold, int limit, int offset) {
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp = tid / warpSize;
    const int nwarps = BLOCK_SIZE / warpSize;
    __shared__ int warp_counts[32];
    int count = 0;

    for (int base = 0; base < ncols && count < limit; base += BLOCK_SIZE) {
        const int col = base + tid;
        const bool equal = col < ncols && top_k_float_to_ordered(src[col]) == threshold;
        const unsigned long long mask = __ballot(equal);
        if (lane == 0) {
            warp_counts[warp] = __popcll(mask);
        }
        __syncthreads();
        int before = count;
        for (int w = 0; w < nwarps; ++w) {
            if (w < warp) {
                before += warp_counts[w];
            }
            count += warp_counts[w];
        }
        const unsigned long long lane_mask = (1ULL << lane) - 1;
        const int pos = before + __popcll(mask & lane_mask);
        if (equal && pos < limit) {
            dst[offset + pos] = col;
        }
        __syncthreads();
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather_equal(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        int ncols,
        int k) {
    const int row = blockIdx.x;
    const top_k_radix_state state = states[row];
    top_k_gather_equal<BLOCK_SIZE>(src + (size_t) row * ncols, dst + (size_t) row * k,
                                   ncols, state.prefix, state.rank, k - state.rank);
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    static const bool scan = [] { const char * e = getenv("GGML_CUDA_TOPK_SCAN"); return e == nullptr || atoi(e) != 0; }();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift, scan);
    }

    ggml_cuda_pool_alloc<int> counts_alloc(pool, (size_t) nrows * blocks_per_row);
    top_k_radix_count_greater<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(src, states, counts_alloc.get(), ncols, blocks_per_row);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, counts_alloc.get(), ncols, k, blocks_per_row);
    top_k_radix_gather_equal<BLOCK_SIZE>
        <<<nrows, BLOCK_SIZE, 0, stream>>>(src, dst, states, ncols, k);
}

// Fused few-row top-k (decode on GLM-5.3-Flash: 1..18 rows of up to ~33k pool scores, k = 512): one 1024-thread
// workgroup per row and one launch instead of top_k_radix_cuda's 12 (whose launch floor alone is ~54 us per call).
// BIT-EXACT with top_k_radix_cuda: the k-th key and the tie rank come from the same exact 8-bit radix select (the digit
// is picked with the same integer suffix scan as top_k_radix_select), the entries above the k-th key are written in
// the radix gather's order (blocks_per_row = min(ceil(ncols/1024), 64) strided 256-column chunks, block-major, then
// (j, t) order inside a block) and the first `rank` ties in column order. Histogram increments are aggregated per
// wave and distinct bin (ReLU'd scores fall into a handful of first-digit bins); every count is an integer, so the
// result does not depend on atomic order. The row is re-read from L2 in each phase (<= 128 KB, just written by the
// indexer) instead of being kept in registers. GGML_CUDA_TOPK_FUSED=0 keeps the radix path.
#define TOP_K_FUSED_NT        1024
#define TOP_K_FUSED_MAX_COLS  32832    // blocks_per_row <= 33 -> <= 132 chunks
#define TOP_K_FUSED_MAX_ROWS  32
#define TOP_K_FUSED_MAX_CHUNK 136

static __global__ void __launch_bounds__(TOP_K_FUSED_NT, 1) top_k_fused_kernel(
        const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = TOP_K_FUSED_NT / warp_size;
    constexpr int sel_warps = 256 / warp_size;               // the waves that hold the 256 histogram bins
    constexpr int grp_warps = 256 / warp_size;               // waves per 256-column group
    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;
    const unsigned long long lt_mask = (1ULL << lane) - 1;   // lanes below this one
    const float * row_src = src + (size_t) blockIdx.x * ncols;
    int         * row_dst = dst + (size_t) blockIdx.x * k;

    __shared__ int      hist[256];
    __shared__ int      warp_tot[nwarps];
    __shared__ int      s_chosen;
    __shared__ int      s_rank;
    __shared__ uint32_t s_prefix;
    __shared__ int      chunk_base[TOP_K_FUSED_MAX_CHUNK];
    __shared__ int      chunk_wcnt[TOP_K_FUSED_MAX_CHUNK][grp_warps];

    // 1. radix select of the k-th key, 8 bits per pass from the top
    uint32_t prefix = 0;
    uint32_t pmask  = 0;
    int      rank   = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) {
            hist[tid] = 0;
        }
        __syncthreads();
        for (int c0 = 0; c0 < ncols; c0 += TOP_K_FUSED_NT) {
            const int col = c0 + tid;
            bool valid = false;
            int  bin   = 0;
            if (col < ncols) {
                const uint32_t key = top_k_float_to_ordered(row_src[col]);
                valid = (key & pmask) == prefix;
                bin   = (key >> shift) & 255;
            }
            unsigned long long act = __ballot(valid);
            while (act) {                                        // one LDS add per (wave, distinct bin)
                const int leader = __ffsll((unsigned long long) act) - 1;
                const int b      = __shfl(bin, leader, warp_size);
                const unsigned long long same = __ballot(valid && bin == b);
                if (lane == leader) {
                    atomicAdd(&hist[b], (int) __popcll(same));
                }
                if (bin == b) {
                    valid = false;
                }
                act &= ~same;
            }
        }
        __syncthreads();

        // the digit: highest bin whose inclusive suffix count reaches rank (0 if none), as in top_k_radix_select
        const int count = tid < 256 ? hist[tid] : 0;
        int sfx = count;
#pragma unroll
        for (int off = 1; off < warp_size; off <<= 1) {
            const int v = __shfl_down(sfx, off, warp_size);
            if (lane + off < warp_size) {
                sfx += v;
            }
        }
        if (lane == 0 && warp < sel_warps) {
            warp_tot[warp] = sfx;
        }
        if (tid == 0) {
            s_chosen = 0;
        }
        __syncthreads();
        if (tid < 256) {
            for (int w = warp + 1; w < sel_warps; ++w) {
                sfx += warp_tot[w];
            }
            if (tid > 0 && sfx >= rank) {
                atomicMax(&s_chosen, tid);
            }
        }
        __syncthreads();
        if (tid == s_chosen) {
            s_rank   = rank - (sfx - count);
            s_prefix = prefix | ((uint32_t) tid << shift);
        }
        __syncthreads();
        prefix = s_prefix;
        rank   = s_rank;
        pmask |= 255u << shift;
    }
    const uint32_t thr = prefix;   // the k-th key; `rank` of its ties are taken

    // 2. entries above thr: per (strided chunk, wave) counts, chunk c = b*J + j covers columns b*256 + j*bpr*256 + t
    const int bpr     = min((ncols + 1023) / 1024, 64);
    const int J       = (ncols + bpr*256 - 1) / (bpr*256);
    const int nchunks = bpr*J;
    const int grp     = tid / 256;
    const int t       = tid % 256;
    const int gw      = t / warp_size;
    for (int c = grp; c < nchunks; c += TOP_K_FUSED_NT/256) {
        const int b   = c / J;
        const int j   = c % J;
        const int col = b*256 + j*bpr*256 + t;
        const bool gt = col < ncols && top_k_float_to_ordered(row_src[col]) > thr;
        const unsigned long long m = __ballot(gt);
        if (lane == 0) {
            chunk_wcnt[c][gw] = (int) __popcll(m);
        }
    }
    __syncthreads();
    if (warp == 0) {                                     // exclusive scan of the chunk totals in chunk order
        constexpr int per_lane = (TOP_K_FUSED_MAX_CHUNK + warp_size - 1) / warp_size;
        int v[per_lane];
        int s = 0;
#pragma unroll
        for (int u = 0; u < per_lane; ++u) {
            const int c = lane*per_lane + u;
            int tot = 0;
            if (c < nchunks) {
#pragma unroll
                for (int w = 0; w < grp_warps; ++w) {
                    tot += chunk_wcnt[c][w];
                }
            }
            v[u] = tot;
            s += tot;
        }
        int incl = s;                                    // inclusive prefix over lanes
#pragma unroll
        for (int off = 1; off < warp_size; off <<= 1) {
            const int x = __shfl_up(incl, off, warp_size);
            if (lane >= off) {
                incl += x;
            }
        }
        int run = incl - s;
#pragma unroll
        for (int u = 0; u < per_lane; ++u) {
            const int c = lane*per_lane + u;
            if (c < nchunks) {
                chunk_base[c] = run;
            }
            run += v[u];
        }
    }
    __syncthreads();
    for (int c = grp; c < nchunks; c += TOP_K_FUSED_NT/256) {
        const int b   = c / J;
        const int j   = c % J;
        const int col = b*256 + j*bpr*256 + t;
        const bool gt = col < ncols && top_k_float_to_ordered(row_src[col]) > thr;
        const unsigned long long m = __ballot(gt);
        if (gt) {
            int pos = chunk_base[c];
            for (int w = 0; w < gw; ++w) {
                pos += chunk_wcnt[c][w];
            }
            row_dst[pos + (int) __popcll(m & lt_mask)] = col;
        }
    }

    // 3. the first `rank` ties in column order, written after the k - rank greater entries
    int found = 0;
    for (int c0 = 0; c0 < ncols && found < rank; c0 += TOP_K_FUSED_NT) {
        const int col = c0 + tid;
        const bool eq = col < ncols && top_k_float_to_ordered(row_src[col]) == thr;
        const unsigned long long m = __ballot(eq);
        __syncthreads();                                 // previous iteration's warp_tot reads are done
        if (lane == 0) {
            warp_tot[warp] = (int) __popcll(m);
        }
        __syncthreads();
        int before = found;
        int total  = 0;
        for (int w = 0; w < nwarps; ++w) {
            const int x = warp_tot[w];
            before += w < warp ? x : 0;
            total  += x;
        }
        const int pos = before + (int) __popcll(m & lt_mask);
        if (eq && pos < rank) {
            row_dst[k - rank + pos] = col;
        }
        found += total;
    }
}

static bool top_k_fused_cuda(const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_TOPK_FUSED"); return e == nullptr || atoi(e) != 0; }();
    if (!on || ncols <= 1024 || ncols > TOP_K_FUSED_MAX_COLS || nrows > TOP_K_FUSED_MAX_ROWS || k > ncols) {
        return false;
    }
    top_k_fused_kernel<<<nrows, TOP_K_FUSED_NT, 0, stream>>>(src, dst, ncols, k);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
// Small k (<= TOP_K_SMALL_MAX) over long rows: two kernels instead of the radix select's 1 + 2*4 + 2 launches.
// Kernel 1: each of NBLK blocks per row keeps the k largest of its slice by k rounds of a block-wide argmax (ties: the
// lowest index), kernel 2 merges the NBLK*k candidates the same way. Exactly the k largest values; among equal values
// the lowest indices, which the radix select's gather does not guarantee (a set difference only on exact ties).
#define TOP_K_SMALL_MAX 32
#define TOP_K_SMALL_NBLK 64

static __device__ __forceinline__ void top_k_small_argmax_block(float & v, int & i, float * s_v, int * s_i) {
    // wave-level max with lowest index on ties, then across the 4 waves of a 256-thread block; result in every thread
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, v, off, 64);
        const int   oi = __shfl_xor_sync(0xffffffff, i, off, 64);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if ((threadIdx.x & 63) == 0) { s_v[threadIdx.x >> 6] = v; s_i[threadIdx.x >> 6] = i; }
    __syncthreads();
    v = s_v[0]; i = s_i[0];
#pragma unroll
    for (int w = 1; w < 4; ++w) {
        if (s_v[w] > v || (s_v[w] == v && s_i[w] < i)) { v = s_v[w]; i = s_i[w]; }
    }
    __syncthreads();
}

template <int EPT>   // elements per thread held in registers
static __global__ void __launch_bounds__(256, 1) top_k_small_partial(
        const float * __restrict__ src, float * __restrict__ cand_v, int * __restrict__ cand_i, const int ncols, const int k) {
    const int row = blockIdx.x / TOP_K_SMALL_NBLK;
    const int blk = blockIdx.x % TOP_K_SMALL_NBLK;
    const float * x = src + (size_t) row*ncols;
    const int slice = (ncols + TOP_K_SMALL_NBLK - 1) / TOP_K_SMALL_NBLK;
    const int c0 = blk*slice;
    float v[EPT]; int ix[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int c = c0 + e*256 + threadIdx.x;
        const bool ok = c < c0 + slice && c < ncols;
        const float xv = ok ? x[c] : -INFINITY;
        v[e]  = xv == xv ? xv : -INFINITY;   // NaN would break the argmax ordering (the radix path ranks it as +inf)
        ix[e] = ok ? c : 0x7fffffff;
    }
    __shared__ float s_v[4]; __shared__ int s_i[4];
    for (int r = 0; r < k; ++r) {
        float bv = v[0]; int bi = ix[0];
#pragma unroll
        for (int e = 1; e < EPT; ++e) {
            if (v[e] > bv || (v[e] == bv && ix[e] < bi)) { bv = v[e]; bi = ix[e]; }
        }
        top_k_small_argmax_block(bv, bi, s_v, s_i);
        if (threadIdx.x == 0) { cand_v[(row*TOP_K_SMALL_NBLK + blk)*k + r] = bv; cand_i[(row*TOP_K_SMALL_NBLK + blk)*k + r] = bi; }
#pragma unroll
        for (int e = 0; e < EPT; ++e) {
            if (ix[e] == bi) { v[e] = -INFINITY; ix[e] = 0x7fffffff; }
        }
    }
}

static __global__ void __launch_bounds__(256, 1) top_k_small_merge(
        const float * __restrict__ cand_v, const int * __restrict__ cand_i, int * __restrict__ dst, const int k) {
    const int row = blockIdx.x;
    const int n = TOP_K_SMALL_NBLK*k;   // <= 2048
    constexpr int EPT = (TOP_K_SMALL_NBLK*TOP_K_SMALL_MAX + 255)/256;   // 8
    float v[EPT]; int ix[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int c = e*256 + threadIdx.x;
        const bool ok = c < n;
        const float cv = ok ? cand_v[row*n + c] : -INFINITY;
        v[e]  = cv == cv ? cv : -INFINITY;
        ix[e] = ok ? cand_i[row*n + c] : 0x7fffffff;
    }
    __shared__ float s_v[4]; __shared__ int s_i[4];
    for (int r = 0; r < k; ++r) {
        float bv = v[0]; int bi = ix[0];
#pragma unroll
        for (int e = 1; e < EPT; ++e) {
            if (v[e] > bv || (v[e] == bv && ix[e] < bi)) { bv = v[e]; bi = ix[e]; }
        }
        top_k_small_argmax_block(bv, bi, s_v, s_i);
        if (threadIdx.x == 0) { dst[row*k + r] = bi == 0x7fffffff ? 0 : bi; }   // never emit the sentinel as an index
#pragma unroll
        for (int e = 0; e < EPT; ++e) {
            if (ix[e] == bi) { v[e] = -INFINITY; ix[e] = 0x7fffffff; }
        }
    }
}

static bool top_k_small_cuda(ggml_cuda_pool & pool, const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    static const bool enabled = !getenv("GGML_CUDA_TOPK_SMALL") || atoi(getenv("GGML_CUDA_TOPK_SMALL")) != 0;
    if (!enabled || k > TOP_K_SMALL_MAX || nrows > 64) {
        return false;
    }
    const int slice = (ncols + TOP_K_SMALL_NBLK - 1) / TOP_K_SMALL_NBLK;
    const int ept = (slice + 255) / 256;
    if (ept > 16) {
        return false;
    }
    ggml_cuda_pool_alloc<float> cv(pool, (size_t) nrows*TOP_K_SMALL_NBLK*k);
    ggml_cuda_pool_alloc<int>   ci(pool, (size_t) nrows*TOP_K_SMALL_NBLK*k);
    const dim3 grid(nrows*TOP_K_SMALL_NBLK);
    switch (ept) {   // rows up to 64*256*EPT columns
        case 1:  top_k_small_partial< 1><<<grid, 256, 0, stream>>>(src, cv.get(), ci.get(), ncols, k); break;
        case 2:  top_k_small_partial< 2><<<grid, 256, 0, stream>>>(src, cv.get(), ci.get(), ncols, k); break;
        case 3:  top_k_small_partial< 3><<<grid, 256, 0, stream>>>(src, cv.get(), ci.get(), ncols, k); break;
        case 4:  top_k_small_partial< 4><<<grid, 256, 0, stream>>>(src, cv.get(), ci.get(), ncols, k); break;
        case 5: case 6: top_k_small_partial< 6><<<grid, 256, 0, stream>>>(src, cv.get(), ci.get(), ncols, k); break;
        case 7: case 8: top_k_small_partial< 8><<<grid, 256, 0, stream>>>(src, cv.get(), ci.get(), ncols, k); break;
        case 9: case 10: case 11: case 12: top_k_small_partial<12><<<grid, 256, 0, stream>>>(src, cv.get(), ci.get(), ncols, k); break;
        case 13: case 14: case 15: case 16: top_k_small_partial<16><<<grid, 256, 0, stream>>>(src, cv.get(), ci.get(), ncols, k); break;
        default: return false;
    }
    top_k_small_merge<<<nrows, 256, 0, stream>>>(cv.get(), ci.get(), dst, k);
    return true;
}
#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        if (!top_k_small_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream) &&
            !top_k_fused_cuda(src0_d, dst_d, ncols, nrows, k, stream)) {
            top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        }
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
