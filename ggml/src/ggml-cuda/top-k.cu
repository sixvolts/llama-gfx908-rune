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
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
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

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
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

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
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

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
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
        v[e]  = ok ? x[c] : -INFINITY;
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
        v[e]  = ok ? cand_v[row*n + c] : -INFINITY;
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
        if (threadIdx.x == 0) { dst[row*k + r] = bi; }
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
        if (!top_k_small_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream)) {
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
