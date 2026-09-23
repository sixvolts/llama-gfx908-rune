#pragma once

#include "common.cuh"

// One row of mul_mat_vec_f<float, float, ncols_dst, 128, false, false> (mmvf.cu), replayed in a 128-thread block
// whose linear thread index is `tid`: same lane partition (col2 = tid, tid + 128, ...), same MMVF_UNROLL = 8 load and
// accumulation order, same warp and cross-warp reduction (buf_iw of warp_size floats, zeroed by the caller's block
// before the first __syncthreads). Leaves column j's full sum in sumf[j] of every thread with tid < warp_size, so the
// caller writes sumf[tid] for tid < ncols_dst exactly as mmvf does. Contraction stays at the default (mmvf compiles
// acc += v*u into an fma; so does this).
template <int ncols_dst>
static __device__ __forceinline__ void ggml_cuda_mmvf_replay_row_128(
        const float2 * __restrict__ x2, const float2 * __restrict__ y2, const int ncols2, const int stride_col_y2,
        float * buf_iw, const int tid, float (&sumf)[ncols_dst]) {
    constexpr int block_size = 128;
    constexpr int warp_size  = 64;
    constexpr int MMVF_UNROLL = 8;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = 0.0f;
    }
    int col2 = tid;
    for (; col2 + (MMVF_UNROLL-1)*block_size < ncols2; col2 += MMVF_UNROLL*block_size) {
        float2 tmpx[MMVF_UNROLL];
#pragma unroll
        for (int u = 0; u < MMVF_UNROLL; ++u) {
            tmpx[u] = x2[col2 + u*block_size];
        }
#pragma unroll
        for (int u = 0; u < MMVF_UNROLL; ++u) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2 + u*block_size];
                ggml_cuda_mad(sumf[j], tmpx[u].x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx[u].y, tmpy.y);
            }
        }
    }
    for (; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float2 tmpy = y2[j*stride_col_y2 + col2];
            ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
        }
    }
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
        buf_iw[tid/warp_size] = sumf[j];
        __syncthreads();
        if (tid < warp_size) {
            sumf[j] = buf_iw[tid];
            sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
        }
        if (j < ncols_dst) {
            __syncthreads();
        }
    }
}

// the block size mul_mat_vec_f picks on CDNA (at most 128 threads) for a row of `ncols` elements
static inline int ggml_cuda_mmvf_block_size_cdna(const int64_t ncols) {
    const int64_t warp_size = 64;
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    for (int64_t block_size = 2*warp_size; block_size <= 128; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    return (int) block_size_best;
}
