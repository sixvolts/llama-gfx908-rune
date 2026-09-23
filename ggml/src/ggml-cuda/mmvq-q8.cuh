#pragma once

#include "common.cuh"

// Dense Q8_0 GEMV for 1..4 columns on CDNA (wave64): bit-identical to mul_mat_vec_q<Q8_0, ncols> (same lane
// partition, accumulation order and reductions) with the K loop unrolled at compile time, every load issued before any
// arithmetic and 8-byte weight loads. Returns false (caller falls back) for unsupported shapes. GGML_MMVQ_Q8_V2=0 disables.
bool ggml_cuda_mmvq_q8_v2(
        const void * vx, const void * vy, float * dst, int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int nchannels_x, int nchannels_y, int nchannels_dst, int nsamples_x, int nsamples_dst, cudaStream_t stream);
