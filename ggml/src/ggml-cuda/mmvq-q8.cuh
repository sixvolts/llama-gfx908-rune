#pragma once

#include "common.cuh"

// Dense Q8_0 GEMV for 1..4 columns on CDNA (wave64): bit-identical to mul_mat_vec_q<Q8_0, ncols> (same lane
// partition, accumulation order and reductions) with the K loop unrolled at compile time, every load issued before any
// arithmetic and 8-byte weight loads. Returns false (caller falls back) for unsupported shapes. GGML_MMVQ_Q8_V2=0 disables.
bool ggml_cuda_mmvq_q8_v2(
        const void * vx, const void * vy, float * dst, int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int nchannels_x, int nchannels_y, int nchannels_dst, int nsamples_x, int nsamples_dst, cudaStream_t stream);

// Short rows (K = 128 / 256 / 512), 1..8 columns, optional channels (batched matmul): several rows per wave, bit-identical
// to mul_mat_vec_q<Q8_0> (see mmvq-q8.cu). Returns false for other shapes. GGML_MMVQ_Q8_SMALLK=0 disables.
bool ggml_cuda_mmvq_q8_smallk(
        const void * vx, const void * vy, float * dst, int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int nchannels_x, int nchannels_y, int nchannels_dst,
        int stride_channel_x, int stride_channel_y, int stride_channel_dst, cudaStream_t stream);

// fuse-moe C: shared-expert gate + up + SWIGLU_CLAMP at 2..8 columns in one mmvq_q8_0_v2 launch (vx = up, vgate = gate),
// bit-identical to the two GEMVs + the GLU kernel. _supported: whether the call below will take the shape.
bool ggml_cuda_mmvq_q8_v2_glu_supported(int ncols_x, int ncols_dst);
bool ggml_cuda_mmvq_q8_v2_glu(
        const void * vx, const void * vgate, const void * vy, float * dst, float glu_limit, int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst, cudaStream_t stream);
// rune fuse-dense (GGML_CUDA_FD_KDA_BETA): arm a sigmoid epilogue for the next ggml_cuda_mmvq_q8_v2 call (K = 4096);
// pending() is true afterwards when that call did not take it (the caller then applies the sigmoid itself)
void ggml_cuda_mmvq_q8_set_sigmoid(bool on);
bool ggml_cuda_mmvq_q8_sigmoid_pending();
