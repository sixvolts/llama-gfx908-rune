#pragma once

#include "common.cuh"

// Few-token MUL_MAT_ID (MoE decode / MTP verify, 1..4 tokens) for CDNA with expert deduplication: every distinct expert
// is read once for all tokens routed to it, and each weight block is unpacked once for all of them. Tolerance class
// (block-wise summation order, see mmvq-moe.cu). Returns false (caller falls back to mmvq) for unsupported types,
// shapes or fusions. GGML_MOE_V2=0 disables it.
bool ggml_cuda_mmvq_moe_dedup(
        ggml_type type, const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int ncols_x, int nrows_x, int nchannels_y,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int stride_channel_x, int stride_channel_y, int stride_channel_dst,
        int ncols_dst, int n_used, int ids_stride, cudaStream_t stream);

// Whether ggml_cuda_mmvq_moe_dedup can fuse a dense shared expert (type_shared, same K) into the routed gate/up/GLU
// launch for this shape (GLM-5.3-Flash: Q4_K routed, Q8_0 shared, K = 4096, 1..8 tokens). GGML_MOE_SHEXP=0 disables.
bool ggml_cuda_mmvq_moe_shexp_supported(ggml_type type, ggml_type type_shared, int ncols_x, int ncols_dst, int n_used,
        ggml_glu_op glu_op, ggml_glu_op glu_op_shared);
