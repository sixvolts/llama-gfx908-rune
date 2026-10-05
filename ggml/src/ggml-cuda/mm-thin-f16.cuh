#pragma once

#include "common.cuh"

// gfx908 prefill: dense Q8_0 x F32 projections with few output rows (GLM-5.3: hc_*_fn M=24, ssm_f_a/g_a and indexer
// k / compressor gate M=128, ssm_beta M=64, kv_a M=512) or a short K (ssm_f_b/g_b K=128). See mm-thin-f16.cu.
bool ggml_cuda_mm_thin_f16_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mm_thin_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
