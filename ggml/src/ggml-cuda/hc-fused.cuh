#pragma once

#include "common.cuh"

// Fused HyperConnection combine (qwen4exp build_hc_combine):
//   w   = scale2(sigmoid(scale1(inject)))          inject   : [hc, nt]
//   dst = residual + repeat(block_out) * w         block_out: [ne0, nt], residual/dst: [ne0, hc, nt]
// Replays the scale -> sigmoid -> scale -> mul -> add device math in the original order,
// so the result is bit-identical to the unfused graph.
void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx,
        const ggml_tensor * residual, const ggml_tensor * block_out, const ggml_tensor * inject,
        float scale1, float bias1, float scale2, float bias2, ggml_tensor * dst);

// Fused HyperConnection mix epilogue (qwen4exp build_hc_mix, hc == 4):
//   gate  = sigmoid(up)                       up, xn : [hc*ne0, nt]
//   gated = xn * gate
//   dst   = scale( ((g0 + g1) + g2) + g3 )    g_c = gated[c*ne0 .. ], dst: [ne0, nt]
// Same products and the same left-fold the unfused cont + fused_add path uses, so bit-identical.
void ggml_cuda_op_hc_mix_epilogue(ggml_backend_cuda_context & ctx,
        const ggml_tensor * xn, const ggml_tensor * up, int64_t hc,
        float scale, float bias, ggml_tensor * dst);

// Fused scale -> silu: dst = silu(scale * x + bias)
void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx,
        const ggml_tensor * x, float scale, float bias, ggml_tensor * dst);

// Fused HyperConnection low-rank path + epilogue (decode, nt == 1, hc == 4, Q8_0 weights):
//   lo  = silu(scale_lo * (w_down . xn) + bias_lo)
//   dst = scale_out * mean_c( xn_c * sigmoid((w_up . lo)_c) ) + bias_out
// Two launches replacing quantize/mmvq/scale/silu/quantize/mmvq/sigmoid/mul/cont/add x3/scale;
// bit-identical to that path (replays mmvq's lane partition and reductions).
void ggml_cuda_op_hc_mix_mega(ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_down, const ggml_tensor * w_up, const ggml_tensor * xn,
        float scale_lo, float bias_lo, float scale_out, float bias_out, ggml_tensor * dst);
