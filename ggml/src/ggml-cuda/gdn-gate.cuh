#pragma once

#include "common.cuh"

// Fused GDN gate chain (qwen4exp build_delta_net, decode-size batches, F32 weights):
//   gate = softplus(W_alpha . x + dt) * a      W_alpha: [K, na], x: [K, nt], dt/a: [na]
//   beta = sigmoid(W_beta . x)                 W_beta : [K, nb]
// One launch of na + nb blocks replaces mul_mat_vec_f x2, add, softplus, mul and sigmoid. Each block replays
// mul_mat_vec_f<float, float, nt, 128> for its row exactly (same lane partition, unrolled load order, warp and
// cross-warp reduction) and applies the unfused elementwise device math, so the outputs are bit-identical.
// Returns false (caller runs the unfused nodes) for shapes mul_mat_vec_f would not run with a 128-thread block.
bool ggml_cuda_op_gdn_gate_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_alpha, const ggml_tensor * w_beta, const ggml_tensor * x,
        const ggml_tensor * dt, const ggml_tensor * a, ggml_tensor * gate, ggml_tensor * beta);
