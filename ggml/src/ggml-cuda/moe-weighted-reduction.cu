#include "moe-weighted-reduction.cuh"

template <bool has_scale, int NE>
static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used_rt) {
    // explicit operation order (bit-exact with the loop form): with the loop fully unrolled the compiler otherwise
    // contracts e0*w0 + e1*w1 the other way round (fma(w0, e0, e1*w1) instead of fma(e1*s1, w1, e0*w0))
#pragma clang fp contract(off)
    const int n_expert_used = NE > 0 ? NE : n_expert_used_rt; // NE: compile-time expert count (fully unrolled)
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = has_scale ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    // unrolled so the experts' loads are issued together (gfx908: one global round trip per expert otherwise);
    // the sum order is unchanged (bit-exact)
#pragma unroll
    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = has_scale ? expert_scale[row] : 1.0f;
        sum = fmaf(experts[row * n_embd + col] * scale, weights[row], sum);
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    // has_scale as a template parameter: a nullptr test inside the expert loop split it into branches, so the
    // unrolled loads could not be issued together (same arithmetic either way)
    // NE = 8 (GLM-5.3 / DeepSeek top-8) is compiled fully unrolled so all experts' loads are issued together
    if (expert_scale != nullptr) {
        if (n_expert_used == 8) {
            moe_weighted_reduction_f32<true, 8><<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
        } else {
            moe_weighted_reduction_f32<true, 0><<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
        }
    } else {
        if (n_expert_used == 8) {
            moe_weighted_reduction_f32<false, 8><<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
        } else {
            moe_weighted_reduction_f32<false, 0><<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
        }
    }
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
