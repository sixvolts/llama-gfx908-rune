#include "gdn-gate.cuh"
#include "mmvf-replay.cuh"

// Elementwise tails exactly as unary.cu / binbcast run them (separate kernels there: no contraction possible);
// block-scoped contract(off) as in hc-fused.cu so the file's default is untouched
static __device__ __forceinline__ float gdn_gate_alpha_tail(const float v, const float dt, const float a) {
#pragma clang fp contract(off)
    const float s = v + dt;
    const float p = (s > 20.0f) ? s : logf(1.0f + expf(s));
    return p * a;
}
static __device__ __forceinline__ float gdn_gate_beta_tail(const float v) {
#pragma clang fp contract(off)
    return 1.0f / (1.0f + expf(-v));
}

// One block per output row (rows [0, na) from W_alpha, [na, na+nb) from W_beta), 128 threads = 2 waves, exactly
// mul_mat_vec_f<float, float, ncols_dst, 128, false, false>'s K loop and reduction.
template <int ncols_dst>
static __global__ void __launch_bounds__(128) gdn_gate_fused_kernel(
        const float * __restrict__ w_alpha, const float * __restrict__ w_beta, const float * __restrict__ y,
        const float * __restrict__ dt, const float * __restrict__ a,
        float * __restrict__ gate, float * __restrict__ beta,
        const int ncols2, const int na, const int stride_row_a, const int stride_row_b, const int stride_col_y2) {
    constexpr int warp_size = 64;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const bool is_alpha = row < na;
    const float2 * x2 = (const float2 *) (is_alpha ? w_alpha + (size_t) row*stride_row_a : w_beta + (size_t) (row - na)*stride_row_b);
    const float2 * y2 = (const float2 *) y;

    __shared__ float buf_iw[warp_size];
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    float sumf[ncols_dst];
    ggml_cuda_mmvf_replay_row_128<ncols_dst>(x2, y2, ncols2, stride_col_y2, buf_iw, tid, sumf);

    if (tid >= ncols_dst) {
        return;
    }
    const float value = sumf[tid];
    if (is_alpha) {
        gate[tid*na + row] = gdn_gate_alpha_tail(value, dt[row], a[row]);
    } else {
        const int nb_row = row - na;
        beta[tid*(gridDim.x - na) + nb_row] = gdn_gate_beta_tail(value);
    }
}

bool ggml_cuda_op_gdn_gate_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_alpha, const ggml_tensor * w_beta, const ggml_tensor * x,
        const ggml_tensor * dt, const ggml_tensor * a, ggml_tensor * gate, ggml_tensor * beta) {
    const int64_t K  = x->ne[0];
    const int64_t nt = x->ne[1];
    const int64_t na = w_alpha->ne[1];
    const int64_t nb = w_beta->ne[1];
    if (nt < 1 || nt > 4 || K % 2 != 0 || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    // the replay is the gfx908 (wave 64, CDNA1) mmvf variant: other architectures pick other kernels for these shapes
    {
        const int cc = ggml_cuda_info().devices[ctx.device].cc;
        if (ggml_cuda_info().devices[ctx.device].warp_size != 64 || !GGML_CUDA_CC_IS_CDNA1(cc)) {
            return false;
        }
    }
    // the kernel replays the 128-thread mul_mat_vec_f variant only
    if (ggml_cuda_mmvf_block_size_cdna(K) != 128) {
        return false;
    }
    const int stride_row_a  = (int) (w_alpha->nb[1] / sizeof(float));
    const int stride_row_b  = (int) (w_beta->nb[1]  / sizeof(float));
    const int stride_col_y2 = (int) (x->nb[1] / sizeof(float) / 2);
    if (stride_row_a % 2 != 0 || stride_row_b % 2 != 0 || (x->nb[1] / sizeof(float)) % 2 != 0) {
        return false;
    }
    const dim3 grid((unsigned) (na + nb), 1, 1);
    const dim3 block(128, 1, 1);
    cudaStream_t stream = ctx.stream();
    const float * wa = (const float *) w_alpha->data;
    const float * wb = (const float *) w_beta->data;
    const float * xd = (const float *) x->data;
    const float * dd = (const float *) dt->data;
    const float * ad = (const float *) a->data;
    float * gd = (float *) gate->data;
    float * bd = (float *) beta->data;
    switch (nt) {
        case 1: gdn_gate_fused_kernel<1><<<grid, block, 0, stream>>>(wa, wb, xd, dd, ad, gd, bd, (int) (K/2), (int) na, stride_row_a, stride_row_b, stride_col_y2); break;
        case 2: gdn_gate_fused_kernel<2><<<grid, block, 0, stream>>>(wa, wb, xd, dd, ad, gd, bd, (int) (K/2), (int) na, stride_row_a, stride_row_b, stride_col_y2); break;
        case 3: gdn_gate_fused_kernel<3><<<grid, block, 0, stream>>>(wa, wb, xd, dd, ad, gd, bd, (int) (K/2), (int) na, stride_row_a, stride_row_b, stride_col_y2); break;
        default: gdn_gate_fused_kernel<4><<<grid, block, 0, stream>>>(wa, wb, xd, dd, ad, gd, bd, (int) (K/2), (int) na, stride_row_a, stride_row_b, stride_col_y2); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
