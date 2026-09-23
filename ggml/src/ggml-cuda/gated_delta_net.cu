#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdlib>

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}


// ---------------------------------------------------------------------------------------------
// Multi-token variant for gfx908 (tolerance-only: different dot-product summation order than the warp-per-column
// kernel above). The reference kernel gives each state column a whole wave and does two 64-lane reductions per token,
// which makes the token loop instruction-issue bound (~4.7 us/token for 48 heads on MI100). Here 16 lanes share a
// column (S_v/16 rows each): a token step is S_v/16 FMAs per phase plus a 4-step DPP reduction (register lane
// permutes, no LDS, no barriers). Each lane loads exactly its own rows of k/q (float4, L1/L2-hot) into a register
// ring that runs D tokens ahead, so the ~1.5 us global round trip is hidden behind D tokens of math.
#if defined(GGML_USE_HIP)
template <int ctrl> static __device__ __forceinline__ float gdn_dpp(const float x) {
    // dpp_ctrl: quad_perm xor1 = 0xB1, xor2 = 0x4E, row_ror:4 = 0x124, row_ror:8 = 0x128
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), ctrl, 0xF, 0xF, true));
}
static __device__ __forceinline__ float gdn_sum16(float x) {
    x += gdn_dpp<0xB1>(x);
    x += gdn_dpp<0x4E>(x);
    x += gdn_dpp<0x124>(x);
    x += gdn_dpp<0x128>(x);
    return x;
}
#else
static __device__ __forceinline__ float gdn_sum16(float x) {
#pragma unroll
    for (int m = 1; m < 16; m <<= 1) {
        x += __shfl_xor_sync(0xFFFFFFFF, x, m, 16);
    }
    return x;
}
#endif // defined(GGML_USE_HIP)

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__(64, 1) // 1 wave per SIMD: the D-token register ring needs > 128 VGPRs; at 2 waves/SIMD it spills to AGPRs and every spill of a pending load forces a vmcnt(0) drain
gated_delta_net_lpc_cuda(const float * q, const float * k, const float * v, const float * g, const float * beta,
                         const float * curr_state, float * dst, float * state,
                         int64_t H, int64_t n_tokens, int64_t sq1, int64_t sq2, int64_t sq3,
                         int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
                         const uint3 neqk1_magic, const uint3 rq3_magic, float scale, int64_t state_slot_stride, int K) {
    constexpr int LPC  = 16;          // lanes per column
    constexpr int NT   = 64;          // threads per block = one wave
    constexpr int COLS = NT / LPC;    // 4 columns per block
    constexpr int RPL  = S_v / LPC;   // rows per lane
    constexpr int D    = KDA ? 4 : 8; // prefetch depth in tokens
    static_assert(S_v % LPC == 0 && RPL % 4 == 0, "unsupported S_v");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      tid      = threadIdx.x;
    const int      sub      = tid % LPC;
    const int      col      = blockIdx.z * COLS + tid / LPC;
    const int      row0     = sub * RPL;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    float s[RPL];
#pragma unroll
    for (int r = 0; r < RPL; r++) {
        s[r] = curr_state[row0 + r];
    }

    const float * kbase = k + iq3 * sq3 + iq1 * sq1 + row0;
    const float * qbase = q + iq3 * sq3 + iq1 * sq1 + row0;
    const float * vbase = v + sequence * sv3 + h_idx * sv1 + col;
    const float * bbase = beta + sequence * sb3 + h_idx * sb1;
    const float * gbase = g + (sequence * sb3 + h_idx * sb1) * (KDA ? S_v : 1) + (KDA ? row0 : 0);

    // register ring, D tokens deep
    float kk[D][RPL];
    float qq[D][RPL];
    float gk[KDA ? D : 1][KDA ? RPL : 1];
    float vv[D], bb[D], gs[D];
    auto fetch = [&](int t, const int d) {
        t = min(t, (int) n_tokens - 1); // clamp instead of branch: past-the-end slots load valid data that is never used
        const float * kt = kbase + t * sq2;
        const float * qt = qbase + t * sq2;
#pragma unroll
        for (int r = 0; r < RPL; r += 4) {
            const float4 k4 = *(const float4 *) (kt + r);
            const float4 q4 = *(const float4 *) (qt + r);
            kk[d][r+0] = k4.x; kk[d][r+1] = k4.y; kk[d][r+2] = k4.z; kk[d][r+3] = k4.w;
            qq[d][r+0] = q4.x; qq[d][r+1] = q4.y; qq[d][r+2] = q4.z; qq[d][r+3] = q4.w;
        }
        vv[d] = vbase[t * sv2];
        bb[d] = bbase[t * sb2];
        if constexpr (KDA) {
            const float * gt = gbase + t * sb2 * S_v;
#pragma unroll
            for (int r = 0; r < RPL; r++) {
                gk[d][r] = gt[r];
            }
        } else {
            gs[d] = gbase[t * sb2];
        }
    };
#pragma unroll
    for (int d = 0; d < D; d++) {
        fetch(d, d);
    }

    // One token step. Kept branch-free (unconditional attn store from all 16 lanes of the group, same value and
    // address) so that a group of D tokens compiles to one straight-line block: the compiler's wait-count
    // analysis then keeps the ring loads pending across tokens instead of draining them at every block boundary.
    auto step = [&](const int t, const int d, const bool do_store) {
        float kr[RPL], qr[RPL], ge[KDA ? RPL : 1];
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            kr[r] = kk[d][r];
            qr[r] = qq[d][r];
            if constexpr (KDA) { ge[r] = expf(gk[d][r]); }
        }
        const float v_col    = vv[d];
        const float beta_val = bb[d];
        const float g_val    = KDA ? 1.0f : expf(gs[d]);
        fetch(t + D, d);

        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            kv += (KDA ? ge[r] : 1.0f) * s[r] * kr[r];
        }
        kv = gdn_sum16(kv);

        const float delta_col = KDA ? (v_col - kv) * beta_val : (v_col - g_val * kv) * beta_val;

        float attn = 0.0f;
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            s[r] = (KDA ? ge[r] : g_val) * s[r] + kr[r] * delta_col;
            attn += s[r] * qr[r];
        }
        attn = gdn_sum16(attn);
        attn_data[(int64_t) t * S_v * H + col] = attn * scale;

        if (keep_rs_t && do_store) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * st = state + target_slot * state_slot_stride + col * S_v + row0;
#pragma unroll
                for (int r = 0; r < RPL; r++) {
                    st[r] = s[r];
                }
            }
        }
    };

    // keep_rs (MTP rollback snapshots): only the last K tokens store a state. Doing that check inside the
    // straight-line D-token group breaks it into basic blocks and the wait-count pass drains the ring every token
    // (2.47x slower, measured), so the tokens that never store run in a separate branch-free main loop.
    // Same per-token arithmetic and order either way (bit-exact).
    int t0 = 0;
    const int n_main = keep_rs_t ? (int) n_tokens - K : (int) n_tokens;
    for (; t0 + D <= n_main; t0 += D) {
#pragma unroll
        for (int d = 0; d < D; d++) {
            step(t0 + d, d, false);
        }
    }
    // remainder (< D tokens without snapshots, plus the snapshot tokens), t0 stays a multiple of D so the ring slot is t % D
    for (; t0 < n_tokens; t0 += D) {
#pragma unroll
        for (int d = 0; d < D; d++) {
            if (t0 + d < n_tokens) {
                step(t0 + d, d, true);
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            state[col * S_v + row0 + r] = s[r];
        }
    }
}

template <int S_v, bool KDA, bool keep_rs_t>
static void launch_gated_delta_net_lpc(
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const dim3 grid_dims(H, n_seqs, S_v / 4);
    const dim3 block_dims(64, 1, 1);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_lpc_cuda<S_v, KDA, keep_rs_t>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
}

// GGML_GDN_LPC: unset/1 = lane-per-column kernel for batches of >= GGML_GDN_LPC_MIN tokens (default 8), 0 = never, 2 = always.
static bool gated_delta_net_use_lpc(const int64_t S_v, const int64_t n_tokens) {
    static const int  mode    = [] { const char * e = getenv("GGML_GDN_LPC");     return e ? atoi(e) : 1; }();
    static const int  min_tok = [] { const char * e = getenv("GGML_GDN_LPC_MIN"); return e ? atoi(e) : 8; }();
    if (mode == 0 || (S_v != 64 && S_v != 128)) {
        return false;
    }
    return mode == 2 || n_tokens >= min_tok;
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    if (gated_delta_net_use_lpc(S_v, n_tokens)) {
        if (S_v == 128) {
            launch_gated_delta_net_lpc<128, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net_lpc<64, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
        return;
    }
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
