#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdlib>

// NPRE > 0 (short batches, e.g. the 3-token MTP verify): every token's k/q/v/g/beta is loaded before the recurrence
// starts, so the per-token global latency is not exposed on the serial state chain. Arithmetic and reduction order
// per token are unchanged (bit-exact); requires n_tokens <= NPRE.
template <int S_v, bool KDA, bool keep_rs_t, int NPRE = 0>
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

    // NPRE: preload all tokens (see the template comment)
    constexpr int NP = NPRE > 0 ? NPRE : 1;
    [[maybe_unused]] float k_pre[NP][rows_per_lane];
    [[maybe_unused]] float q_pre[NP][rows_per_lane];
    [[maybe_unused]] float g_pre[NP][KDA ? rows_per_lane : 1];
    [[maybe_unused]] float v_pre[NP], beta_pre[NP];
    if constexpr (NPRE > 0) {
#pragma unroll
        for (int t = 0; t < NPRE; t++) {
            if (t < n_tokens) {
                const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
                const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
                const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    k_pre[t][r] = k_t[i];
                    q_pre[t][r] = q_t[i];
                    if constexpr (KDA) {
                        g_pre[t][r] = g[gb_offset * S_v + i];
                    }
                }
                if constexpr (!KDA) {
                    g_pre[t][0] = g[gb_offset];
                }
                v_pre[t]    = v[sequence * sv3 + t * sv2 + h_idx * sv1 + col];
                beta_pre[t] = beta[gb_offset];
            }
        }
    }

#pragma unroll
    for (int t = 0; t < (NPRE > 0 ? NPRE : n_tokens); t++) {
        if (NPRE > 0 && t >= n_tokens) {
            continue;   // not break: a single-exit constant-trip loop can be fully unrolled (preload arrays stay in VGPRs)
        }
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = NPRE > 0 ? beta_pre[NPRE > 0 ? t : 0] : *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = NPRE > 0 ? k_pre[NPRE > 0 ? t : 0][r] : k_t[i];
            q_reg[r] = NPRE > 0 ? q_pre[NPRE > 0 ? t : 0][r] : q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(NPRE > 0 ? g_pre[NPRE > 0 ? t : 0][0] : *g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = ((NPRE > 0 ? v_pre[NPRE > 0 ? t : 0] : v_t[col]) - g_val * kv_col) * beta_val;

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
                kv_shard += expf(NPRE > 0 ? g_pre[NPRE > 0 ? t : 0][KDA ? r : 0] : g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = ((NPRE > 0 ? v_pre[NPRE > 0 ? t : 0] : v_t[col]) - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(NPRE > 0 ? g_pre[NPRE > 0 ? t : 0][KDA ? r : 0] : g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
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

template <int S_v, bool KDA, bool keep_rs_t, bool G_EXP = false>
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
    constexpr int D    = KDA ? 4 : 8; // prefetch depth in tokens (8 with the exp(g) input measured 5-7% slower)
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
            if constexpr (KDA) { ge[r] = G_EXP ? gk[d][r] : expf(gk[d][r]); }
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

// rune gfx908: exp(g) of the KDA gate once per element. The lpc kernel evaluates expf(g) for every (token, row) in each
// of the S_v columns of a head (S_v x redundant, ~half of its per-token instruction stream at 1 wave/SIMD); here each
// element is exponentiated once with the same expf, and the kernel reads the result: identical values, bit-exact.
// GGML_GDN_EXPG=0 disables.
static __global__ void gdn_expg_f32(const float * __restrict__ g, float * __restrict__ eg, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        eg[i] = expf(g[i]);
    }
}

// rune gfx908 KDA prefill, LDS-shared token ring (GGML_GDN_LDS, default on; =0 restores the lpc kernel).
// Same lane-per-column arithmetic as gated_delta_net_lpc_cuda<S_v, true, keep_rs_t, true> (16 lanes per state column,
// S_v/16 rows each, identical per-token operation order and DPP reductions -> bit-exact), different data flow: a block
// is 4 waves = 16 columns of ONE head, and the head's per-token k / q / exp(g) rows, the 16 columns' v and beta are
// staged once per block in an LDS ring of TB tokens (double-buffered, one barrier per TB tokens) instead of every
// 4-column wave holding its own D-token register ring. That removes the 4x-per-block (32x-per-head) re-reads of k/q/g
// through L2 and the >128-VGPR register ring, so the 1024 waves of a GLM-5.3 KDA layer fit in one round (the lpc kernel
// runs 1 wave/SIMD: 3 rounds of 480/480/64 waves) and the per-token chain reads LDS instead of waiting on global loads.
template <int S_v, bool keep_rs_t, int TB>
__global__ void __launch_bounds__(256, 2)
gated_delta_net_lds_cuda(const float * q, const float * k, const float * v, const float * eg, const float * beta,
                         const float * curr_state, float * dst, float * state,
                         int64_t H, int64_t n_tokens, int64_t sq1, int64_t sq2, int64_t sq3,
                         int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
                         const uint3 neqk1_magic, const uint3 rq3_magic, float scale, int64_t state_slot_stride, int K) {
    constexpr int LPC  = 16;          // lanes per column (as lpc)
    constexpr int NT   = 256;         // 4 waves
    constexpr int COLS = NT / LPC;    // 16 columns per block
    constexpr int RPL  = S_v / LPC;   // rows per lane
    constexpr int V4   = S_v / 4;     // float4 per k/q/eg row
    // stage layout (floats): k[TB][S_v] q[TB][S_v] eg[TB][S_v] v[TB][COLS] beta[TB]
    constexpr int OFF_Q = TB*S_v, OFF_G = 2*TB*S_v, OFF_V = 3*TB*S_v, OFF_B = 3*TB*S_v + TB*COLS;
    constexpr int STAGE = OFF_B + 4*((TB + 3)/4);
    constexpr int NLD4  = 3*TB*V4 + TB*COLS/4; // float4 slots per stage (k, q, eg, v); beta separately
    constexpr int NSLOT = (NLD4 + NT - 1) / NT;
    static_assert(S_v % LPC == 0 && RPL % 4 == 0 && COLS % 4 == 0, "unsupported S_v");

    __shared__ __align__(16) float ring[2][STAGE];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      tid      = threadIdx.x;
    const int      sub      = tid % LPC;
    const int      colb     = blockIdx.z * COLS;
    const int      cl       = tid / LPC;           // column inside the block
    const int      col      = colb + cl;
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

    const float * kbase = k + iq3 * sq3 + iq1 * sq1;
    const float * qbase = q + iq3 * sq3 + iq1 * sq1;
    const float * vbase = v + sequence * sv3 + h_idx * sv1 + colb;
    const float * bbase = beta + sequence * sb3 + h_idx * sb1;
    const float * gbase = eg + (sequence * sb3 + h_idx * sb1) * S_v;

    // per-thread load slots of a stage (2 per thread at S_v = 128, TB = 4): source at token 0, per-token source stride,
    // token slot (-1: idle), LDS offset. Built branch-free and held in scalars: arrays assigned in branches went to scratch.
    static_assert(NSLOT <= 2, "more than 2 load slots per thread");
    struct slot { const float * src; int64_t str; int tok; int dst; };
    auto make_slot = [&](const int i) -> slot {
        const int seg = i < TB*V4 ? 0 : i < 2*TB*V4 ? 1 : i < 3*TB*V4 ? 2 : i < NLD4 ? 3 : 4;
        const int j   = i - (seg < 3 ? seg : 3)*TB*V4;
        const int per = seg < 3 ? V4 : COLS/4;          // float4 per token in this array
        const int tok = seg < 4 ? j / per : -1;
        const int f4  = seg < 4 ? j % per : 0;
        slot sl;
        sl.src = seg == 0 ? kbase + 4*f4 : seg == 1 ? qbase + 4*f4 : seg == 2 ? gbase + 4*f4 : seg == 3 ? vbase + 4*f4 : kbase;
        sl.str = seg < 2 ? sq2 : seg == 2 ? sb2*S_v : seg == 3 ? sv2 : 0;
        sl.tok = tok;
        sl.dst = seg < 3 ? seg*TB*S_v + tok*S_v + 4*f4 : seg == 3 ? OFF_V + tok*COLS + 4*f4 : 0;
        return sl;
    };
    const slot sl0 = make_slot(tid);
    const slot sl1 = make_slot(NT + tid);
    float4 lr0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), lr1 = lr0;
    float  breg = 0.0f;
    // stage c covers tokens c*TB .. c*TB + TB - 1 (clamped to n_tokens - 1: past-the-end slots load valid data, unused);
    // loads are unconditional (idle slots re-read k), only the LDS stores are masked
    auto load_stage = [&](const int c) {
        lr0 = *(const float4 *) (sl0.src + min(c*TB + max(sl0.tok, 0), (int) n_tokens - 1)*sl0.str);
        if (NSLOT > 1) {
            lr1 = *(const float4 *) (sl1.src + min(c*TB + max(sl1.tok, 0), (int) n_tokens - 1)*sl1.str);
        }
        if (tid < TB) {
            const int t = min(c*TB + tid, (int) n_tokens - 1);
            breg = bbase[t * sb2];
        }
    };
    auto store_stage = [&](const int buf) {
        if (sl0.tok >= 0) {
            *(float4 *) &ring[buf][sl0.dst] = lr0;
        }
        if (NSLOT > 1 && sl1.tok >= 0) {
            *(float4 *) &ring[buf][sl1.dst] = lr1;
        }
        if (tid < TB) {
            ring[buf][OFF_B + tid] = breg;
        }
    };

    // one token step: the arithmetic of the lpc kernel's step() with G_EXP (ge = exp(g) read from the ring)
    // st: the stage buffer, d: token slot inside the stage
    auto step = [&](const int t, const float * st, const int d, const bool do_store) {
        float kr[RPL], qr[RPL], ge[RPL];
#pragma unroll
        for (int r = 0; r < RPL; r += 4) {
            const float4 k4 = *(const float4 *) (st +         d*S_v + row0 + r);
            const float4 q4 = *(const float4 *) (st + OFF_Q + d*S_v + row0 + r);
            const float4 g4 = *(const float4 *) (st + OFF_G + d*S_v + row0 + r);
            kr[r+0] = k4.x; kr[r+1] = k4.y; kr[r+2] = k4.z; kr[r+3] = k4.w;
            qr[r+0] = q4.x; qr[r+1] = q4.y; qr[r+2] = q4.z; qr[r+3] = q4.w;
            ge[r+0] = g4.x; ge[r+1] = g4.y; ge[r+2] = g4.z; ge[r+3] = g4.w;
        }
        const float v_col    = st[OFF_V + d*COLS + cl];
        const float beta_val = st[OFF_B + d];

        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            kv += ge[r] * s[r] * kr[r];
        }
        kv = gdn_sum16(kv);

        const float delta_col = (v_col - kv) * beta_val;

        float attn = 0.0f;
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            s[r] = ge[r] * s[r] + kr[r] * delta_col;
            attn += s[r] * qr[r];
        }
        attn = gdn_sum16(attn);
        attn_data[(int64_t) t * S_v * H + col] = attn * scale;

        if (keep_rs_t && do_store) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * sp = state + target_slot * state_slot_stride + col * S_v + row0;
#pragma unroll
                for (int r = 0; r < RPL; r++) {
                    sp[r] = s[r];
                }
            }
        }
    };

    const int nstage = ((int) n_tokens + TB - 1) / TB;
    load_stage(0);
    store_stage(0);
    __syncthreads();

    // stages whose tokens never store a snapshot run branch-free (as the lpc kernel's main loop)
    const int n_main = keep_rs_t ? (int) n_tokens - K : (int) n_tokens;
    for (int c = 0; c < nstage; ++c) {
        const int buf = c & 1;
        const bool next = c + 1 < nstage;
        if (next) {
            load_stage(c + 1);
        }
        const float * st0 = ring[buf];
        const int t0 = c*TB;
        if (t0 + TB <= n_main) {
#pragma unroll
            for (int d = 0; d < TB; d++) {
                step(t0 + d, st0, d, false);
            }
        } else {
#pragma unroll
            for (int d = 0; d < TB; d++) {
                if (t0 + d < n_tokens) {
                    step(t0 + d, st0, d, true);
                }
            }
        }
        if (next) {
            store_stage(buf ^ 1);
        }
        __syncthreads();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            state[col * S_v + row0 + r] = s[r];
        }
    }
}

static bool gated_delta_net_use_expg() {
    const char * e = getenv("GGML_GDN_EXPG");
    return e == nullptr || atoi(e) != 0;
}

template <int S_v, bool KDA, bool keep_rs_t>
static void launch_gated_delta_net_lpc(
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3, float scale, int64_t state_slot_stride, int K, cudaStream_t stream,
        ggml_cuda_pool * pool = nullptr) {
    if constexpr (KDA) {
        if (pool != nullptr && gated_delta_net_use_expg()) {
            // g is contiguous [S_v, H, n_tokens, n_seqs] (asserted by the caller); the kernel indexes it with the beta
            // strides times S_v, so the same layout serves the exponentiated copy
            const int64_t n = S_v * H * n_tokens * n_seqs;
            ggml_cuda_pool_alloc<float> eg(*pool, n);
            gdn_expg_f32<<<(n + 255) / 256, 256, 0, stream>>>(g_d, eg.get(), n);
            static const bool lds = [] { const char * e = getenv("GGML_GDN_LDS"); return e == nullptr || atoi(e) != 0; }();
            if constexpr (S_v == 128) if (lds) {
                // LDS-shared token ring, 4 waves = 16 columns of one head per block (bit-exact with the lpc kernel)
                const dim3 grid_lds(H, n_seqs, S_v / 16);
                const uint3 neqk1_m = init_fastdiv_values(neqk1);
                const uint3 rq3_m   = init_fastdiv_values(rq3);
                gated_delta_net_lds_cuda<S_v, keep_rs_t, 4><<<grid_lds, 256, 0, stream>>>(
                    q_d, k_d, v_d, (const float *) eg.get(), b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3,
                    sv1, sv2, sv3, sb1, sb2, sb3, neqk1_m, rq3_m, scale, state_slot_stride, K);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            const dim3 grid_dims(H, n_seqs, S_v / 4);
            const dim3 block_dims(64, 1, 1);
            const uint3 neqk1_magic = init_fastdiv_values(neqk1);
            const uint3 rq3_magic   = init_fastdiv_values(rq3);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_lpc_cuda<S_v, KDA, keep_rs_t, true>, launch_params,
                q_d, k_d, v_d, (const float *) eg.get(), b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            return;
        }
    }
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
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream, ggml_cuda_pool * pool = nullptr) {
    //TODO: Add chunked kernel for even faster pre-fill
    if (gated_delta_net_use_lpc(S_v, n_tokens)) {
        if (S_v == 128) {
            launch_gated_delta_net_lpc<128, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, pool);
        } else {
            launch_gated_delta_net_lpc<64, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, pool);
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
            // 2..4 tokens (MTP verify): preload variant, bit-exact with the plain loop. GGML_GDN_PRELOAD=0 disables.
            static const bool preload = !getenv("GGML_GDN_PRELOAD") || atoi(getenv("GGML_GDN_PRELOAD")) != 0;
            if (preload && n_tokens >= 2 && n_tokens <= 4) {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, 4>, launch_params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                break;
            }
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
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, &ctx.pool());
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream, &ctx.pool());
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
