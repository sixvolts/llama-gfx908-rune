// GGML_OP_SPARSE_ATTN on CDNA (gfx908): attention of each query over its own list of KV cells, one KV head whose V
// is a prefix of the K row (DSA over absorbed MLA; see ggml_sparse_attn in ggml.h).
//
// One block = one query x 16 heads x a range of 64-cell tiles, 4 waves. Per tile:
//   QK   each wave gathers the K rows of its 16 cells straight from the cache into registers (16-byte loads) and runs
//        S^T[cell][head] = K.Q^T on v_mfma_f32_16x16x16f16 over D. Q is held in registers as f16 for the whole block.
//        The contraction order over D is permuted (a lane's 16-byte load feeds two MFMA steps); Q uses the same order.
//   soft online softmax per head across the block's 64 cells (lane-group shuffles + LDS across waves), P -> LDS (f16)
//   PV   V^T is staged through LDS in 128-wide chunks of D_v from the K registers; each wave owns 2 of the 8 16-wide
//        d-tiles of a chunk and runs O[head][d] += P.V on the same MFMA.
// Precision matches the dense -fa off path (f16 K, Q and P, f32 accumulation and softmax). For few queries the cell
// range is split across blocks and a second kernel merges the partial (O, max, sum) triples.

#include "sparse-attn.cuh"

static constexpr int SA_HEADS = 16;  // heads per block: the MFMA tile's N
static constexpr int SA_TILE  = 64;  // cells per tile: 16 per wave
static constexpr int SA_DCH   = 128; // width of the V^T chunk staged in LDS
static constexpr int SA_NW    = 4;   // waves per block
static constexpr int SA_PAD   = 4;   // LDS row padding in halfs (rows stay 8-byte aligned)

#if defined(AMD_MFMA_AVAILABLE)
typedef _Float16 sa_half4 __attribute__((ext_vector_type(4)));
typedef float    sa_float4 __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ sa_half4 sa_h4(const uint32_t lo, const uint32_t hi) {
    union { uint2 u; sa_half4 h; } c;
    c.u = make_uint2(lo, hi);
    return c.h;
}
#endif // defined(AMD_MFMA_AVAILABLE)

template <int D, int DV>
__launch_bounds__(SA_NW*64, 1)
static __global__ void sparse_attn_mfma(
        const float   * __restrict__ q,
        const char    * __restrict__ k,
        const int32_t * __restrict__ idx,
        const float   * __restrict__ mask,
        float         * __restrict__ dst,
        float         * __restrict__ part_o,
        float2        * __restrict__ part_ml,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbq3,
        const int64_t nbk2, const int64_t nbk3,
        const int n_head, const int n_q, const int n_kv, const int n_sel,
        const float scale, const int tiles_per_split) {
#if defined(AMD_MFMA_AVAILABLE)
    static_assert(D % 32 == 0 && DV % SA_DCH == 0 && DV <= D, "sparse_attn: unsupported head size");

    constexpr int NP = D/32;      // 16-byte K loads per lane per cell (two MFMA k-steps each)
    constexpr int NC = DV/SA_DCH; // V^T chunks

    const int tid  = threadIdx.x;
    const int w    = tid / 64;
    const int lane = tid % 64;
    const int l16  = lane % 16;
    const int g    = lane / 16;

    const int tok   = blockIdx.x; // s*n_q + t
    const int s     = tok / n_q;
    const int t     = tok % n_q;
    const int h0    = blockIdx.y*SA_HEADS;
    const int split = blockIdx.z;

    const int32_t * idx_row  = idx  + (int64_t) tok*n_sel;
    const float   * mask_row = mask + (int64_t) tok*n_sel;

    __shared__ _Float16 sP [SA_HEADS][SA_TILE + SA_PAD];
    __shared__ _Float16 sVT[SA_DCH  ][SA_TILE + SA_PAD];
    __shared__ float    sRed[SA_NW][SA_HEADS];
    __shared__ float    sAlpha[SA_HEADS];
    __shared__ float    sM[SA_HEADS];
    __shared__ float    sL[SA_HEADS];

    // Q^T as the B operand of QK: lane = head h0 + l16; k-slot 4*g + e of step 2p (2p+1) is d = 32p + 8g + e (+ 4)
    sa_half4 qb[2*NP];
    {
        const float * qh = (const float *) ((const char *) q + (int64_t) (h0 + l16)*nbq1 + (int64_t) t*nbq2 + (int64_t) s*nbq3);
#pragma unroll
        for (int p = 0; p < NP; ++p) {
            const float4 a = *(const float4 *) (qh + 32*p + 8*g);
            const float4 b = *(const float4 *) (qh + 32*p + 8*g + 4);
            qb[2*p + 0] = sa_half4{(_Float16) a.x, (_Float16) a.y, (_Float16) a.z, (_Float16) a.w};
            qb[2*p + 1] = sa_half4{(_Float16) b.x, (_Float16) b.y, (_Float16) b.z, (_Float16) b.w};
        }
    }

    // O[head = 4g + v][d = chunk*128 + (2w + i)*16 + l16], the MFMA D layout
    sa_float4 o[NC][2];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        o[c][0] = sa_float4{0.0f, 0.0f, 0.0f, 0.0f};
        o[c][1] = sa_float4{0.0f, 0.0f, 0.0f, 0.0f};
    }

    // running max and sum of head h0 + l16 (every lane of that head holds a copy)
    float m_run = -INFINITY;
    float l_run = 0.0f;

    const int n_tiles = (n_sel + SA_TILE - 1)/SA_TILE;
    const int it0     = split*tiles_per_split;
    const int it1     = min(n_tiles, it0 + tiles_per_split);

    for (int it = it0; it < it1; ++it) {
        // ---- QK: this lane's cell is slot it*64 + 16w + l16 (the A operand's row)
        const int  jl   = it*SA_TILE + 16*w + l16;
        int        cell = jl < n_sel ? idx_row[jl] : 0;
        cell = min(max(cell, 0), n_kv - 1);
        const char * krow = k + (int64_t) cell*nbk2 + (int64_t) s*nbk3;

        uint4 kr[NP];
#pragma unroll
        for (int p = 0; p < NP; ++p) {
            kr[p] = *(const uint4 *) (krow + 2*(32*p + 8*g));
        }

        sa_float4 acc = sa_float4{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int p = 0; p < NP; ++p) {
            acc = __builtin_amdgcn_mfma_f32_16x16x16f16(sa_h4(kr[p].x, kr[p].y), qb[2*p + 0], acc, 0, 0, 0);
            acc = __builtin_amdgcn_mfma_f32_16x16x16f16(sa_h4(kr[p].z, kr[p].w), qb[2*p + 1], acc, 0, 0, 0);
        }

        // ---- online softmax: acc[v] = S[cell 16w + 4g + v][head l16]
        float sv[4];
        float mx = -INFINITY;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int jc = it*SA_TILE + 16*w + 4*g + v;
            const float mk = jc < n_sel ? mask_row[jc] : -INFINITY;
            sv[v] = mk == -INFINITY ? -INFINITY : acc[v]*scale + mk;
            mx = fmaxf(mx, sv[v]);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 64));
        mx = fmaxf(mx, __shfl_xor(mx, 32, 64));
        if (g == 0) {
            sRed[w][l16] = mx;
        }
        __syncthreads();

        const float m_tile = fmaxf(fmaxf(sRed[0][l16], sRed[1][l16]), fmaxf(sRed[2][l16], sRed[3][l16]));
        const float m_new  = fmaxf(m_run, m_tile);
        // m_new == -inf: nothing visible yet, keep the (zero) state; m_run == -inf, m_new finite: exp(-inf) = 0
        const float alpha  = m_new == -INFINITY ? 1.0f : expf(m_run - m_new);

        float pv[4];
        float ps = 0.0f;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            pv[v] = sv[v] == -INFINITY ? 0.0f : expf(sv[v] - m_new);
            ps += pv[v];
        }
        ps += __shfl_xor(ps, 16, 64);
        ps += __shfl_xor(ps, 32, 64);
        __syncthreads(); // every wave has read the maxima

        if (g == 0) {
            sRed[w][l16] = ps;
        }
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            sP[l16][16*w + 4*g + v] = (_Float16) pv[v];
        }
        if (w == 0 && g == 0) {
            sAlpha[l16] = alpha;
        }
        __syncthreads();

        l_run = alpha*l_run + (sRed[0][l16] + sRed[1][l16] + sRed[2][l16] + sRed[3][l16]);
        m_run = m_new;

        // ---- PV: rescale O by its heads' alpha, then accumulate chunk by chunk
        {
            float al[4];
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                al[v] = sAlpha[4*g + v];
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        o[c][i][v] *= al[v];
                    }
                }
            }
        }

#pragma unroll
        for (int c = 0; c < NC; ++c) {
            // V^T[d - 128c][cell 16w + l16] from this lane's K registers: kr[p] holds d = 32p + 8g + e
#pragma unroll
            for (int pp = 0; pp < SA_DCH/32; ++pp) {
                const int p = c*(SA_DCH/32) + pp;
                union { uint4 u; _Float16 h[8]; } kv;
                kv.u = kr[p];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    sVT[32*pp + 8*g + e][16*w + l16] = kv.h[e];
                }
            }
            __syncthreads();

#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int dt = 2*w + i; // d-tile within the chunk
#pragma unroll
                for (int ks = 0; ks < SA_TILE/16; ++ks) {
                    const sa_half4 a = *(const sa_half4 *) &sP [l16        ][16*ks + 4*g];
                    const sa_half4 b = *(const sa_half4 *) &sVT[16*dt + l16][16*ks + 4*g];
                    o[c][i] = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, o[c][i], 0, 0, 0);
                }
            }
            __syncthreads(); // before the next chunk (or tile) overwrites sVT / sP
        }
    }

    // ---- output: stats of head 4g + v for the O layout
    if (w == 0 && g == 0) {
        sM[l16] = m_run;
        sL[l16] = l_run;
    }
    __syncthreads();

    const int64_t row0 = (int64_t) tok*n_head + h0; // (token, head) row of the output

    if (part_o == nullptr) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int   hh  = 4*g + v;
            const float l   = sL[hh];
            const float inv = l > 0.0f ? 1.0f/l : 0.0f;
            float * out = dst + (row0 + hh)*DV;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    out[c*SA_DCH + (2*w + i)*16 + l16] = o[c][i][v]*inv;
                }
            }
        }
    } else {
        const int64_t n_rows = (int64_t) gridDim.x*n_head;
        float * po = part_o + (int64_t) split*n_rows*DV;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int hh = 4*g + v;
            float * out = po + (row0 + hh)*DV;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    out[c*SA_DCH + (2*w + i)*16 + l16] = o[c][i][v];
                }
            }
        }
        if (w == 0 && g == 0) {
            part_ml[(int64_t) split*n_rows + row0 + l16] = make_float2(m_run, l_run);
        }
    }
#else
    GGML_UNUSED_VARS(q, k, idx, mask, dst, part_o, part_ml, nbq1, nbq2, nbq3, nbk2, nbk3,
            n_head, n_q, n_kv, n_sel, scale, tiles_per_split);
    NO_DEVICE_CODE;
#endif // defined(AMD_MFMA_AVAILABLE)
}

// merge the split partials of one (token, head) row: O = sum_s O_s e^(m_s - M) / sum_s l_s e^(m_s - M)
template <int DV>
static __global__ void sparse_attn_combine(
        const float * __restrict__ part_o, const float2 * __restrict__ part_ml, float * __restrict__ dst,
        const int64_t n_rows, const int n_split) {
    const int64_t row = blockIdx.x;

    float M = -INFINITY;
    for (int sp = 0; sp < n_split; ++sp) {
        M = fmaxf(M, part_ml[sp*n_rows + row].x);
    }

    float L = 0.0f;
    for (int sp = 0; sp < n_split; ++sp) {
        const float2 ml = part_ml[sp*n_rows + row];
        L += ml.x == -INFINITY ? 0.0f : ml.y*expf(ml.x - M);
    }
    const float inv = L > 0.0f ? 1.0f/L : 0.0f;

    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        float acc = 0.0f;
        for (int sp = 0; sp < n_split; ++sp) {
            const float m = part_ml[sp*n_rows + row].x;
            if (m != -INFINITY) {
                acc += part_o[(sp*n_rows + row)*DV + d]*expf(m - M);
            }
        }
        dst[row*DV + d] = acc*inv;
    }
}

bool ggml_cuda_sparse_attn_supported(int device, const ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * idx  = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (!amd_mfma_available(ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || idx->type != GGML_TYPE_I32 || mask->type != GGML_TYPE_F32) {
        return false;
    }
    // instantiated for the GLM-5.3 DSA shape only: 512-wide latent rows, heads in groups of 16
    const int64_t D_v = dst->ne[0];
    if (q->ne[0] != 512 || D_v != 512 || q->ne[1] % SA_HEADS != 0) {
        return false;
    }
    // 16-byte row loads and float4 Q loads
    if (k->nb[2] % 16 != 0 || k->nb[3] % 16 != 0 || q->nb[1] % 16 != 0 || q->nb[2] % 16 != 0 || q->nb[3] % 16 != 0) {
        return false;
    }
    return ggml_is_contiguous(idx) && ggml_is_contiguous(mask) && ggml_is_contiguous(dst);
}

void ggml_cuda_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * idx  = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(ggml_cuda_sparse_attn_supported(ctx.device, dst));

    constexpr int D  = 512;
    constexpr int DV = 512;

    const float scale  = ggml_get_op_params_f32(dst, 0);
    const int   n_head = q->ne[1];
    const int   n_q    = q->ne[2];
    const int   n_ns   = q->ne[3];
    const int   n_kv   = k->ne[2];
    const int   n_sel  = idx->ne[0];
    const int   n_tok  = n_q*n_ns;

    const int n_tiles = (n_sel + SA_TILE - 1)/SA_TILE;
    const int n_hg    = n_head/SA_HEADS;

    // few queries (decode, speculative verify): split the cell range so that every CU gets work
    const int n_cu   = ggml_cuda_info().devices[ctx.device].nsm;
    const int want   = (2*n_cu + n_tok*n_hg - 1)/(n_tok*n_hg);
    int tiles_per_split = (n_tiles + std::max(1, want) - 1)/std::max(1, want);
    tiles_per_split = std::max(1, tiles_per_split);
    const int n_split = (n_tiles + tiles_per_split - 1)/tiles_per_split;

    cudaStream_t stream = ctx.stream();

    const dim3 grid(n_tok, n_hg, n_split);
    const dim3 block(SA_NW*64, 1, 1);

    if (n_split == 1) {
        sparse_attn_mfma<D, DV><<<grid, block, 0, stream>>>(
                (const float *) q->data, (const char *) k->data, (const int32_t *) idx->data, (const float *) mask->data,
                (float *) dst->data, nullptr, nullptr,
                q->nb[1], q->nb[2], q->nb[3], k->nb[2], k->nb[3],
                n_head, n_q, n_kv, n_sel, scale, tiles_per_split);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int64_t n_rows = (int64_t) n_tok*n_head;
    ggml_cuda_pool_alloc<float>  part_o (ctx.pool(), (size_t) n_split*n_rows*DV);
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool(), (size_t) n_split*n_rows);

    sparse_attn_mfma<D, DV><<<grid, block, 0, stream>>>(
            (const float *) q->data, (const char *) k->data, (const int32_t *) idx->data, (const float *) mask->data,
            (float *) dst->data, part_o.get(), part_ml.get(),
            q->nb[1], q->nb[2], q->nb[3], k->nb[2], k->nb[3],
            n_head, n_q, n_kv, n_sel, scale, tiles_per_split);
    CUDA_CHECK(cudaGetLastError());

    sparse_attn_combine<DV><<<n_rows, 128, 0, stream>>>(part_o.get(), part_ml.get(), (float *) dst->data, n_rows, n_split);
    CUDA_CHECK(cudaGetLastError());
}
