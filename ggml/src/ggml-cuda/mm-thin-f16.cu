// gfx908 prefill: dense Q8_0 x F32 projections with few output rows or a short K, on f16 MFMA.
//
// The rocBLAS path for these shapes runs three kernels per call: dequantize_block_q8_0_f16 (whole weight -> f16),
// convert_unary<float, half> (whole activation -> f16) and an f16 GEMM with f32 accumulation whose tiles are mostly
// padding at M = 24..512 (1-8 TFLOPS). GLM-5.3-Flash has ~370 such calls per 512-token ubatch (hc_*_fn K=16384 M=24
// x90 alone: 350 us each, for ~34 MB of activations = ~30 us of HBM).
//
// Here one kernel streams the activations once and converts both operands in registers / LDS:
//   - weights: each block dequantizes a 16R-row x KC chunk of the Q8_0 matrix to f16 in LDS with exactly the
//     arithmetic of dequantize_block_q8_0_f16 (__hmul2(half(q), d)), double-buffered, shared by its 4 waves;
//   - activations: each wave loads 16 tokens x KC straight from global memory and rounds them to f16 exactly as
//     convert_unary does (__float2half);
//   - v_mfma_f32_16x16x16f16: f16 x f16 products, f32 accumulation (as rocBLAS HSS).
// The f16 inputs are bit-identical to the rocBLAS path's; only the f32 summation order differs (MFMA tree within 16 k,
// then sequential over k, then a fixed-order sum over K splits): tolerance class, GGML_CUDA_MM_THIN_F16=0 disables.
//
// Tile: block = 4 waves x 16*TT tokens, R x 16 output rows, a K range of KS chunks. Few blocks for a tall K (hc_fn: 8
// token tiles x 1 row group) -> split K across blocks; partial sums go to a scratch buffer and a second kernel adds
// them in split order (deterministic, no atomics). Equal-batch MUL_MAT (DSA k_b / v_b, 64 heads) uses grid z.
//
// Shapes (ggml_cuda_mm_thin_f16_supported): Q8_0, N >= 129 (prefill), M <= 512 or K <= 128, plus - only with
// GGML_CUDA_MM_THIN_F16_MMQ=1 - the shapes the gfx908 dense rule sends to MMQ (TT = 2: 32 tokens per wave).
// Larger rocBLAS shapes are not taken (this kernel measured 1.2-1.4x slower than rocBLAS there).
// GPU 1, N = 512, rocBLAS path (incl. its dequant/convert) -> this kernel: hc_fn 16384->24 350 -> 84 us, 4096->128
// 165 -> 68, 4096->64 153 -> 54, 4096->512 263 -> 133, 128->8192 96 -> 72, DSA k_b 459 -> 226, v_b 614 -> 216;
// error vs a double reference equal to rocBLAS's (NMSE 7.13e-8 vs 7.13e-8). MMQ shapes (opt-in): 4096->8192 760 ->
// 648, 1536->16384 591 -> 472, 4096->12288 1107 -> 862 us, NMSE 1.4e-5 (q8_1 activations) -> 6.7e-8.

#include "mm-thin-f16.cuh"

#if defined(AMD_MFMA_AVAILABLE)
typedef _Float16 thin_half4  __attribute__((ext_vector_type(4)));
typedef float    thin_float4 __attribute__((ext_vector_type(4)));
#endif // defined(AMD_MFMA_AVAILABLE)

static constexpr int THIN_NW = 4; // waves per block, 16 tokens each

template <int R, int KC>
static constexpr int thin_lds_stride() { return KC + 8; } // halfs per LDS row: 16-byte aligned rows, banks staggered

#if defined(AMD_MFMA_AVAILABLE)
static __device__ __forceinline__ _Float16 thin_f2h(const float x) {
    // convert_unary<float, half>: __float2half (round to nearest even); reinterpret its bits, no float round trip
    const __half h = __float2half(x);
    return __builtin_bit_cast(_Float16, h);
}
#endif // defined(AMD_MFMA_AVAILABLE)

template <int R, int KC, int TT>
__launch_bounds__(THIN_NW*64, 2)
static __global__ void mm_thin_f16_q8_0(
        const char * __restrict__ W, const float * __restrict__ X, float * __restrict__ D, float * __restrict__ part,
        const int M, const int K, const int N, const int64_t row_bytes, const int64_t sx, const int64_t sd,
        const int chunks_per_split, const int nsplit, const int64_t sw2, const int64_t sx2, const int64_t sd2) {
#if defined(AMD_MFMA_AVAILABLE)
    constexpr int ROWS = 16*R;
    constexpr int LS   = thin_lds_stride<R, KC>();
    constexpr int NG   = ROWS*(KC/8) / (THIN_NW*64); // 8-value groups per thread per chunk
    constexpr int NS   = KC/16;                       // MFMA k-steps per chunk
    static_assert(NG >= 1 && ROWS*(KC/8) % (THIN_NW*64) == 0, "bad thin tile");
    static_assert(KC % 32 == 0, "KC must cover whole Q8_0 blocks");

    __shared__ __align__(16) half tile[2][ROWS*LS];

    const int tid  = threadIdx.x;
    const int w    = tid / 64;
    const int lane = tid % 64;
    const int l16  = lane % 16;
    const int g    = lane / 16;

    // TT token tiles of 16 per wave (TT = 2: each B operand read from LDS feeds two MFMAs)
    const int tok0  = blockIdx.x*(16*TT*THIN_NW) + 16*TT*w;
    const int row0  = blockIdx.y*ROWS;
    const int split = blockIdx.z % nsplit;
    const int batch = blockIdx.z / nsplit; // equal-batch MUL_MAT (e.g. GLM-5.3 DSA k_b / v_b, 64 heads); nsplit == 1 then
    W += batch*sw2;
    X += batch*sx2;
    D += batch*sd2;

    const int nchunks_total = K / KC;
    const int c0 = split*chunks_per_split;
    const int c1 = min(nchunks_total, c0 + chunks_per_split);

    // activations: this lane's token rows (clamped; rows >= N are computed but never stored)
    const float * xrow[TT];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
        xrow[tt] = X + (int64_t) min(tok0 + 16*tt + l16, N - 1)*sx;
    }

    // weight groups of this thread: group gi = tid + 256*n -> row gi/(KC/8), values 8*(gi%(KC/8)) .. +7 of the chunk
    const char * wblk[NG]; // Q8_0 block of this group in chunk 0 (scale d at +0, quants at +2)
    int          wofs[NG]; // byte offset of the group's 8 quants inside the block
    int          wlds[NG];
#pragma unroll
    for (int n = 0; n < NG; ++n) {
        const int gi  = tid + THIN_NW*64*n;
        const int r   = gi / (KC/8);
        const int sub = gi % (KC/8);
        const int row = min(row0 + r, M - 1);
        wblk[n] = W + (int64_t) row*row_bytes + (sub/4)*(int64_t) sizeof(block_q8_0);
        wofs[n] = 2 + 8*(sub % 4);
        wlds[n] = r*LS + 8*sub;
    }

    uint2 wq[NG];
    half  wd[NG];
    auto load_w = [&](const int c) {
        const int64_t off = (int64_t) c*(KC/32)*sizeof(block_q8_0);
#pragma unroll
        for (int n = 0; n < NG; ++n) {
            const char * b = wblk[n] + off;
            wq[n] = *(const uint2 *) (b + wofs[n]); // 8 quants (2-byte aligned: unaligned global loads are fine on gfx908)
            wd[n] = *(const half  *)  b;            // block scale d
        }
    };
    auto store_w = [&](const int buf) {
#pragma unroll
        for (int n = 0; n < NG; ++n) {
            const half2 d2 = __half2half2(wd[n]);
            const char * q = (const char *) &wq[n];
            half2 v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // dequantize_block_q8_0_f16: __hmul2(make_half2(qs.x, qs.y), __half2half2(d))
                v[i] = __hmul2(make_half2(q[2*i + 0], q[2*i + 1]), d2);
            }
            *(uint4 *) &tile[buf][wlds[n]] = *(const uint4 *) v;
        }
    };

    float4 xr[TT][NS];
    auto load_x = [&](const int c) {
#pragma unroll
        for (int tt = 0; tt < TT; ++tt) {
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                xr[tt][s] = *(const float4 *) (xrow[tt] + c*KC + 16*s + 4*g);
            }
        }
    };

    thin_float4 acc[TT][R];
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            acc[tt][r] = thin_float4{0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    if (c0 < c1) {
        load_w(c0);
        load_x(c0);
        store_w(0);
    }
    __syncthreads();

    for (int c = c0; c < c1; ++c) {
        const int buf = (c - c0) & 1;

        // this chunk's activations -> f16 A operands (convert_unary: __float2half)
        thin_half4 a[TT][NS];
#pragma unroll
        for (int tt = 0; tt < TT; ++tt) {
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                a[tt][s] = thin_half4{thin_f2h(xr[tt][s].x), thin_f2h(xr[tt][s].y), thin_f2h(xr[tt][s].z), thin_f2h(xr[tt][s].w)};
            }
        }
        const bool next = c + 1 < c1;
        if (next) {
            load_w(c + 1);
            load_x(c + 1);
        }

        // k-step s, lane group g: k = c*KC + 16*s + 4*g + (0..3) for both operands
#pragma unroll
        for (int s = 0; s < NS; ++s) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const thin_half4 b = *(const thin_half4 *) &tile[buf][(16*r + l16)*LS + 16*s + 4*g];
#pragma unroll
                for (int tt = 0; tt < TT; ++tt) {
                    acc[tt][r] = __builtin_amdgcn_mfma_f32_16x16x16f16(a[tt][s], b, acc[tt][r], 0, 0, 0);
                }
            }
        }

        if (next) {
            store_w(buf ^ 1);
        }
        __syncthreads();
    }

    // acc[tt][r][v] = D[token tok0 + 16tt + 4g + v][row row0 + 16r + l16]
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + 16*r + l16;
        if (row >= M) {
            continue;
        }
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int tok = tok0 + 16*tt + 4*g + v;
            if (tok >= N) {
                continue;
            }
            if (part) {
                part[((int64_t) split*N + tok)*M + row] = acc[tt][r][v];
            } else {
                D[(int64_t) tok*sd + row] = acc[tt][r][v];
            }
        }
    }
    }
#else
    GGML_UNUSED_VARS(W, X, D, part, M, K, N, row_bytes, sx, sd, chunks_per_split, nsplit, sw2, sx2, sd2);
    NO_DEVICE_CODE;
#endif // defined(AMD_MFMA_AVAILABLE)
}

// D[t][m] = sum over splits s = 0..ns-1 (in order) of part[s][t][m]
static __global__ void mm_thin_f16_reduce(const float * __restrict__ part, float * __restrict__ D,
        const int M, const int N, const int ns, const int64_t sd) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (int64_t) N*M) {
        return;
    }
    const int t = i / M;
    const int m = i % M;
    float s = part[i];
    for (int k = 1; k < ns; ++k) {
        s += part[(int64_t) k*N*M + i];
    }
    D[(int64_t) t*sd + m] = s;
}

static int64_t thin_env(const char * name, const int64_t def) {
    const char * e = getenv(name);
    return e ? atoll(e) : def;
}

bool ggml_cuda_mm_thin_f16_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    static const bool    on     = thin_env("GGML_CUDA_MM_THIN_F16", 1) != 0;
    static const int64_t max_m  = thin_env("GGML_CUDA_MM_THIN_F16_MAX_M", 512);  // thin: M <= max_m (any K)
    static const int64_t max_k  = thin_env("GGML_CUDA_MM_THIN_F16_MAX_K", 128);  // short K: K <= max_k (any M)
    static const int64_t min_n  = thin_env("GGML_CUDA_MM_THIN_F16_MIN_N", 129);  // prefill only (decode: mmvq)
    if (!on || !GGML_CUDA_CC_IS_CDNA1(cc)) {
        return false;
    }
    // Shapes that the gfx908 dense rule sends to MMQ (M >= GGML_MMQ_DENSE_MIN_M, K % 256 == 0: GLM-5.3 KDA q/k/v
    // 4096->8192, DSA q_b 1536->16384, dense gate/up 4096->12288) stay on MMQ (q8_1 activations) unless
    // GGML_CUDA_MM_THIN_F16_MMQ=1: then they run here with 32 tokens per wave (15-22% faster than MMQ at N=512, and
    // ~200x lower error vs an exact reference: f16 instead of q8_1 activations; tolerance class). Large shapes that go
    // to rocBLAS stay there (this kernel measured 1.2-1.4x slower than rocBLAS on them).
    static const int64_t mmq_min_m = thin_env("GGML_MMQ_DENSE_MIN_M", 6144);
    static const bool    take_mmq  = thin_env("GGML_CUDA_MM_THIN_F16_MMQ", 0) != 0;
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    const bool mmq_shape = mmq_min_m > 0 && M >= mmq_min_m && K % 256 == 0 && src0->type == GGML_TYPE_Q8_0
        && src0->ne[2] == 1 && src1->ne[2] == 1;
    if (mmq_shape && !take_mmq) {
        return false;
    }
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && (M <= max_m || K <= max_k || mmq_shape) && N >= min_n && K % 64 == 0
        && src0->ne[2] == src1->ne[2] && src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[2] == src1->ne[2]
        && src0->nb[0] == ggml_type_size(src0->type) && src0->nb[1] == ggml_row_size(src0->type, K)
        && src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && src1->nb[2] % 16 == 0
        && dst->nb[0] == sizeof(float) && (uintptr_t) src1->data % 16 == 0
        && N < INT32_MAX/4 && M*N < INT32_MAX;
}

template <int R, int KC, int TT>
static void mm_thin_f16_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int M = src0->ne[1];
    const int K = src0->ne[0];
    const int N = src1->ne[1];
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    cudaStream_t stream = ctx.stream();

    const int ntok  = (N + 16*TT*THIN_NW - 1) / (16*TT*THIN_NW);
    const int nrow  = (M + 16*R - 1) / (16*R);
    const int nchk  = K / KC;
    // fewer output tiles than CUs: split K until there are ~target blocks (3 per CU), keeping >= min_chunks chunks per split
    static const int64_t target_per_cu = thin_env("GGML_CUDA_MM_THIN_F16_BLOCKS_PER_CU", 3);
    static const int64_t min_chunks    = thin_env("GGML_CUDA_MM_THIN_F16_MIN_CHUNKS", 4);
    const int64_t target = target_per_cu*nsm;
    const int nb = src0->ne[2]; // equal batches (no K split then)
    int ns = 1;
    if (nb == 1 && (int64_t) ntok*nrow < nsm) { // enough output tiles for every CU: no split (partials cost 2x the output traffic)
        ns = (int) std::min<int64_t>((target + ntok*nrow - 1)/(ntok*nrow), std::max<int64_t>(1, nchk/min_chunks));
    }
    const int cps = (nchk + ns - 1) / ns;
    ns = (nchk + cps - 1) / cps;

    const float * X = (const float *) src1->data;
    float       * D = (float *) dst->data;
    const int64_t sx = src1->nb[1] / sizeof(float);
    const int64_t sd = dst->nb[1]  / sizeof(float);
    const int64_t row_bytes = src0->nb[1];

    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (ns > 1) {
        part.alloc((size_t) ns*N*M);
    }
    const dim3 grid(ntok, nrow, ns*nb);
    mm_thin_f16_q8_0<R, KC, TT><<<grid, THIN_NW*64, 0, stream>>>(
        (const char *) src0->data, X, D, ns > 1 ? part.ptr : nullptr, M, K, N, row_bytes, sx, sd, cps, ns,
        (int64_t) src0->nb[2], (int64_t) (src1->nb[2]/sizeof(float)), (int64_t) (dst->nb[2]/sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
    if (ns > 1) {
        const int64_t n = (int64_t) N*M;
        mm_thin_f16_reduce<<<(n + 255)/256, 256, 0, stream>>>(part.ptr, D, M, N, ns, sd);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_mm_thin_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t M = src0->ne[1];
    // large M with a long K (the MMQ-rule shapes): 32 tokens per wave (each LDS B operand feeds two MFMAs);
    // GGML_CUDA_MM_THIN_F16_TT=1 keeps 16
    static const int64_t tt = thin_env("GGML_CUDA_MM_THIN_F16_TT", 2);
    if (M <= 32) {
        mm_thin_f16_launch<2, 64, 1>(ctx, src0, src1, dst);
    } else if (M <= 64) {
        mm_thin_f16_launch<4, 64, 1>(ctx, src0, src1, dst);
    } else if (tt == 2 && M > 512 && src0->ne[0] > 128) {
        mm_thin_f16_launch<8, 32, 2>(ctx, src0, src1, dst);
    } else {
        mm_thin_f16_launch<8, 32, 1>(ctx, src0, src1, dst);
    }
}
