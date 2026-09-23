#include "mmvq-moe.cuh"
#include "unary.cuh"

#include <cstdlib>
#include <type_traits>

// Few-token MUL_MAT_ID (MoE decode and the MTP verify batch) for gfx908, tolerance class.
//
// mul_mat_vec_q_moe runs one warp per (token, expert slot) through vec_dot_*_q8_1: every 2-int weight fragment is
// re-unpacked (nibbles, 6-bit scales, fp16 scales) and re-scaled in float for every token, ~100 VALU instructions per
// (row, token) for ~10 dp4a of real work, and every token re-reads its experts. At 3 tokens the kernel is VALU/latency
// bound (~49% VALU busy, ~640 GB/s) rather than bandwidth bound.
//
// Here a lane owns whole quant blocks of a row (Q4_K: a 64-weight half-superblock pair, 16-byte loads of header and
// quants; Q5_1: one 32-weight block), unpacks them once, and applies them to every token routed to that expert:
// per token only the dp4a chain and two FMAs per 32 weights remain. Each 32-weight block is scaled after its integer
// dot product, so the summation order differs from the reference mmvq (tolerance class, KL-gated); the Q4_K min term
// uses d8*sum(q8) from the exact integer sum as the reference does, Q5_1 uses ds.y as its reference does. The batch's routing is read with one load + ballot: the first
// (token, slot) pair that selects an expert computes it for all later tokens, the other pairs exit, so each distinct
// expert is read once. GGML_MOE_V2=0 disables the path; GGML_MOE_DEDUP_STATS=1 counts distinct experts.

static constexpr int moe_warp_size = 64;

// routing of the whole batch in one load (lane l = token*n_used + slot < 64) and one ballot for "selects expert e":
// returns false if an earlier token already selects e (that pair's block does the work), else fills sl[t] with the
// slot of e in token t's routing (-1 if token t does not use it)
template <int NTMAX>
static __device__ __forceinline__ bool moe_route(const int32_t * __restrict__ ids, const int n_used, const int ncols_dst,
        const int ids_stride, int & e, int (&sl)[NTMAX]) {
    const int lane = threadIdx.x;
    const int tok  = blockIdx.y / n_used;
    int idv = -1;
    if (lane < n_used*ncols_dst) {
        const int lt = lane / n_used;
        idv = ids[lane - lt*n_used + lt*ids_stride];
    }
    e = __builtin_amdgcn_readlane(idv, blockIdx.y);
    const uint64_t m = __ballot(idv == e);
    if (m & ((uint64_t{1} << (tok*n_used)) - 1)) {
        return false;
    }
#pragma unroll
    for (int t = 0; t < NTMAX; ++t) {
        const uint64_t mt = t < ncols_dst ? (m >> (t*n_used)) & ((uint64_t{1} << n_used) - 1) : 0;
        sl[t] = (t >= tok && mt) ? __builtin_ctzll(mt) : -1;
    }
    return true;
}

static __device__ __forceinline__ void moe_count(unsigned int * stats, const int n_used, const int ncols_dst) {
    if (stats && blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        // [0] distinct experts, [1] (token, slot) pairs, [2 + ncols_dst - 1] calls per batch size
        atomicAdd(&stats[0], 1u);
        if (blockIdx.y == 0) {
            atomicAdd(&stats[1], (unsigned int) (n_used*ncols_dst));
            atomicAdd(&stats[2 + ncols_dst - 1], 1u);
        }
    }
}

// the activations (q8_1 rows) of every token routed to e, staged in shared memory: one cooperative copy. With S8, also
// each block's d8*sum(q8) in fp32 from the exact integer sum (the reference Q4_K dot product does not use the fp16 ds.y)
template <int NTMAX, int NI, bool S8 = false, int NBY = 1>
static __device__ __forceinline__ void moe_stage_y(int (&y_sh)[NTMAX][NI], const void * __restrict__ vy, const int (&sl)[NTMAX],
        const int nchannels_y, const int stride_channel_y, const int stride_col_y, float (*s8_sh)[NBY] = nullptr) {
    const int nthreads = blockDim.x*blockDim.y;
    const int tid      = threadIdx.y*blockDim.x + threadIdx.x;
#pragma unroll
    for (int t = 0; t < NTMAX; ++t) {
        if (sl[t] < 0) {
            continue;
        }
        const int * ysrc = (const int *) ((const block_q8_1 *) vy + (sl[t] % nchannels_y)*stride_channel_y + t*stride_col_y);
        for (int idx = tid; idx < NI; idx += nthreads) {
            y_sh[t][idx] = ysrc[idx];
        }
        if constexpr (S8) {
            for (int ib = tid; ib < NBY; ib += nthreads) {
                const block_q8_1 * b = (const block_q8_1 *) ysrc + ib;
                const int * q = (const int *) b->qs;
                int sum = 0;
#pragma unroll
                for (int k = 0; k < QK8_1/4; ++k) {
                    sum = ggml_cuda_dp4a(0x01010101, q[k], sum);
                }
                s8_sh[t][ib] = __low2float(b->ds) * sum;
            }
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------------------------------
// Q4_K (gate/up, optionally both with SWIGLU): 8 lanes per row, a wave covers 8 rows, a lane owns pairs p = l8 + 8k of
// the row (pair = 32 qs bytes = sub-blocks 2(p%4), 2(p%4)+1 of superblock p/4). NSB superblocks per row, NSB even.
struct moe_q4K_pair {
    int   q[8];                 // 32 qs bytes
    float dsc_lo, dsc_hi;       // d*sc
    float dmm_lo, dmm_hi;       // dmin*m
};

static __device__ __forceinline__ void moe_q4K_load(const block_q4_K * __restrict__ b, const int pp, moe_q4K_pair & w) {
    const int4 h  = *((const int4 *) b);                        // d, dmin, scales[12]
    const int4 q0 = *((const int4 *) (b->qs + 32*pp));
    const int4 q1 = *((const int4 *) (b->qs + 32*pp + 16));
    w.q[0] = q0.x; w.q[1] = q0.y; w.q[2] = q0.z; w.q[3] = q0.w;
    w.q[4] = q1.x; w.q[5] = q1.y; w.q[6] = q1.z; w.q[7] = q1.w;

    const float2 dm = __half22float2(*((const half2 *) &h.x));
    const int s03 = h.y, s47 = h.z, s811 = h.w;               // scales[0..3], [4..7], [8..11]
    const int j0 = 2*pp;                                        // get_scale_min_k4 for j0 and j0 + 1
    int sc[2], mn[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int j = j0 + s;
        if (j < 4) {
            sc[s] = (s03 >> (8*j)) & 63;
            mn[s] = (s47 >> (8*j)) & 63;
        } else {
            const int b8  = (s811 >> (8*(j - 4))) & 0xFF;       // scales[j + 4]
            const int bm4 = (s03  >> (8*(j - 4))) & 0xFF;       // scales[j - 4]
            const int b0  = (s47  >> (8*(j - 4))) & 0xFF;       // scales[j]
            sc[s] = (b8 & 0xF) | ((bm4 >> 6) << 4);
            mn[s] = (b8 >>  4) | ((b0  >> 6) << 4);
        }
    }
    w.dsc_lo = dm.x*sc[0];
    w.dsc_hi = dm.x*sc[1];
    w.dmm_lo = dm.y*mn[0];
    w.dmm_hi = dm.y*mn[1];
}

static __device__ __forceinline__ float moe_q4K_dot(const moe_q4K_pair & w, const block_q8_1 * __restrict__ ylo, const float * s8) {
    const block_q8_1 * yhi = ylo + 1;
    const int * ul = (const int *) ylo->qs;
    const int * uh = (const int *) yhi->qs;
    int dl = 0, dh = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        dl = ggml_cuda_dp4a( w.q[k]       & 0x0F0F0F0F, ul[k], dl);
        dh = ggml_cuda_dp4a((w.q[k] >> 4) & 0x0F0F0F0F, uh[k], dh);
    }
    const float d8l = __low2float(ylo->ds);
    const float d8h = __low2float(yhi->ds);
    return (w.dsc_lo*d8l*dl - w.dmm_lo*s8[0]) + (w.dsc_hi*d8h*dh - w.dmm_hi*s8[1]);
}

template <int NSB, int NW, int RG, int NTMAX, bool GLU>
static __global__ void __launch_bounds__(NW*moe_warp_size, 1) mmvq_moe_q4K(
        const void * __restrict__ vx, const void * __restrict__ vgate, const void * __restrict__ vy,
        const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int nrows, const int stride_row_x, const int stride_channel_x,
        const int stride_col_y, const int stride_channel_y, const int nchannels_y,
        const int stride_col_dst, const int stride_channel_dst,
        const int n_used, const int ncols_dst, const int ids_stride, unsigned int * __restrict__ stats) {
    static_assert(NSB % 2 == 0, "8 lanes per row need an even superblock count");
    constexpr int NP  = NSB/2;                              // pairs per lane and row
    constexpr int NBY = NSB*(QK_K/QK8_1);
    constexpr int NI  = NBY*(int) (sizeof(block_q8_1)/sizeof(int));
    __shared__ int   y_sh[NTMAX][NI];
    __shared__ float s8_sh[NTMAX][NBY];

    int e;
    int sl[NTMAX];
    if (!moe_route<NTMAX>(ids, n_used, ncols_dst, ids_stride, e, sl)) {
        return;
    }
    moe_count(stats, n_used, ncols_dst);

    // the wave does RG groups of 8 rows; its work items i = g*NP + k (group g, pair k) are streamed with a
    // one-item register prefetch so the loads of item i+1 are in flight while item i is applied to every token
    const int lane = threadIdx.x;
    const int l8   = lane % 8;
    const int row_base = (blockIdx.x*NW + threadIdx.y)*(8*RG) + lane/8;
    const block_q4_K * bx = (const block_q4_K *) vx    + e*stride_channel_x;
    const block_q4_K * bg = (const block_q4_K *) vgate + e*stride_channel_x;

    auto load_item = [&](const int i, moe_q4K_pair & wu, moe_q4K_pair & wg) {
        const int row = row_base + (i / NP)*8;
        const int p   = l8 + 8*(i % NP);
        if (row < nrows) {
            moe_q4K_load(bx + row*stride_row_x + p/4, p%4, wu);
            if constexpr (GLU) {
                moe_q4K_load(bg + row*stride_row_x + p/4, p%4, wg);
            }
        }
    };

    moe_q4K_pair cu, cg, nu, ng;
    load_item(0, cu, cg);
    moe_stage_y<NTMAX, NI, true, NBY>(y_sh, vy, sl, nchannels_y, stride_channel_y, stride_col_y, s8_sh);

    float acc[NTMAX], accg[NTMAX];
#pragma unroll
    for (int t = 0; t < NTMAX; ++t) {
        acc[t]  = 0.0f;
        accg[t] = 0.0f;
    }
#pragma unroll 1
    for (int i = 0; i < RG*NP; ++i) {
        if (i + 1 < RG*NP) {
            load_item(i + 1, nu, ng);
        }
        const int k = i % NP;
        const int p = l8 + 8*k;
#pragma unroll
        for (int t = 0; t < NTMAX; ++t) {
            if (sl[t] < 0) {
                continue;
            }
            const int ib = (p/4)*(QK_K/QK8_1) + 2*(p%4);
            const block_q8_1 * ylo = (const block_q8_1 *) y_sh[t] + ib;
            acc[t] += moe_q4K_dot(cu, ylo, &s8_sh[t][ib]);
            if constexpr (GLU) {
                accg[t] += moe_q4K_dot(cg, ylo, &s8_sh[t][ib]);
            }
        }
        if (k == NP - 1) {
            // group done: sum the 8 lanes of each row and write
            const int row = row_base + (i / NP)*8;
#pragma unroll
            for (int t = 0; t < NTMAX; ++t) {
                if (sl[t] < 0) {
                    continue;
                }
#pragma unroll
                for (int off = 1; off < 8; off <<= 1) {
                    acc[t] += __shfl_xor_sync(0xffffffff, acc[t], off, moe_warp_size);
                    if constexpr (GLU) {
                        accg[t] += __shfl_xor_sync(0xffffffff, accg[t], off, moe_warp_size);
                    }
                }
                if (l8 == 0 && row < nrows) {
                    float r = acc[t];
                    if constexpr (GLU) {
                        r *= ggml_cuda_op_silu_single(accg[t]);
                    }
                    dst[sl[t]*stride_channel_dst + t*stride_col_dst + row] = r;
                }
                acc[t]  = 0.0f;
                accg[t] = 0.0f;
            }
        }
        cu = nu;
        cg = ng;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Q5_1 (down): 4 lanes per row, a wave covers 16 rows, a lane owns blocks b = l4 + 4k (NB blocks per row, NB % 4 == 0).
struct moe_q51_blk {
    int   v[8];   // 5-bit quants as bytes: v[0..3] = elements 0..15, v[4..7] = elements 16..31
    float d, m;
};

static __device__ __forceinline__ void moe_q51_load(const block_q5_1 * __restrict__ b, moe_q51_blk & w) {
    const int2 a  = *((const int2 *) b);                 // dm, qh
    const int2 q0 = *((const int2 *) b->qs);
    const int2 q1 = *((const int2 *) (b->qs + 8));
    const float2 dm = __half22float2(*((const half2 *) &a.x));
    w.d = dm.x;
    w.m = dm.y;
    const int qs[4] = { q0.x, q0.y, q1.x, q1.y };
    const int qh = a.y;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int vh = qh >> (4*k);
        int lo = (qs[k] >> 0) & 0x0F0F0F0F;
        lo |= (vh <<  4) & 0x00000010;
        lo |= (vh << 11) & 0x00001000;
        lo |= (vh << 18) & 0x00100000;
        lo |= (vh << 25) & 0x10000000;
        int hi = (qs[k] >> 4) & 0x0F0F0F0F;
        hi |= (vh >> 12) & 0x00000010;
        hi |= (vh >>  5) & 0x00001000;
        hi |= (vh <<  2) & 0x00100000;
        hi |= (vh <<  9) & 0x10000000;
        w.v[k]     = lo;
        w.v[k + 4] = hi;
    }
}

static __device__ __forceinline__ float moe_q51_dot(const moe_q51_blk & w, const block_q8_1 * __restrict__ yb) {
    const int * u = (const int *) yb->qs;
    int s = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        s = ggml_cuda_dp4a(w.v[k], u[k], s);
    }
    const float2 ds = __half22float2(yb->ds);
    return w.d*ds.x*s + w.m*ds.y;
}

template <int NB, int NW, int RG, int NTMAX>
static __global__ void __launch_bounds__(NW*moe_warp_size, 1) mmvq_moe_q51(
        const void * __restrict__ vx, const void * __restrict__ vy,
        const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int nrows, const int stride_row_x, const int stride_channel_x,
        const int stride_col_y, const int stride_channel_y, const int nchannels_y,
        const int stride_col_dst, const int stride_channel_dst,
        const int n_used, const int ncols_dst, const int ids_stride, unsigned int * __restrict__ stats) {
    static_assert(NB % 4 == 0, "4 lanes per row");
    constexpr int NK = NB/4;
    constexpr int NI = NB*(int) (sizeof(block_q8_1)/sizeof(int));
    __shared__ int y_sh[NTMAX][NI];

    int e;
    int sl[NTMAX];
    if (!moe_route<NTMAX>(ids, n_used, ncols_dst, ids_stride, e, sl)) {
        return;
    }
    moe_count(stats, n_used, ncols_dst);

    // RG groups of 16 rows per wave, items i = g*NK + k streamed with a one-item prefetch (see mmvq_moe_q4K)
    const int lane = threadIdx.x;
    const int l4   = lane % 4;
    const int row_base = (blockIdx.x*NW + threadIdx.y)*(16*RG) + lane/4;
    const block_q5_1 * bx = (const block_q5_1 *) vx + e*stride_channel_x;

    auto load_item = [&](const int i, moe_q51_blk & w) {
        const int row = row_base + (i / NK)*16;
        if (row < nrows) {
            moe_q51_load(bx + row*stride_row_x + l4 + 4*(i % NK), w);
        }
    };

    moe_q51_blk cw, nw;
    load_item(0, cw);
    moe_stage_y<NTMAX, NI>(y_sh, vy, sl, nchannels_y, stride_channel_y, stride_col_y);

    float acc[NTMAX];
#pragma unroll
    for (int t = 0; t < NTMAX; ++t) {
        acc[t] = 0.0f;
    }
#pragma unroll 1
    for (int i = 0; i < RG*NK; ++i) {
        if (i + 1 < RG*NK) {
            load_item(i + 1, nw);
        }
        const int k = i % NK;
#pragma unroll
        for (int t = 0; t < NTMAX; ++t) {
            if (sl[t] >= 0) {
                acc[t] += moe_q51_dot(cw, (const block_q8_1 *) y_sh[t] + l4 + 4*k);
            }
        }
        if (k == NK - 1) {
            const int row = row_base + (i / NK)*16;
#pragma unroll
            for (int t = 0; t < NTMAX; ++t) {
                if (sl[t] < 0) {
                    continue;
                }
                acc[t] += __shfl_xor_sync(0xffffffff, acc[t], 1, moe_warp_size);
                acc[t] += __shfl_xor_sync(0xffffffff, acc[t], 2, moe_warp_size);
                if (l4 == 0 && row < nrows) {
                    dst[sl[t]*stride_channel_dst + t*stride_col_dst + row] = acc[t];
                }
                acc[t] = 0.0f;
            }
        }
        cw = nw;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// GGML_MOE_DEDUP_STATS=1: per-device counters (fixed device addresses, so HIP graphs keep working), printed at exit
static unsigned int * moe_dedup_stats(const int device) {
    static const bool on = getenv("GGML_MOE_DEDUP_STATS") && atoi(getenv("GGML_MOE_DEDUP_STATS")) != 0;
    static unsigned int * ptr[GGML_CUDA_MAX_DEVICES] = { nullptr };
    if (!on) {
        return nullptr;
    }
    if (!ptr[device]) {
        CUDA_CHECK(cudaMalloc((void **) &ptr[device], 8*sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(ptr[device], 0, 8*sizeof(unsigned int)));
        static bool registered = false;
        if (!registered) {
            registered = true;
            atexit([] {
                for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
                    if (!ptr[d]) {
                        continue;
                    }
                    unsigned int h[8];
                    if (cudaSetDevice(d) != cudaSuccess || cudaMemcpy(h, ptr[d], sizeof(h), cudaMemcpyDeviceToHost) != cudaSuccess) {
                        continue;
                    }
                    fprintf(stderr, "moe_dedup_stats dev %d: distinct %u / pairs %u (%.3f), calls nt1 %u nt2 %u nt3 %u nt4 %u\n", d,
                        h[0], h[1], h[1] ? (double) h[0] / h[1] : 0.0, h[2], h[3], h[4], h[5]);
                }
            });
        }
    }
    return ptr[device];
}

bool ggml_cuda_mmvq_moe_dedup(
        ggml_type type, const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion,
        float * dst, int ncols_x, int nrows_x, int nchannels_y,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int stride_channel_x, int stride_channel_y, int stride_channel_dst,
        int ncols_dst, int n_used, int ids_stride, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    // host pass: the CDNA macro only exists in the device pass, so check the device here
    static const bool enabled = !getenv("GGML_MOE_V2") || atoi(getenv("GGML_MOE_V2")) != 0;
    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;
    if (!enabled || ncols_dst < 1 || ncols_dst > 4 || n_used*ncols_dst > 64 || !GGML_CUDA_CC_IS_CDNA(cc)) {
        return false;
    }
    if (fusion.x_bias || fusion.gate_bias || fusion.x_scale || fusion.gate_scale) {
        return false;
    }
    const bool glu = fusion.gate != nullptr;
    if (glu && fusion.glu_op != GGML_GLU_OP_SWIGLU) {
        return false;
    }
    unsigned int * stats = moe_dedup_stats(device);
    // tuning: GGML_MOE_V2_Q4K = 10*NW + RG, GGML_MOE_V2_Q51 = 10*NW + RG (NW waves per block, RG row groups per wave)
    static const int cfg_q4k = getenv("GGML_MOE_V2_Q4K") ? atoi(getenv("GGML_MOE_V2_Q4K")) : 41;
    static const int cfg_q51 = getenv("GGML_MOE_V2_Q51") ? atoi(getenv("GGML_MOE_V2_Q51")) : 41;
    switch (type) {
        case GGML_TYPE_Q4_K: {
            if (ncols_x != 10*QK_K) {
                return false;
            }
            auto launch = [&](auto nw_tag, auto rg_tag) {
                constexpr int NW = decltype(nw_tag)::value;
                constexpr int RG = decltype(rg_tag)::value;
                const dim3 block_dims(moe_warp_size, NW, 1);
                const dim3 block_nums((nrows_x + 8*RG*NW - 1) / (8*RG*NW), n_used*ncols_dst, 1);
                if (glu) {
                    mmvq_moe_q4K<10, NW, RG, 4, true><<<block_nums, block_dims, 0, stream>>>(vx, fusion.gate, vy, ids, dst,
                        nrows_x, stride_row_x, stride_channel_x, stride_col_y, stride_channel_y, nchannels_y,
                        stride_col_dst, stride_channel_dst, n_used, ncols_dst, ids_stride, stats);
                } else {
                    mmvq_moe_q4K<10, NW, RG, 4, false><<<block_nums, block_dims, 0, stream>>>(vx, vx, vy, ids, dst,
                        nrows_x, stride_row_x, stride_channel_x, stride_col_y, stride_channel_y, nchannels_y,
                        stride_col_dst, stride_channel_dst, n_used, ncols_dst, ids_stride, stats);
                }
            };
            switch (cfg_q4k) {
                case 11: launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{}); break;
                case 12: launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{}); break;
                case 22: launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{}); break;
                case 41: launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{}); break;
                case 42: launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{}); break;
                default: launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{}); break;
            }
            return true;
        }
        case GGML_TYPE_Q5_1: {
            if (ncols_x != 20*QK5_1 || glu) {
                return false;
            }
            auto launch = [&](auto nw_tag, auto rg_tag) {
                constexpr int NW = decltype(nw_tag)::value;
                constexpr int RG = decltype(rg_tag)::value;
                const dim3 block_dims(moe_warp_size, NW, 1);
                const dim3 block_nums((nrows_x + 16*RG*NW - 1) / (16*RG*NW), n_used*ncols_dst, 1);
                mmvq_moe_q51<20, NW, RG, 4><<<block_nums, block_dims, 0, stream>>>(vx, vy, ids, dst,
                    nrows_x, stride_row_x, stride_channel_x, stride_col_y, stride_channel_y, nchannels_y,
                    stride_col_dst, stride_channel_dst, n_used, ncols_dst, ids_stride, stats);
            };
            switch (cfg_q51) {
                case 11: launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{}); break;
                case 12: launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 2>{}); break;
                case 22: launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 2>{}); break;
                case 41: launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{}); break;
                case 42: launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{}); break;
                default: launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{}); break;
            }
            return true;
        }
        default:
            return false;
    }
#else
    GGML_UNUSED_VARS(type, vx, vy, ids, fusion, dst, ncols_x, nrows_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst, ncols_dst, n_used, ids_stride, stream);
    return false;
#endif // defined(GGML_USE_HIP)
}
