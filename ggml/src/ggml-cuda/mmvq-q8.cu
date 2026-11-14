#include "mmvq-q8.cuh"
#include "reduce-dpp.cuh"
#include "vecdotq.cuh"
#include "unary.cuh"

#include <cstdlib>
#include <cstring>

// mul_mat_vec_q<GGML_TYPE_Q8_0, ncols_dst> on the GCN parameter table: nwarps = 2, rows_per_block = 1 (ncols 1) or 2,
// blocks_per_iter = vdr*nwarps*warp_size/qi = 32, lane (tid = 64*ty + tx) owns quant blocks kbx = tid/4 + it*32 at int
// offset kqs = 2*(tid%4), accumulates vec_dot_q8_0_q8_1 over its kbx in order, warp 1 hands its partials to warp 0 through
// shared memory, warp 0 does warp_reduce_sum<64>. This kernel replays exactly that; it only changes WHEN the loads are
// issued (all up front, the row's NB blocks known at compile time) and HOW (one 8-byte load for the lane's two ints,
// which the 34-byte Q8_0 block leaves 2-byte aligned; CDNA global loads accept that).
// NW = warps per block of the replayed table entry (GCN: 2 for 1..4 columns, 1 for 5..8), which fixes the lane ->
// block mapping (kbx = tid/4 + it*bpi, bpi = 16*NW) and whether warp 1 hands its partials over shared memory; RPB is a
// tuning knob (a row's order does not depend on it). CH = iterations whose loads are issued together (all of them when
// CH >= NIT): at 5..8 columns the full set does not fit in registers; chunking changes when loads are issued, never the
// per-accumulator order.
// GLU (fuse-moe C): vx = up, vg = gate of a shared-expert pair, both rows run the same lane sequence as their own
// unfused launches, then dst = swiglu_clamp(gate, up) as the GLU kernel computes it (ggml_cuda_op_swiglu_clamp_single).
template <int NB, int ncols, int RPB, int NW = 2, int CH = 1 << 20, bool GLU = false>
static __global__ void __launch_bounds__(64*NW, 1) mmvq_q8_0_v2(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int nrows, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const void * __restrict__ vg, const float glu_limit) {
    constexpr int warp_size = 64;
    constexpr int nwarps    = NW;
    constexpr int qi        = QI8_0;                 // 8
    constexpr int vdr       = VDR_Q8_0_Q8_1_MMVQ;    // 2
    constexpr int bpi       = vdr*nwarps*warp_size/qi;   // blocks per iteration: 32 (2 warps) or 16 (1 warp)
    constexpr int NIT       = (NB + bpi - 1)/bpi;
    constexpr int C         = CH < NIT ? CH : NIT;   // iterations per load chunk

    const int tid  = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = RPB*blockIdx.x;
    const int kb_l = tid/(qi/vdr);
    const int kqs  = vdr*(tid % (qi/vdr));

    float tmp[ncols][RPB] = {{0.0f}};
    float tmpg[ncols][RPB] = {{0.0f}};

#pragma unroll
    for (int it0 = 0; it0 < NIT; it0 += C) {
        // loads: weights (2 ints + d per row and iteration), activations (2 ints + d per column and iteration)
        int  v[C][RPB][vdr];
        half dw[C][RPB];
        int  vgt[C][RPB][vdr];
        half dwg[C][RPB];
        int  u[C][ncols][vdr];
        half du[C][ncols];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const int kbx = kb_l + (it0 + c)*bpi;
            if (it0 + c < NIT && kbx < NB) {
#pragma unroll
                for (int i = 0; i < RPB; ++i) {
                    const block_q8_0 * bq = (const block_q8_0 *) vx + (row0 + i)*stride_row_x + kbx;
                    if (RPB == 1 || row0 + i < nrows) {
                        int2 q;
                        __builtin_memcpy(&q, bq->qs + 4*kqs, sizeof(q));   // = get_int_b2(bq->qs, kqs), get_int_b2(bq->qs, kqs+1)
                        v[c][i][0] = q.x;
                        v[c][i][1] = q.y;
                        dw[c][i]   = bq->d;
                        if constexpr (GLU) {
                            const block_q8_0 * bg = (const block_q8_0 *) vg + (row0 + i)*stride_row_x + kbx;
                            int2 qg;
                            __builtin_memcpy(&qg, bg->qs + 4*kqs, sizeof(qg));
                            vgt[c][i][0] = qg.x;
                            vgt[c][i][1] = qg.y;
                            dwg[c][i]    = bg->d;
                        }
                    } else {
                        v[c][i][0] = 0; v[c][i][1] = 0; dw[c][i] = __float2half(0.0f);
                        if constexpr (GLU) {
                            vgt[c][i][0] = 0; vgt[c][i][1] = 0; dwg[c][i] = __float2half(0.0f);
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    const block_q8_1 * by = (const block_q8_1 *) vy + j*stride_col_y + kbx;
                    u[c][j][0] = get_int_b4(by->qs, kqs);
                    u[c][j][1] = get_int_b4(by->qs, kqs + 1);
                    du[c][j]   = __low2half(by->ds);
                }
            }
        }

#pragma unroll
        for (int c = 0; c < C; ++c) {
            const int kbx = kb_l + (it0 + c)*bpi;
            if (it0 + c < NIT && kbx < NB) {
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
#pragma unroll
                    for (int i = 0; i < RPB; ++i) {
                        tmp[j][i] += vec_dot_q8_0_q8_1_impl<float, vdr>(v[c][i], u[c][j], dw[c][i], du[c][j]);
                        if constexpr (GLU) {
                            tmpg[j][i] += vec_dot_q8_0_q8_1_impl<float, vdr>(vgt[c][i], u[c][j], dwg[c][i], du[c][j]);
                        }
                    }
                }
            }
        }
    }

    if constexpr (nwarps > 1) {
        __shared__ float tmp_shared[nwarps-1][ncols][RPB][warp_size];
        __shared__ float tmpg_shared[GLU ? nwarps-1 : 1][GLU ? ncols : 1][GLU ? RPB : 1][warp_size];
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
#pragma unroll
                for (int i = 0; i < RPB; ++i) {
                    tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                    if constexpr (GLU) {
                        tmpg_shared[threadIdx.y-1][j][i][threadIdx.x] = tmpg[j][i];
                    }
                }
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int i = 0; i < RPB; ++i) {
                tmp[j][i] += tmp_shared[0][j][i][threadIdx.x];
                if constexpr (GLU) {
                    tmpg[j][i] += tmpg_shared[0][j][i][threadIdx.x];
                }
            }
        }
    }
    // warp_reduce_sum<64> per accumulator, level by level (same xor tree, exact)
#pragma unroll
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int i = 0; i < RPB; ++i) {
                tmp[j][i] += ggml_cuda_shfl_xor_td<warp_size>(tmp[j][i], offset);
                if constexpr (GLU) {
                    tmpg[j][i] += ggml_cuda_shfl_xor_td<warp_size>(tmpg[j][i], offset);
                }
            }
        }
    }
    dst += row0;
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i = 0; i < RPB; ++i) {
            if ((int) threadIdx.x == i && (RPB == 1 || row0 + i < nrows)) {
                if constexpr (GLU) {
                    dst[j*stride_col_dst + i] = ggml_cuda_op_swiglu_clamp_single(tmpg[j][i], tmp[j][i], glu_limit);
                } else {
                    dst[j*stride_col_dst + i] = tmp[j][i];
                }
            }
        }
    }
}

// GLU launches (fuse-moe C) go through the same row / chunk selection as the plain ones (thread-local vg and limit)
static thread_local const void * mmvq_q8_glu_gate  = nullptr;
static thread_local float        mmvq_q8_glu_limit = 0.0f;
// rows per block of the GLU launch (exact for any value): two matrices' loads up front need registers; MI100, K = 4096 x
// 2048 rows, us, gate + up + GLU unfused -> fused: 2 columns 32.4 -> 21.8, 3: 34.6 -> 25.2, 4: 57.9 -> 28.9, 6 (chunk 1):
// 43.8 -> 34.6 at 1 row per block; 2 or 4 rows spill (4 columns, 4 rows: 97 us). GGML_CUDA_FUSE_MOE_C_RPB overrides.
static int mmvq_q8_glu_rpb() {
    static const int r = getenv("GGML_CUDA_FUSE_MOE_C_RPB") ? atoi(getenv("GGML_CUDA_FUSE_MOE_C_RPB")) : 1;
    return r;
}

template <int NB, int ncols, int RPB, int NW = 2, int CH = 1 << 20>
static void mmvq_q8_0_v2_launch(const void * vx, const void * vy, float * dst, int nrows, int stride_row_x, int stride_col_y,
        int stride_col_dst, cudaStream_t stream) {
    const dim3 block_nums((nrows + RPB - 1)/RPB, 1, 1);
    const dim3 block_dims(64, NW, 1);
    if (NB == 128 && mmvq_q8_glu_gate != nullptr) {   // GLU only instantiated for K = 4096 (GLM shared expert)
        mmvq_q8_0_v2<NB, ncols, RPB, NW, CH, true><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, stride_row_x,
            stride_col_y, stride_col_dst, mmvq_q8_glu_gate, mmvq_q8_glu_limit);
    } else {
        mmvq_q8_0_v2<NB, ncols, RPB, NW, CH><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, stride_row_x, stride_col_y,
            stride_col_dst, nullptr, 0.0f);
    }
    CUDA_CHECK(cudaGetLastError());
}

// rows per block: a row's accumulation order does not depend on it (each row is one lane-sequence over its blocks),
// so it is a pure tuning knob. GGML_MMVQ_Q8_RPB=<n> overrides the defaults (tuning only).
template <int NB, int ncols>
static void mmvq_q8_0_v2_rpb(int rpb, const void * vx, const void * vy, float * dst, int nrows, int stride_row_x,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    // long rows, 2..4 columns: the loads are issued c iterations at a time (exact: only the load timing changes);
    // default 1 for K >= 12288, where all loads up front run out of registers (MI100, 3 columns, us: K = 16384
    // mul_mat_vec_q 105.6, all up front 194.7, chunk 1 87.9; K = 12288 80.5 / 76.0 / 67.4; K = 8192 50.1 all up front,
    // 48.6 chunk 1). GGML_MMVQ_Q8_CH24=<c> overrides, 0 = all up front
    if constexpr (NB >= 256 && ncols > 1) {
        static const int ch_env = getenv("GGML_MMVQ_Q8_CH24") ? atoi(getenv("GGML_MMVQ_Q8_CH24")) : -1;
        const int ch = ch_env >= 0 ? ch_env : (NB >= 384 ? 1 : 0);
#define MMVQ_Q8_CH24(R, C) mmvq_q8_0_v2_launch<NB, ncols, R, 2, C>(vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream)
        if (ch > 0) {
            if (rpb == 4) {
                ch == 1 ? MMVQ_Q8_CH24(4, 1) : ch == 2 ? MMVQ_Q8_CH24(4, 2) : MMVQ_Q8_CH24(4, 4);
            } else {
                ch == 1 ? MMVQ_Q8_CH24(2, 1) : ch == 2 ? MMVQ_Q8_CH24(2, 2) : MMVQ_Q8_CH24(2, 4);
            }
            return;
        }
#undef MMVQ_Q8_CH24
    }
    switch (rpb) {
        case 1:  mmvq_q8_0_v2_launch<NB, ncols, 1>(vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 4:  mmvq_q8_0_v2_launch<NB, ncols, 4>(vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 8:  mmvq_q8_0_v2_launch<NB, ncols, 8>(vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        default: mmvq_q8_0_v2_launch<NB, ncols, 2>(vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
    }
}

// 5..8 columns: the GCN table's nwarps = 1 entry (rows_per_block 2 there; any RPB is exact). GGML_MMVQ_Q8_RPB8 and
// GGML_MMVQ_Q8_CH8 override the rows per block and the load chunk (tuning only). MI100, 6 columns, us (generic ->
// 4 rows, chunk 1): 4096x8192 75.3 -> 54.7, 8192x4096 79.6 -> 56.7, 1536x16384 57.7 -> 46.2, 2048x4096 25.1 -> 19.4.
template <int NB, int ncols>
static void mmvq_q8_0_v2_wide(int nrows, const void * vx, const void * vy, float * dst, int stride_row_x, int stride_col_y,
        int stride_col_dst, cudaStream_t stream) {
    static const int rpb_env = getenv("GGML_MMVQ_Q8_RPB8") ? atoi(getenv("GGML_MMVQ_Q8_RPB8")) : 4;
    // short matrices (<= 512 rows: thin projections): GGML_MMVQ_Q8_RPB8_SMALL=<1|2|4> rows per block (tuning, exact)
    // (MI100, 6 columns, K = 4096, us, 4 rows -> 1 row per block: 64 rows 15.3 -> 10.3, 128 rows 15.4 -> 11.4, 512 rows
    // 15.9 -> 12.8; 1536 rows 18.9 -> 22.0, so only up to 512 rows; 0 disables)
    static const int rpb_small = getenv("GGML_MMVQ_Q8_RPB8_SMALL") ? atoi(getenv("GGML_MMVQ_Q8_RPB8_SMALL")) : 1;
    static const int ch  = getenv("GGML_MMVQ_Q8_CH8")  ? atoi(getenv("GGML_MMVQ_Q8_CH8"))  : 1;
    int rpb = rpb_small > 0 && nrows <= 512 ? rpb_small : rpb_env;
    if (mmvq_q8_glu_gate != nullptr) {
        rpb = mmvq_q8_glu_rpb();
    }
#define MMVQ_Q8_WIDE(R, C) mmvq_q8_0_v2_launch<NB, ncols, R, 1, C>(vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream)
    if (rpb == 1) {
        MMVQ_Q8_WIDE(1, 1);
    } else if (rpb == 4) {
        ch == 1 ? MMVQ_Q8_WIDE(4, 1) : ch == 3 ? MMVQ_Q8_WIDE(4, 3) : ch == 4 ? MMVQ_Q8_WIDE(4, 4) : MMVQ_Q8_WIDE(4, 2);
    } else if (rpb == 8) {
        ch == 1 ? MMVQ_Q8_WIDE(8, 1) : ch == 3 ? MMVQ_Q8_WIDE(8, 3) : ch == 4 ? MMVQ_Q8_WIDE(8, 4) : MMVQ_Q8_WIDE(8, 2);
    } else {
        ch == 1 ? MMVQ_Q8_WIDE(2, 1) : ch == 3 ? MMVQ_Q8_WIDE(2, 3) : ch == 4 ? MMVQ_Q8_WIDE(2, 4) : MMVQ_Q8_WIDE(2, 2);
    }
#undef MMVQ_Q8_WIDE
}

template <int NB>
static void mmvq_q8_0_v2_ncols(int ncols, const void * vx, const void * vy, float * dst, int nrows, int stride_row_x,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    static const int rpb_env = getenv("GGML_MMVQ_Q8_RPB") ? atoi(getenv("GGML_MMVQ_Q8_RPB")) : 0;
    // measured on MI100 (bit-exact for any value): 1 row per block for one column, 4 for 2..4 columns on wide
    // matrices (6144 rows x 2560: 28.3 -> 24.0 us at 3 columns, 33.4 -> 26.6 at 4), 2 for narrow ones
    const int rpb = mmvq_q8_glu_gate != nullptr ? mmvq_q8_glu_rpb() : rpb_env > 0 ? rpb_env : (ncols == 1 ? 1 : (nrows >= 2048 ? 4 : 2));
    switch (ncols) {
        case 1: mmvq_q8_0_v2_rpb<NB, 1>(rpb, vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 2: mmvq_q8_0_v2_rpb<NB, 2>(rpb, vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 3: mmvq_q8_0_v2_rpb<NB, 3>(rpb, vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 4: mmvq_q8_0_v2_rpb<NB, 4>(rpb, vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 5: mmvq_q8_0_v2_wide<NB, 5>(nrows, vx, vy, dst, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 6: mmvq_q8_0_v2_wide<NB, 6>(nrows, vx, vy, dst, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 7: mmvq_q8_0_v2_wide<NB, 7>(nrows, vx, vy, dst, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        default: mmvq_q8_0_v2_wide<NB, 8>(nrows, vx, vy, dst, stride_row_x, stride_col_y, stride_col_dst, stream); break;
    }
}


// Short rows (K = 128 / 256 / 512, i.e. NB = 4 / 8 / 16 blocks): mul_mat_vec_q's lane partition (kbx = tid/4, one loop
// iteration) leaves only 4*NB of a block's lanes with work, the rest add zeros. Here a wave packs 64/(4*NB) rows side by
// side, R times; each row keeps its lanes' terms and the xor butterfly over its 4*NB lanes, which is the part of
// warp_reduce_sum<64> that does not add zeros, so the sums are bit-identical. Channels (batched matmul) on grid.y.
template <int NB, int ncols, int R>
static __global__ void __launch_bounds__(64, 1) mmvq_q8_0_smallk(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int nrows, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst) {
    constexpr int qi  = QI8_0;                 // 8
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;    // 2
    constexpr int LPR = NB*(qi/vdr);           // lanes per row: 16 / 32 / 64
    constexpr int RW  = 64/LPR;                // rows per wave pass

    const int lane = threadIdx.x;
    const int sub  = lane/LPR;
    const int l    = lane % LPR;
    const int kbx  = l/(qi/vdr);
    const int kqs  = vdr*(l % (qi/vdr));
    const int ch   = blockIdx.y;
    const int row0 = blockIdx.x*(R*RW) + sub;

    const block_q8_0 * x = (const block_q8_0 *) vx + (ch/channel_ratio)*stride_channel_x + kbx;
    const block_q8_1 * y = (const block_q8_1 *) vy + ch*stride_channel_y + kbx;

    int  v[R][vdr];
    half dw[R];
    int  u[ncols][vdr];
    half du[ncols];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r*RW;
        if (row < nrows) {
            const block_q8_0 * bq = x + row*stride_row_x;
            int2 q;
            __builtin_memcpy(&q, bq->qs + 4*kqs, sizeof(q));
            v[r][0] = q.x;
            v[r][1] = q.y;
            dw[r]   = bq->d;
        } else {
            v[r][0] = 0; v[r][1] = 0; dw[r] = __float2half(0.0f);
        }
    }
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const block_q8_1 * by = y + j*stride_col_y;
        u[j][0] = get_int_b4(by->qs, kqs);
        u[j][1] = get_int_b4(by->qs, kqs + 1);
        du[j]   = __low2half(by->ds);
    }

    float tmp[ncols][R] = {{0.0f}};
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            tmp[j][r] += vec_dot_q8_0_q8_1_impl<float, vdr>(v[r], u[j], dw[r], du[j]);
        }
    }
#pragma unroll
    for (int offset = LPR/2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                tmp[j][r] += ggml_cuda_shfl_xor_td<64>(tmp[j][r], offset);
            }
        }
    }
    if (l == 0) {
        dst += ch*stride_channel_dst;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r*RW;
            if (row < nrows) {
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    dst[j*stride_col_dst + row] = tmp[j][r];
                }
            }
        }
    }
}

template <int NB, int ncols>
static void mmvq_q8_0_smallk_r(int r, const void * vx, const void * vy, float * dst, int nrows, int stride_row_x, int stride_col_y,
        int stride_col_dst, int nchannels_dst, int channel_ratio, int stride_channel_x, int stride_channel_y, int stride_channel_dst,
        cudaStream_t stream) {
    constexpr int RW = 64/(4*NB);
    const dim3 block_dims(64, 1, 1);
#define MMVQ_Q8_SMALLK(R) do { \
        const dim3 block_nums((nrows + R*RW - 1)/(R*RW), nchannels_dst, 1); \
        mmvq_q8_0_smallk<NB, ncols, R><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, stride_row_x, stride_col_y, \
            stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst); } while (0)
    switch (r) {
        case 1:  MMVQ_Q8_SMALLK(1); break;
        case 2:  MMVQ_Q8_SMALLK(2); break;
        case 8:  MMVQ_Q8_SMALLK(8); break;
        default: MMVQ_Q8_SMALLK(4); break;
    }
#undef MMVQ_Q8_SMALLK
    CUDA_CHECK(cudaGetLastError());
}

template <int NB>
static void mmvq_q8_0_smallk_ncols(int ncols, int r, const void * vx, const void * vy, float * dst, int nrows, int stride_row_x,
        int stride_col_y, int stride_col_dst, int nchannels_dst, int channel_ratio, int stride_channel_x, int stride_channel_y,
        int stride_channel_dst, cudaStream_t stream) {
#define MMVQ_Q8_SMALLK_N(N) mmvq_q8_0_smallk_r<NB, N>(r, vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, \
        nchannels_dst, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, stream)
    switch (ncols) {
        case 1: MMVQ_Q8_SMALLK_N(1); break;
        case 2: MMVQ_Q8_SMALLK_N(2); break;
        case 3: MMVQ_Q8_SMALLK_N(3); break;
        case 4: MMVQ_Q8_SMALLK_N(4); break;
        case 5: MMVQ_Q8_SMALLK_N(5); break;
        case 6: MMVQ_Q8_SMALLK_N(6); break;
        case 7: MMVQ_Q8_SMALLK_N(7); break;
        default: MMVQ_Q8_SMALLK_N(8); break;
    }
#undef MMVQ_Q8_SMALLK_N
}

bool ggml_cuda_mmvq_q8_smallk(
        const void * vx, const void * vy, float * dst, int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int nchannels_x, int nchannels_y, int nchannels_dst,
        int stride_channel_x, int stride_channel_y, int stride_channel_dst, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    // GGML_MMVQ_Q8_SMALLK=0 disables; GGML_MMVQ_Q8_SMALLK_R=<1|2|4|8> overrides the row passes per wave (tuning, exact)
    static const bool enabled = !getenv("GGML_MMVQ_Q8_SMALLK") || atoi(getenv("GGML_MMVQ_Q8_SMALLK")) != 0;
    static const int  r_env   = getenv("GGML_MMVQ_Q8_SMALLK_R") ? atoi(getenv("GGML_MMVQ_Q8_SMALLK_R")) : 0;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!enabled || !GGML_CUDA_CC_IS_CDNA(cc) || ncols_dst < 1 || ncols_dst > 8 || nchannels_x < 1 ||
        nchannels_y != nchannels_dst || nchannels_dst % nchannels_x != 0) {
        return false;
    }
    const int channel_ratio = nchannels_dst/nchannels_x;
    // MI100, us (mul_mat_vec_q -> here), 3 / 6 columns: K = 128 x 8192 rows 13.7 -> 6.1 / 11.7 -> 6.9 (R = 2);
    // K = 256 x 512 rows x 64 channels 41.7 -> 17.2 / 59.8 -> 24.1, K = 512 x 256 x 64 25.7 -> 17.9 / 36.8 -> 25.5 (R = 4)
    const int r = r_env > 0 ? r_env : (ncols_x == 128 ? 2 : 4);
    switch (ncols_x) {
        case 128: mmvq_q8_0_smallk_ncols< 4>(ncols_dst, r, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                      nchannels_dst, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, stream); return true;
        case 256: mmvq_q8_0_smallk_ncols< 8>(ncols_dst, r, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                      nchannels_dst, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, stream); return true;
        case 512: mmvq_q8_0_smallk_ncols<16>(ncols_dst, r, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                      nchannels_dst, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst, stream); return true;
        default: return false;
    }
#else
    GGML_UNUSED_VARS(vx, vy, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y,
        nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst, stream);
    return false;
#endif
}

bool ggml_cuda_mmvq_q8_v2(
        const void * vx, const void * vy, float * dst, int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int nchannels_x, int nchannels_y, int nchannels_dst, int nsamples_x, int nsamples_dst, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    static const bool enabled = !getenv("GGML_MMVQ_Q8_V2") || atoi(getenv("GGML_MMVQ_Q8_V2")) != 0;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    // GGML_MMVQ_Q8_V2_WIDE=0 keeps 5..8 columns on mul_mat_vec_q
    static const bool wide = !getenv("GGML_MMVQ_Q8_V2_WIDE") || atoi(getenv("GGML_MMVQ_Q8_V2_WIDE")) != 0;
    if (!enabled || !GGML_CUDA_CC_IS_CDNA(cc) || ncols_dst < 1 || ncols_dst > (wide ? 8 : 4) ||
        nchannels_x != 1 || nchannels_y != 1 || nchannels_dst != 1 || nsamples_x != 1 || nsamples_dst != 1) {
        return false;
    }
    // K = 12288 / 16384 at 2..4 columns with chunked loads (see mmvq_q8_0_v2_rpb); GGML_MMVQ_Q8_LONGK=0 keeps them on
    // mul_mat_vec_q
    static const bool longk = !getenv("GGML_MMVQ_Q8_LONGK") || atoi(getenv("GGML_MMVQ_Q8_LONGK")) != 0;
    switch (ncols_x) {
        case  640: mmvq_q8_0_v2_ncols< 20>(ncols_dst, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); return true;
        // GLM-5.3-Flash (hidden 4096: 6.2 of its 9.4 GB of dense Q8_0 weights have K = 4096)
        case 1536: mmvq_q8_0_v2_ncols< 48>(ncols_dst, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); return true;
        case 2048: mmvq_q8_0_v2_ncols< 64>(ncols_dst, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); return true;
        case 4096: mmvq_q8_0_v2_ncols<128>(ncols_dst, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); return true;
        case 8192: mmvq_q8_0_v2_ncols<256>(ncols_dst, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); return true;
        // the loads are all issued up front, so very long rows run out of registers at several columns (MI100, 3 columns:
        // K=12288 +0%, K=16384 +50..76% slower than mul_mat_vec_q); one column is still 11..20% faster
        // 5..8 columns load one iteration at a time (chunk 1), so long rows are fine there
        case 12288: if (ncols_dst > 1 && ncols_dst < 5 && !longk) { return false; }
                    mmvq_q8_0_v2_ncols<384>(ncols_dst, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); return true;
        case 16384: if (ncols_dst > 1 && ncols_dst < 5 && !longk) { return false; }
                    mmvq_q8_0_v2_ncols<512>(ncols_dst, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); return true;
        // K = 128 only at 5..8 columns: below that mul_mat_vec_q takes its small_k entry (2 warps, rows = nwarps), which
        // this kernel does not replay
        case  128: if (ncols_dst < 5) { return false; }
                   mmvq_q8_0_v2_ncols<  4>(ncols_dst, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); return true;
        case 2560: mmvq_q8_0_v2_ncols< 80>(ncols_dst, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); return true;
        case 6144: mmvq_q8_0_v2_ncols<192>(ncols_dst, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); return true;
        default: return false;
    }
#else
    GGML_UNUSED_VARS(vx, vy, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y,
        nchannels_dst, nsamples_x, nsamples_dst, stream);
    return false;
#endif
}

bool ggml_cuda_mmvq_q8_v2_glu_supported(int ncols_x, int ncols_dst) {
#if defined(GGML_USE_HIP)
    static const bool enabled = !getenv("GGML_MMVQ_Q8_V2") || atoi(getenv("GGML_MMVQ_Q8_V2")) != 0;
    static const bool wide    = !getenv("GGML_MMVQ_Q8_V2_WIDE") || atoi(getenv("GGML_MMVQ_Q8_V2_WIDE")) != 0;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    // 2..8 columns: where the unfused gate and up GEMVs both run mmvq_q8_0_v2 (1 column keeps the upstream fused kernel)
    return ggml_cuda_fuse_moe('C') && enabled && GGML_CUDA_CC_IS_CDNA(cc) && ncols_x == 4096 &&
        ncols_dst >= 2 && ncols_dst <= (wide ? 8 : 4);
#else
    GGML_UNUSED_VARS(ncols_x, ncols_dst);
    return false;
#endif
}

bool ggml_cuda_mmvq_q8_v2_glu(
        const void * vx, const void * vgate, const void * vy, float * dst, float glu_limit, int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    if (!ggml_cuda_mmvq_q8_v2_glu_supported(ncols_x, ncols_dst)) {
        return false;
    }
    mmvq_q8_glu_gate  = vgate;
    mmvq_q8_glu_limit = glu_limit;
    const bool ok = ggml_cuda_mmvq_q8_v2(vx, vy, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
        1, 1, 1, 1, 1, stream);
    mmvq_q8_glu_gate = nullptr;
    GGML_ASSERT(ok);
    ggml_cuda_fuse_moe_count('C');
    return true;
}
