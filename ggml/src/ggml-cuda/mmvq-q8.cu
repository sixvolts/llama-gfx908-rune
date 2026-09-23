#include "mmvq-q8.cuh"
#include "reduce-dpp.cuh"
#include "vecdotq.cuh"

#include <cstdlib>
#include <cstring>

// mul_mat_vec_q<GGML_TYPE_Q8_0, ncols_dst> on the GCN parameter table: nwarps = 2, rows_per_block = 1 (ncols 1) or 2,
// blocks_per_iter = vdr*nwarps*warp_size/qi = 32, lane (tid = 64*ty + tx) owns quant blocks kbx = tid/4 + it*32 at int
// offset kqs = 2*(tid%4), accumulates vec_dot_q8_0_q8_1 over its kbx in order, warp 1 hands its partials to warp 0 through
// shared memory, warp 0 does warp_reduce_sum<64>. This kernel replays exactly that; it only changes WHEN the loads are
// issued (all up front, the row's NB blocks known at compile time) and HOW (one 8-byte load for the lane's two ints,
// which the 34-byte Q8_0 block leaves 2-byte aligned; CDNA global loads accept that).
template <int NB, int ncols, int RPB>
static __global__ void __launch_bounds__(128, 1) mmvq_q8_0_v2(
        const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
        const int nrows, const int stride_row_x, const int stride_col_y, const int stride_col_dst) {
    constexpr int warp_size = 64;
    constexpr int nwarps    = 2;
    constexpr int qi        = QI8_0;                 // 8
    constexpr int vdr       = VDR_Q8_0_Q8_1_MMVQ;    // 2
    constexpr int bpi       = vdr*nwarps*warp_size/qi;   // 32 blocks per iteration
    constexpr int NIT       = (NB + bpi - 1)/bpi;

    const int tid  = warp_size*threadIdx.y + threadIdx.x;
    const int row0 = RPB*blockIdx.x;
    const int kb_l = tid/(qi/vdr);
    const int kqs  = vdr*(tid % (qi/vdr));

    // loads: weights (2 ints + d per row and iteration), activations (2 ints + d per column and iteration)
    int  v[NIT][RPB][vdr];
    half dw[NIT][RPB];
    int  u[NIT][ncols][vdr];
    half du[NIT][ncols];
#pragma unroll
    for (int it = 0; it < NIT; ++it) {
        const int kbx = kb_l + it*bpi;
        if (kbx < NB) {
#pragma unroll
            for (int i = 0; i < RPB; ++i) {
                const block_q8_0 * bq = (const block_q8_0 *) vx + (row0 + i)*stride_row_x + kbx;
                if (RPB == 1 || row0 + i < nrows) {
                    int2 q;
                    __builtin_memcpy(&q, bq->qs + 4*kqs, sizeof(q));   // = get_int_b2(bq->qs, kqs), get_int_b2(bq->qs, kqs+1)
                    v[it][i][0] = q.x;
                    v[it][i][1] = q.y;
                    dw[it][i]   = bq->d;
                } else {
                    v[it][i][0] = 0; v[it][i][1] = 0; dw[it][i] = __float2half(0.0f);
                }
            }
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                const block_q8_1 * by = (const block_q8_1 *) vy + j*stride_col_y + kbx;
                u[it][j][0] = get_int_b4(by->qs, kqs);
                u[it][j][1] = get_int_b4(by->qs, kqs + 1);
                du[it][j]   = __low2half(by->ds);
            }
        }
    }

    float tmp[ncols][RPB] = {{0.0f}};
#pragma unroll
    for (int it = 0; it < NIT; ++it) {
        const int kbx = kb_l + it*bpi;
        if (kbx < NB) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
#pragma unroll
                for (int i = 0; i < RPB; ++i) {
                    tmp[j][i] += vec_dot_q8_0_q8_1_impl<float, vdr>(v[it][i], u[it][j], dw[it][i], du[it][j]);
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1][ncols][RPB][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int i = 0; i < RPB; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
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
            }
        }
    }
    dst += row0;
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i = 0; i < RPB; ++i) {
            if ((int) threadIdx.x == i && (RPB == 1 || row0 + i < nrows)) {
                dst[j*stride_col_dst + i] = tmp[j][i];
            }
        }
    }
}

template <int NB, int ncols, int RPB>
static void mmvq_q8_0_v2_launch(const void * vx, const void * vy, float * dst, int nrows, int stride_row_x, int stride_col_y,
        int stride_col_dst, cudaStream_t stream) {
    const dim3 block_nums((nrows + RPB - 1)/RPB, 1, 1);
    const dim3 block_dims(64, 2, 1);
    mmvq_q8_0_v2<NB, ncols, RPB><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst);
}

// rows per block: a row's accumulation order does not depend on it (each row is one lane-sequence over its blocks),
// so it is a pure tuning knob. GGML_MMVQ_Q8_RPB=<n> overrides the defaults (tuning only).
template <int NB, int ncols>
static void mmvq_q8_0_v2_rpb(int rpb, const void * vx, const void * vy, float * dst, int nrows, int stride_row_x,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    switch (rpb) {
        case 1:  mmvq_q8_0_v2_launch<NB, ncols, 1>(vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 4:  mmvq_q8_0_v2_launch<NB, ncols, 4>(vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 8:  mmvq_q8_0_v2_launch<NB, ncols, 8>(vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        default: mmvq_q8_0_v2_launch<NB, ncols, 2>(vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
    }
}

template <int NB>
static void mmvq_q8_0_v2_ncols(int ncols, const void * vx, const void * vy, float * dst, int nrows, int stride_row_x,
        int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    static const int rpb_env = getenv("GGML_MMVQ_Q8_RPB") ? atoi(getenv("GGML_MMVQ_Q8_RPB")) : 0;
    // measured on MI100 (bit-exact for any value): 1 row per block for one column, 4 for 2..4 columns on wide
    // matrices (6144 rows x 2560: 28.3 -> 24.0 us at 3 columns, 33.4 -> 26.6 at 4), 2 for narrow ones
    const int rpb = rpb_env > 0 ? rpb_env : (ncols == 1 ? 1 : (nrows >= 2048 ? 4 : 2));
    switch (ncols) {
        case 1: mmvq_q8_0_v2_rpb<NB, 1>(rpb, vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 2: mmvq_q8_0_v2_rpb<NB, 2>(rpb, vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        case 3: mmvq_q8_0_v2_rpb<NB, 3>(rpb, vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
        default: mmvq_q8_0_v2_rpb<NB, 4>(rpb, vx, vy, dst, nrows, stride_row_x, stride_col_y, stride_col_dst, stream); break;
    }
}


bool ggml_cuda_mmvq_q8_v2(
        const void * vx, const void * vy, float * dst, int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int nchannels_x, int nchannels_y, int nchannels_dst, int nsamples_x, int nsamples_dst, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    static const bool enabled = !getenv("GGML_MMVQ_Q8_V2") || atoi(getenv("GGML_MMVQ_Q8_V2")) != 0;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!enabled || !GGML_CUDA_CC_IS_CDNA(cc) || ncols_dst < 1 || ncols_dst > 4 ||
        nchannels_x != 1 || nchannels_y != 1 || nchannels_dst != 1 || nsamples_x != 1 || nsamples_dst != 1) {
        return false;
    }
    switch (ncols_x) {
        case  640: mmvq_q8_0_v2_ncols< 20>(ncols_dst, vx, vy, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst, stream); return true;
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
