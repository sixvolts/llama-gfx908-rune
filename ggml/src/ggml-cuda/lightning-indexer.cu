#include "common.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(TURING_MMA_AVAILABLE)

typedef union {
    int2 i2;
    half2 h2[2];
} half4;

// TODO add support for AMD cards via rocWMMA
#include <mma.h>
namespace wmma = nvcuda::wmma;

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEADS_PER_INNER_LOOP = 8;
    constexpr int K_EMBD_PER_INNER_LOOP = 16;
    constexpr int N_EMBD_PADDED = N_EMBD + 8;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each block processes K_VECS_PER_BLOCK K vectors
    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load weights and first Q tile to shared memory

    __shared__ float w_shared[N_HEAD];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][N_EMBD_PADDED / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }

    // total number of half4 elements in HEADS_PER_INNER_LOOP x N_EMBD Q tile
    constexpr int N_Q_TILE = HEADS_PER_INNER_LOOP * (N_EMBD / 4);
    // number of registers needed in each thread to store Q tile in thread block
    constexpr int N_Q_NEXT = (N_Q_TILE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma unroll
    for (int i_q = tid; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (N_EMBD / 4);
        const int i_embd = i_q % (N_EMBD / 4);
        const float4 q = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        half4 q_packed;
        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
        q_shared_h[i_head][i_embd] = q_packed.i2;
    }

    // phase 2 - load (and dequantize if needed) K to shared mem

    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][N_EMBD_PADDED / 4][2];

    constexpr int n_k = K_VECS_PER_BLOCK * (N_EMBD / 4);

    if constexpr (TYPE_K == GGML_TYPE_F16) {
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    }

    __syncthreads();

    // phase 3 - calculate lightning indexer scores

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    // load K fragment
    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        // we don't use accumulator for anything, fill it with zeros
        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
        wmma::fill_fragment(frag_acc, 0.0f);

        // load Q fragment
        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

        // preload next Q tile to registers during matrix multiplication
        float4 q_next[N_Q_NEXT];

        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (N_EMBD / 4);
                const int i_embd =               i_q % (N_EMBD / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
            }
        }

        // perform matrix multiplication
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        // make sure all threads finished using q_shared_h so we can store next tile
        __syncthreads();

        // write preloaded Q tile to shared memory
        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_q / (N_EMBD / 4);
                const int i_embd = i_q % (N_EMBD / 4);
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
                q_shared_h[i_head][i_embd] = q_packed.i2;
                ++i_q_next;
            }
        }

        // accumulate QK multiplication results from all block warps
        // (there are 256 threads in block and 256 matmul outputs)
        // TODO it will break if WARP_SIZE is not 32
        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += qk_shared[w][h][k];
        }

        // ReLU, weight
        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        // wait until qk_shared[0] is no longer used
        __syncthreads();

        // reuse qk_shared[0] for storing partial results
        qk_shared[0][h][k] = sum;

        // wait until all threads write their results
        __syncthreads();

        // accumulate result over heads
        if (tid < K_VECS_PER_BLOCK) {
#pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        // make sure all threads finished using qk_shared
        __syncthreads();
    }

    // phase 4 - store output to VRAM

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = start_kv + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k + __half2float(m_base[i_kv]);
        }
    }
}

#else // defined(TURING_MMA_AVAILABLE)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {
    GGML_UNUSED_VARS(Q, K, W, M, dst,
        n_stream, n_batch, n_kv,
        nb1, nb2, nb3,
        nbq1, nbq2, nbq3,
        nbk1, nbk2, nbk3,
        nbw1, nbw2, nbw3,
        nem3);
    NO_DEVICE_CODE;
}

#endif // defined(TURING_MMA_AVAILABLE)
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (N_EMBD)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_vec(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load (and dequantize if needed) K to registers

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
        // direct copy of float4
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const float4 * k_base = (const float4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else {
        // dequantize remaining types to float
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    // load weights and Q only for N_HEAD_INNER heads at once to reduce shared memory usage
    constexpr int N_HEAD_INNER = N_HEAD / 4;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // phase 2 - load weights and Q to shared memory

        __shared__ float  w_shared[N_HEAD_INNER];
        __shared__ float4 q_shared_f[N_HEAD_INNER][N_EMBD / 4];

        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = N_HEAD_INNER * (N_EMBD / 4);
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_head_inner][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        }

        __syncthreads();

        // phase 3 - calculate lightning indexer scores

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            // dot product of floats
            const float4 q_vec = q_shared_f[i_head_inner][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                // ReLU, weight
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid] + __half2float(m_base[i_kv]);
        }
    }
}

// rune gfx908: tiled f32 indexer, BIT-EXACT with lightning_indexer_kernel_vec (f32 K path).
//
// The vec kernel spends its time in warp shuffles (one 5-step butterfly per (query, head, key)), keeps only one
// query per block (every K row is re-read for each of the n_batch queries) and scores every padded pool, including
// the ones the mask hides. This kernel reproduces the vec kernel's arithmetic exactly, per output:
//   leaf l (= vec lane l, dims 4l..4l+3): s_l = 0; s_l += q.x*k.x; s_l += q.y*k.y; s_l += q.z*k.z; s_l += q.w*k.w
//   dot = the xor-butterfly sum over the 32 leaves (offsets 16, 8, 4, 2, 1): a balanced binary tree over the leaves
//         taken in bit-reversed lane order, pairs (l, l^16) first -> evaluated here per thread as the same tree
//         (IEEE add is commutative, so only the grouping matters, and the grouping is identical)
//   score += relu(dot) * w[h] over heads 0..N_HEAD-1 in order; dst = score + mask
// but each thread computes a 2 query x 4 key micro-tile with no shuffles, K is staged once per block for 32 queries
// and all heads, and a (32 query x 64 key) tile whose mask is entirely -inf is not scored at all: its outputs are
// written as the mask value (-inf), which is what score + (-inf) gives for any finite score.
// GGML_CUDA_LI_TILED=0 restores the vec kernel.
namespace li_tiled {

constexpr int BQ      = 32;          // queries per block
constexpr int BP      = 64;          // keys (pools) per block
constexpr int NT      = 256;         // threads per block
constexpr int TQ      = 2;           // queries per thread
constexpr int TP      = 4;           // keys per thread
constexpr int ROW     = 128 + 4;     // LDS row stride in floats (pad: spreads key rows over banks)

static_assert((BQ/TQ) * (BP/TP) == NT, "thread tile");

constexpr int bitrev5(int x) {
    return ((x & 1) << 4) | ((x & 2) << 2) | (x & 4) | ((x & 8) >> 2) | ((x & 16) >> 4);
}

template <int LO, int N>
struct tree {
    static __device__ __forceinline__ void run(float (&out)[TQ*TP], const float * const (&qr)[TQ], const float * const (&kr)[TP]) {
        if constexpr (N == 1) {
            constexpr int l = bitrev5(LO);
            float4 qv[TQ];
            float4 kv[TP];
#pragma unroll
            for (int i = 0; i < TQ; ++i) {
                qv[i] = *(const float4 *) (qr[i] + 4*l);
            }
#pragma unroll
            for (int j = 0; j < TP; ++j) {
                kv[j] = *(const float4 *) (kr[j] + 4*l);
            }
#pragma unroll
            for (int i = 0; i < TQ; ++i) {
#pragma unroll
                for (int j = 0; j < TP; ++j) {
                    float s = 0.0f;
                    ggml_cuda_mad(s, qv[i].x, kv[j].x);
                    ggml_cuda_mad(s, qv[i].y, kv[j].y);
                    ggml_cuda_mad(s, qv[i].z, kv[j].z);
                    ggml_cuda_mad(s, qv[i].w, kv[j].w);
                    out[i*TP + j] = s;
                }
            }
        } else {
            float a[TQ*TP];
            float b[TQ*TP];
            tree<LO,       N/2>::run(a, qr, kr);
            tree<LO + N/2, N/2>::run(b, qr, kr);
#pragma unroll
            for (int o = 0; o < TQ*TP; ++o) {
                out[o] = a[o] + b[o];
            }
        }
    }
};

// the same tree evaluated as a binary counter over the leaves j = 0..31 (leaf j = vec lane bitrev5(j)) with the
// operands of leaf j+1 loaded before leaf j is computed (explicit register double buffer): identical additions
// (level-L partial = left partial + right partial), but the LDS loads of the next leaf overlap the FMAs of this one.
static __device__ __forceinline__ void leaf_load(float4 (&qv)[TQ], float4 (&kv)[TP], const float * const (&qr)[TQ],
        const float * const (&kr)[TP], const int l) {
#pragma unroll
    for (int i = 0; i < TQ; ++i) {
        qv[i] = *(const float4 *) (qr[i] + 4*l);
    }
#pragma unroll
    for (int j = 0; j < TP; ++j) {
        kv[j] = *(const float4 *) (kr[j] + 4*l);
    }
}

static __device__ __forceinline__ void tree_stream(float (&out)[TQ*TP], const float * const (&qr)[TQ], const float * const (&kr)[TP]) {
    float st[5][TQ*TP];   // st[L]: pending left partial of level L (sum of 2^L leaves)
    float4 qa[TQ], ka[TP], qb[TQ], kb[TP];
    leaf_load(qa, ka, qr, kr, bitrev5(0));
#pragma unroll
    for (int jj = 0; jj < 32; ++jj) {
        float4 (&qc)[TQ] = (jj & 1) ? qb : qa;
        float4 (&kc)[TP] = (jj & 1) ? kb : ka;
        if (jj + 1 < 32) {
            if (jj & 1) {
                leaf_load(qa, ka, qr, kr, bitrev5(jj + 1));
            } else {
                leaf_load(qb, kb, qr, kr, bitrev5(jj + 1));
            }
        }
        float c[TQ*TP];
#pragma unroll
        for (int i = 0; i < TQ; ++i) {
#pragma unroll
            for (int j = 0; j < TP; ++j) {
                float s = 0.0f;
                ggml_cuda_mad(s, qc[i].x, kc[j].x);
                ggml_cuda_mad(s, qc[i].y, kc[j].y);
                ggml_cuda_mad(s, qc[i].z, kc[j].z);
                ggml_cuda_mad(s, qc[i].w, kc[j].w);
                c[i*TP + j] = s;
            }
        }
        // carry up while the counter bit is set: level L completes when bits 0..L of jj are all 1
#pragma unroll
        for (int L = 0; L < 5; ++L) {
            if (((jj >> L) & 1) == 0) {
#pragma unroll
                for (int o = 0; o < TQ*TP; ++o) {
                    st[L][o] = c[o];
                }
                break;
            }
#pragma unroll
            for (int o = 0; o < TQ*TP; ++o) {
                c[o] = st[L][o] + c[o];
            }
            if (L == 4) {
#pragma unroll
                for (int o = 0; o < TQ*TP; ++o) {
                    out[o] = c[o];
                }
            }
        }
    }
}

} // namespace li_tiled

template <int64_t N_HEAD, bool STREAM>
static __global__ void __launch_bounds__(li_tiled::NT, 1) lightning_indexer_kernel_tiled_f32(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3) {
    using namespace li_tiled;
    GGML_UNUSED_VARS(n_stream, nb2, nbk1, nbw2, nbm2);

    const int tid      = threadIdx.x;
    const int tx       = tid % (BP/TP);      // key group: keys tx + 16*j
    const int ty       = tid / (BP/TP);      // query group: queries TQ*ty + i
    const int p0       = blockIdx.x * BP;
    const int q0       = blockIdx.y * BQ;
    const int i_stream = blockIdx.z;

    // phase 0 - mask of this thread's outputs; skip the tile when the mask hides all of it
    float mval[TQ*TP];
    int visible = 0;
#pragma unroll
    for (int i = 0; i < TQ; ++i) {
        const int iq = q0 + TQ*ty + i;
        const half * m_row = (const half *) ((const char *) M + (int64_t) iq*nbm1 + (i_stream % nem3)*nbm3);
#pragma unroll
        for (int j = 0; j < TP; ++j) {
            const int ip = p0 + tx + (BP/TP)*j;
            float mv = -INFINITY;
            if (iq < n_batch && ip < n_kv) {
                mv = __half2float(m_row[ip]);
            }
            mval[i*TP + j] = mv;
            visible |= !(isinf(mv) && mv < 0.0f);
        }
    }
    if (!__syncthreads_or(visible)) {
#pragma unroll
        for (int i = 0; i < TQ; ++i) {
            const int iq = q0 + TQ*ty + i;
            if (iq >= n_batch) {
                continue;
            }
            float * d_row = (float *) ((char *) dst + (int64_t) iq*nb1 + i_stream*nb3);
#pragma unroll
            for (int j = 0; j < TP; ++j) {
                const int ip = p0 + tx + (BP/TP)*j;
                if (ip < n_kv) {
                    d_row[ip] = mval[i*TP + j];
                }
            }
        }
        return;
    }

    __shared__ __align__(16) float k_s[BP*ROW];
    __shared__ __align__(16) float q_s[BQ*ROW];
    __shared__ float w_s[BQ][N_HEAD];

    // phase 1 - K tile (all heads share it) and the head weights of the tile's queries
    for (int e = tid; e < BP*32; e += NT) {
        const int r  = e / 32;
        const int c4 = e % 32;
        const int ip = p0 + r;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (ip < n_kv) {
            v = ((const float4 *) (K + (int64_t) ip*nbk2 + i_stream*nbk3))[c4];
        }
        *(float4 *) (k_s + r*ROW + 4*c4) = v;
    }
    for (int e = tid; e < BQ*N_HEAD; e += NT) {
        const int r  = e / N_HEAD;
        const int h  = e % N_HEAD;
        const int iq = q0 + r;
        w_s[r][h] = iq < n_batch ? ((const float *) ((const char *) W + (int64_t) iq*nbw1 + i_stream*nbw3))[h] : 0.0f;
    }

    const char * q_base = (const char *) Q + i_stream*nbq3;
    constexpr int NQ4 = BQ*32 / NT;     // float4 of one head's Q tile per thread
    float4 q_next[NQ4];
#pragma unroll
    for (int u = 0; u < NQ4; ++u) {
        const int e  = tid + u*NT;
        const int r  = e / 32;
        const int c4 = e % 32;
        const int iq = q0 + r;
        q_next[u] = iq < n_batch ? ((const float4 *) (q_base + (int64_t) iq*nbq2))[c4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const float * qr[TQ];
    const float * kr[TP];
#pragma unroll
    for (int i = 0; i < TQ; ++i) {
        qr[i] = q_s + (TQ*ty + i)*ROW;
    }
#pragma unroll
    for (int j = 0; j < TP; ++j) {
        kr[j] = k_s + (tx + (BP/TP)*j)*ROW;
    }

    float score[TQ*TP];
#pragma unroll
    for (int o = 0; o < TQ*TP; ++o) {
        score[o] = 0.0f;
    }

    for (int h = 0; h < N_HEAD; ++h) {
        __syncthreads();     // previous head's q_s reads are done (and, for h == 0, k_s/w_s are written)
#pragma unroll
        for (int u = 0; u < NQ4; ++u) {
            const int e  = tid + u*NT;
            const int r  = e / 32;
            const int c4 = e % 32;
            *(float4 *) (q_s + r*ROW + 4*c4) = q_next[u];
        }
        __syncthreads();
        if (h + 1 < N_HEAD) {
#pragma unroll
            for (int u = 0; u < NQ4; ++u) {
                const int e  = tid + u*NT;
                const int r  = e / 32;
                const int c4 = e % 32;
                const int iq = q0 + r;
                q_next[u] = iq < n_batch ?
                    ((const float4 *) (q_base + (int64_t) iq*nbq2 + (int64_t) (h + 1)*nbq1))[c4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        float dot[TQ*TP];
        if constexpr (STREAM) {
            tree_stream(dot, qr, kr);
        } else {
            tree<0, 32>::run(dot, qr, kr);
        }

#pragma unroll
        for (int i = 0; i < TQ; ++i) {
            const float w_val = w_s[TQ*ty + i][h];
#pragma unroll
            for (int j = 0; j < TP; ++j) {
                float sum = dot[i*TP + j];
                // ReLU, weight
                sum = (sum > 0.0f) ? sum : 0.0f;
                score[i*TP + j] += sum * w_val;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TQ; ++i) {
        const int iq = q0 + TQ*ty + i;
        if (iq >= n_batch) {
            continue;
        }
        float * d_row = (float *) ((char *) dst + (int64_t) iq*nb1 + i_stream*nb3);
#pragma unroll
        for (int j = 0; j < TP; ++j) {
            const int ip = p0 + tx + (BP/TP)*j;
            if (ip < n_kv) {
                d_row[ip] = score[i*TP + j] + mval[i*TP + j];
            }
        }
    }
}

#if defined(GGML_USE_HIP)
// rune gfx908: decode-shaped f32 indexer (1..8 queries), BIT-EXACT with lightning_indexer_kernel_vec (f32 K path).
//
// In decode the vec kernel is latency-bound: every (query, head, key) runs a dependent 5-step xor butterfly (2048
// shuffles per lane per block for 32 heads x 8 keys), ~85 us per call however short the context. Here each
// (query, key) score is computed by 4 lanes: lane g evaluates the leaves 8g..8g+7 of the vec kernel's 32-leaf tree
// (leaf j = vec lane bitrev5(j), the same 4-FMA leaf chain, the same pairwise grouping inside the 8 leaves), and the
// four subtrees are combined as ((t0 + t1) + (t2 + t3)) with two xor shuffles = the tree's top two levels (IEEE add is
// commutative, so every lane ends with identical bits). Then relu(dot) * w is accumulated over the heads in order and
// the mask is added, exactly as in the vec kernel. A key's K row stays in registers across all heads (8 float4 per
// lane); Q and W are staged in LDS per chunk of heads and read as broadcasts. A block whose keys are all hidden
// (mask -inf for every query) writes the mask value without scoring, as the tiled kernel does.
// GGML_CUDA_LI_DECODE=0 keeps the vec kernel.
namespace li_dec {
constexpr int NT  = 256;          // threads per block
constexpr int KPB = NT / 4;       // keys per block (4 lanes per key)

constexpr int bitrev3(int x) {
    return ((x & 1) << 2) | (x & 2) | ((x & 4) >> 2);
}

// leaves LO..LO+N-1 (local, 0..7) of this lane's 8-leaf subtree: kr[j] = K float4 of leaf j, q4 = this head's Q row
// as float4 with the lane's leaf j at q4[4*bitrev3(j) + r]
template <int LO, int N>
struct sub {
    static __device__ __forceinline__ float run(const float4 (&kr)[8], const float4 * q4, const int r) {
        if constexpr (N == 1) {
            const float4 qv = q4[4*bitrev3(LO) + r];
            float s = 0.0f;
            ggml_cuda_mad(s, qv.x, kr[LO].x);
            ggml_cuda_mad(s, qv.y, kr[LO].y);
            ggml_cuda_mad(s, qv.z, kr[LO].z);
            ggml_cuda_mad(s, qv.w, kr[LO].w);
            return s;
        } else {
            const float a = sub<LO,       N/2>::run(kr, q4, r);
            const float b = sub<LO + N/2, N/2>::run(kr, q4, r);
            return a + b;
        }
    }
};
} // namespace li_dec

template <int64_t N_HEAD, int NQ, int HC>
static __global__ void __launch_bounds__(li_dec::NT, 1) lightning_indexer_kernel_decode_f32(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3) {
    using namespace li_dec;
    GGML_UNUSED_VARS(n_stream, nb2, nbk1, nbw2, nbm2);
    static_assert(N_HEAD % HC == 0, "head chunks");

    const int tid      = threadIdx.x;
    const int g        = tid % 4;                    // subtree of this lane: leaves 8g..8g+7
    const int r        = ((g & 1) << 1) | (g >> 1);  // bitrev2(g): leaf 8g+j is vec lane 4*bitrev3(j) + r
    const int ip       = blockIdx.x*KPB + tid/4;
    const int i_stream = blockIdx.z;
    // queries q0 .. q0+NQ-1 of this block (grid.y = ceil(n_batch / NQ)): the verify batch's queries run in parallel blocks
    const int q0       = blockIdx.y*NQ;

    // mask of this lane's outputs; skip the block when the mask hides every key for every query
    float mval[NQ];
    int visible = 0;
    const char * m_base = (const char *) M + (i_stream % nem3)*nbm3;
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        float mv = -INFINITY;
        if (q0 + i < n_batch && ip < n_kv) {
            const half * m_row = (const half *) (m_base + (int64_t) (q0 + i)*nbm1);
            mv = __half2float(m_row[ip]);
        }
        mval[i] = mv;
        visible |= !(isinf(mv) && mv < 0.0f);
    }
    if (!__syncthreads_or(visible)) {
        if (g == 0 && ip < n_kv) {
#pragma unroll
            for (int i = 0; i < NQ; ++i) {
                if (q0 + i < n_batch) {
                    ((float *) ((char *) dst + (int64_t) (q0 + i)*nb1 + i_stream*nb3))[ip] = mval[i];
                }
            }
        }
        return;
    }

    // this lane's 8 K float4 (leaf j -> vec lane 4*bitrev3(j) + r), kept for all heads
    float4 kr[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        kr[j] = ip < n_kv ? ((const float4 *) (K + (int64_t) ip*nbk2 + i_stream*nbk3))[4*bitrev3(j) + r]
                          : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    __shared__ __align__(16) float4 q_s[NQ][HC][32];
    __shared__ float w_s[NQ][N_HEAD];
    for (int e = tid; e < NQ*N_HEAD; e += NT) {
        const int i = e / N_HEAD;
        const int h = e % N_HEAD;
        w_s[i][h] = q0 + i < n_batch ? ((const float *) ((const char *) W + (int64_t) (q0 + i)*nbw1 + i_stream*nbw3))[h] : 0.0f;
    }

    const char * q_base = (const char *) Q + i_stream*nbq3;
    float score[NQ];
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        score[i] = 0.0f;
    }

    for (int h0 = 0; h0 < N_HEAD; h0 += HC) {
        __syncthreads();   // previous chunk's q_s reads are done (and w_s is written)
        for (int e = tid; e < NQ*HC*32; e += NT) {
            const int i  = e / (HC*32);
            const int hh = (e / 32) % HC;
            const int c4 = e % 32;
            q_s[i][hh][c4] = q0 + i < n_batch ? ((const float4 *) (q_base + (int64_t) (q0 + i)*nbq2 + (int64_t) (h0 + hh)*nbq1))[c4]
                                              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();
#pragma unroll 4
        for (int hh = 0; hh < HC; ++hh) {
#pragma unroll
            for (int i = 0; i < NQ; ++i) {
                float t = sub<0, 8>::run(kr, q_s[i][hh], r);
                // quad_perm [1,0,3,2] = xor 1, [2,3,0,1] = xor 2 within each 4-lane group (exact register moves)
                t = t + __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(t), 0xB1, 0xF, 0xF, false));   // (t0 + t1), (t2 + t3)
                t = t + __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(t), 0x4E, 0xF, 0xF, false));   // (t0 + t1) + (t2 + t3)
                const float w_val = w_s[i][h0 + hh];
                float sum = t;
                // ReLU, weight
                sum = (sum > 0.0f) ? sum : 0.0f;
                score[i] += sum * w_val;
            }
        }
    }

    if (g == 0 && ip < n_kv) {
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            if (q0 + i < n_batch) {
                ((float *) ((char *) dst + (int64_t) (q0 + i)*nb1 + i_stream*nb3))[ip] = score[i] + mval[i];
            }
        }
    }
}

static bool lightning_indexer_use_decode(const ggml_tensor * k, int64_t n_batch) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_LI_DECODE"); return e == nullptr || atoi(e) != 0; }();
    return on && k->type == GGML_TYPE_F32 && n_batch >= 1 && n_batch <= 8;
}
#endif // defined(GGML_USE_HIP)

// GGML_CUDA_LI_TILED: unset/1 = tiled kernel for f32 K when the batch has >= GGML_CUDA_LI_TILED_MIN queries (default 32;
// MI100, 8192 pools: vec/tiled = 0.20x at 1 query, 0.58x at 8, 0.99x at 16, 1.80x at 32, 2.84x at 128), 0 = never,
// 2 = always
static bool lightning_indexer_use_tiled(const ggml_tensor * k, int64_t n_batch) {
    const char * e = getenv("GGML_CUDA_LI_TILED");
    const int mode = e != nullptr ? atoi(e) : 1;
    if (mode == 0 || k->type != GGML_TYPE_F32) {
        return false;
    }
    const char * em = getenv("GGML_CUDA_LI_TILED_MIN");
    const int64_t min_batch = em != nullptr ? atoi(em) : 32;
    return mode == 2 || n_batch >= min_batch;
}

#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)         \
    if (K->type == (type_K)) {                                                              \
        lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K> \
            <<<grid, block, 0, ctx.stream()>>>(                                             \
            q_d, k_d, w_d, m_d, dst_d,                                                      \
            n_stream, n_batch, n_kv,                                                        \
            nb1, nb2, nb3,                                                                  \
            nbq1, nbq2, nbq3,                                                               \
            nbk1, nbk2, nbk3,                                                               \
            nbw1, nbw2, nbw3,                                                               \
            nbm1, nbm2, nbm3,                                                               \
            nem3                                                                            \
        );                                                                                  \
    } else

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];

    const float *   q_d = (const float *)   q->data;
    const char  *   k_d = (const char  *)   k->data;
    const float *   w_d = (const float *)   w->data;
    const half  *   m_d = (const half  *)   m->data;
    float       * dst_d = (      float *) dst->data;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;

    if (n_embd == 128 && (n_head == 64 || n_head == 32) && lightning_indexer_use_tiled(k, n_batch)) {
        dim3 block(li_tiled::NT, 1, 1);
        dim3 grid((n_kv + li_tiled::BP - 1) / li_tiled::BP, (n_batch + li_tiled::BQ - 1) / li_tiled::BQ, n_stream);
        auto launch = [&](auto kern) {
            kern<<<grid, block, 0, ctx.stream()>>>(
                q_d, k_d, w_d, m_d, dst_d,
                n_stream, n_batch, n_kv,
                nb1, nb2, nb3,
                nbq1, nbq2, nbq3,
                nbk1, nbk2, nbk3,
                nbw1, nbw2, nbw3,
                nbm1, nbm2, nbm3,
                nem3);
        };
        // binary-counter tree with register double-buffered leaf loads (same arithmetic, ~10% faster);
        // GGML_CUDA_LI_TILED_STREAM=0 uses the recursive tree
        const char * es = getenv("GGML_CUDA_LI_TILED_STREAM");
        const bool stream = es == nullptr || atoi(es) != 0;
        if (n_head == 64) {
            stream ? launch(lightning_indexer_kernel_tiled_f32<64, true>) : launch(lightning_indexer_kernel_tiled_f32<64, false>);
        } else {
            stream ? launch(lightning_indexer_kernel_tiled_f32<32, true>) : launch(lightning_indexer_kernel_tiled_f32<32, false>);
        }
        return;
    }

#if defined(GGML_USE_HIP)
    if (n_embd == 128 && (n_head == 64 || n_head == 32) && lightning_indexer_use_decode(k, n_batch)) {
        dim3 block(li_dec::NT, 1, 1);
        dim3 grid((n_kv + li_dec::KPB - 1) / li_dec::KPB, n_batch, n_stream);   // one query per block
        auto launch = [&](auto kern) {
            kern<<<grid, block, 0, ctx.stream()>>>(
                q_d, k_d, w_d, m_d, dst_d,
                n_stream, n_batch, n_kv,
                nb1, nb2, nb3,
                nbq1, nbq2, nbq3,
                nbk1, nbk2, nbk3,
                nbw1, nbw2, nbw3,
                nbm1, nbm2, nbm3,
                nem3);
        };
        // one query per block (the verify's 3 queries run as parallel blocks; one block holding all queries scored them
        // serially and was slower than the vec kernel at 3 queries); LDS: 32 heads x 512 B of Q + W
        if (n_head == 64) {
            launch(lightning_indexer_kernel_decode_f32<64, 1, 32>);
        } else {
            launch(lightning_indexer_kernel_decode_f32<32, 1, 32>);
        }
        return;
    }
#endif // defined(GGML_USE_HIP)

    if (n_embd == 128 && n_head == 64) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 32) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else {
        GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    if (neq0 != 128) {
        return false;
    }

    if (neq1 != 64 && neq1 != 32) {
        return false;
    }

    // alignment checks
    for (const ggml_tensor * t : {q, k}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch(k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}
