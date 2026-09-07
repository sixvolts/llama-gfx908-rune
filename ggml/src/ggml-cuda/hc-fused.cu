#include "hc-fused.cuh"

#define CUDA_HC_COMBINE_BLOCK_SIZE 256

// residual/dst: [ne0, hc, nt] contiguous, block_out: [ne0, nt] contiguous (broadcast over hc),
// inject: [hc, nt] contiguous (broadcast over ne0). One thread per output element.
static __global__ void hc_combine_f32(
        const float * residual, const float * GGML_CUDA_RESTRICT block_out, const float * GGML_CUDA_RESTRICT inject,
        const float scale1, const float bias1, const float scale2, const float bias2,
        const int64_t ne0, const int64_t hc, const int64_t n, float * dst) {
    // The unfused graph rounds the mul and the add separately (two kernels). HIP's default fast
    // contraction would fuse `residual + block_out*w` into one fma (and __fmul_rn/__fadd_rn do NOT
    // prevent that on this toolchain - verified byte-exact with golden/hc_bitcheck), so turn it off here.
    // The scale steps stay explicit fmaf: scale_f32's own `scale*x + bias` is contracted the same way.
#pragma clang fp contract(off)
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t e  = i % ne0;
    const int64_t ct = i / ne0;   // c + t*hc
    const int64_t c  = ct % hc;
    const int64_t t  = ct / hc;

    // Rounding must match the unfused kernels exactly. scale_f32's `scale*x + bias` is one statement, which
    // hipcc contracts to an fma (fast contraction is the HIP default), so use fmaf explicitly. The mul and add
    // were SEPARATE kernels, each rounding its result; contraction is disabled above so they stay separate.
    const float x  = fmaf(scale1, inject[c + t*hc], bias1);     // scale_f32
    const float sg = 1.0f / (1.0f + expf(-x));                  // op_sigmoid
    const float w  = fmaf(scale2, sg, bias2);                   // scale_f32
    const float m  = block_out[e + t*ne0] * w;                  // op_mul  (own rounding, contraction off)
    dst[i] = residual[i] + m;                                   // op_add  (own rounding, contraction off)
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx,
        const ggml_tensor * residual, const ggml_tensor * block_out, const ggml_tensor * inject,
        float scale1, float bias1, float scale2, float bias2, ggml_tensor * dst) {
    GGML_ASSERT(residual->type == GGML_TYPE_F32 && block_out->type == GGML_TYPE_F32 &&
                inject->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(residual) && ggml_is_contiguous(block_out) &&
                ggml_is_contiguous(inject) && ggml_is_contiguous(dst));

    const int64_t ne0 = dst->ne[0];
    const int64_t hc  = dst->ne[1];
    const int64_t nt  = dst->ne[2];
    const int64_t n   = ne0 * hc * nt;

    GGML_ASSERT(dst->ne[3] == 1 && ggml_nelements(residual) == n);
    GGML_ASSERT(block_out->ne[0] == ne0 && ggml_nelements(block_out) == ne0 * nt);
    GGML_ASSERT(inject->ne[0] == hc && ggml_nelements(inject) == hc * nt);

    const int64_t num_blocks = (n + CUDA_HC_COMBINE_BLOCK_SIZE - 1) / CUDA_HC_COMBINE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(num_blocks, CUDA_HC_COMBINE_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_combine_f32, launch_params,
        (const float *) residual->data, (const float *) block_out->data, (const float *) inject->data,
        scale1, bias1, scale2, bias2, ne0, hc, n, (float *) dst->data);
}

#include "unary.cuh"

static __global__ void hc_mix_epilogue_f32(
        const float * GGML_CUDA_RESTRICT xn, const float * up,   // up may alias dst (decode in-place)
        const float scale, const float bias, const int64_t ne0, const int64_t hc, const int64_t n,
        float * dst) {
#pragma clang fp contract(off)   // mul and adds were separate kernels: keep their separate rounding
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t e = i % ne0;
    const int64_t t = i / ne0;
    const int64_t base = t * hc * ne0 + e;

    // g_c exactly as the unfused graph: op_mul(xn, op_sigmoid(up))
    float acc = 0.0f;
    for (int64_t cidx = 0; cidx < hc; ++cidx) {
        const int64_t j = base + cidx * ne0;
        const float g = xn[j] * (1.0f / (1.0f + expf(-up[j])));   // op_mul, own rounding
        acc = (cidx == 0) ? g : acc + g;   // cont(g0), then fused_add left-fold, own rounding
    }
    dst[i] = fmaf(scale, acc, bias);       // scale_f32 (contracted form)
}

void ggml_cuda_op_hc_mix_epilogue(ggml_backend_cuda_context & ctx,
        const ggml_tensor * xn, const ggml_tensor * up, int64_t hc,
        float scale, float bias, ggml_tensor * dst) {
    GGML_ASSERT(xn->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(xn) && ggml_is_contiguous(up) && ggml_is_contiguous(dst));
    const int64_t ne0 = dst->ne[0];
    const int64_t nt  = dst->ne[1];
    const int64_t n   = ne0 * nt;
    GGML_ASSERT(ggml_nelements(xn) == hc * n && ggml_nelements(up) == hc * n);
    const int64_t num_blocks = (n + 255) / 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, 256, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_mix_epilogue_f32, launch_params,
        (const float *) xn->data, (const float *) up->data, scale, bias, ne0, hc, n, (float *) dst->data);
}

static __global__ void scale_silu_f32(const float * x, const float scale, const float bias,
        const int64_t n, float * dst) {   // x may alias dst (in-place)
#pragma clang fp contract(off)
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = ggml_cuda_op_silu_single(fmaf(scale, x[i], bias));   // scale_f32 contracted form, then silu
}

void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx,
        const ggml_tensor * x, float scale, float bias, ggml_tensor * dst) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst) && ggml_nelements(x) == ggml_nelements(dst));
    const int64_t n = ggml_nelements(dst);
    const int64_t num_blocks = (n + 255) / 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, 256, 0, ctx.stream());
    ggml_cuda_kernel_launch(scale_silu_f32, launch_params, (const float *) x->data, scale, bias, n, (float *) dst->data);
}

// ---------------------------------------------------------------------------------------------
// HyperConnection low-rank "megakernel" (qwen4exp build_hc_mix, decode nt == 1, hc == 4):
//   lo    = silu(scale_lo * (w_down . xn) + bias_lo)          w_down: Q8_0 [K_down = hc*ne0, N_down]
//   up    = w_up . lo                                         w_up  : Q8_0 [K_up = N_down, hc*ne0]
//   dst   = scale_out * (((xn0*sig(up0) + xn1*sig(up1)) + xn2*sig(up2)) + xn3*sig(up3)) + bias_out
// Two kernels replace quantize(lo) + mmvq(up) + sigmoid/mul/cont/add/add/add/scale + scale/silu:
//   hc_down_silu_q8_0 : mmvq<Q8_0, ncols_dst=1, GCN table: nwarps=2, rows_per_block=1> + scale_silu
//   hc_up_mix_q8_0    : quantize_q8_1(lo) (per block, in shared memory) + mmvq small_k (nwarps=2,
//                       rows_per_block=2: only the lanes whose kbx < blocks_per_row do work, the idle
//                       warp contributes an explicit +0.0f) + hc_mix_epilogue
// Every accumulation happens in the same lane, in the same order, through the same device functions
// (vec_dot_q8_0_q8_1, warp_reduce_*) as the kernels it replaces, so the result is bit-identical
// (verified with golden/hc_bitcheck mega). Contraction stays at the default for the GEMV part
// (matching mmvq.cu) and is turned off only for the elementwise tails, as in the fused kernels above.
#include "vecdotq.cuh"
#include <type_traits>
#include <cstdlib>
#include "quantize.cuh"

#define HC_MEGA_QI  8   // ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi
#define HC_MEGA_VDR 2   // VDR_Q8_0_Q8_1_MMVQ

template <int NB>   // NB > 0: compile-time block count (K loop fully unrolled, same per-lane order); NB == 0: runtime loop
static __global__ void __launch_bounds__(2*ggml_cuda_get_physical_warp_size(), 1) hc_down_silu_q8_0(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int blocks_per_row_rt, const float scale, const float bias) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = 2;
    constexpr int qi        = HC_MEGA_QI;
    constexpr int vdr       = HC_MEGA_VDR;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const int tid = warp_size*threadIdx.y + threadIdx.x;
    const int row = blockIdx.x;
    const int blocks_per_row = NB > 0 ? NB : blocks_per_row_rt;

    float tmp = 0.0f;
    const int kbx_offset = row*blocks_per_row;
    const int kqs = vdr * (tid % (qi/vdr));
    if constexpr (NB > 0) {
#pragma unroll
        for (int it = 0; it < (NB + blocks_per_iter - 1) / blocks_per_iter; ++it) {
            const int kbx = tid / (qi/vdr) + it*blocks_per_iter;
            if (kbx < NB) {
                tmp += vec_dot_q8_0_q8_1(vx, &y[kbx], kbx_offset + kbx, kqs);
            }
        }
    } else {
        for (int kbx = tid / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
            tmp += vec_dot_q8_0_q8_1(vx, &y[kbx], kbx_offset + kbx, kqs);
        }
    }

    __shared__ float tmp_shared[nwarps-1][warp_size];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y-1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
    tmp += tmp_shared[0][threadIdx.x];
    tmp = warp_reduce_sum<warp_size>(tmp);

    if (threadIdx.x == 0) {
#pragma clang fp contract(off)
        dst[row] = ggml_cuda_op_silu_single(fmaf(scale, tmp, bias));   // scale_silu_f32
    }
}

// J outputs (and so 4*J up-projection rows) per block: the lo quantize replay is amortized and each warp
// issues the loads of its J rows before reducing them, one row's accumulation order unchanged.
template <int J>
static __global__ void __launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1) hc_up_mix_q8_0(
        const void * __restrict__ vx, const float * __restrict__ lo, const float * xn, float * dst,
        const int blocks_per_row, const int n_lo, const int64_t ne0, const float scale, const float bias, const int dbg_skip) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int hc  = 4;
    constexpr int qi  = HC_MEGA_QI;
    constexpr int vdr = HC_MEGA_VDR;
    constexpr int lanes_per_warp_blocks = vdr*warp_size/qi;   // kbx range one warp covers (16 on warp 64)

    __shared__ block_q8_1 y_sh[2*lanes_per_warp_blocks];
    __shared__ float u_sh[hc][J];

    const int tid  = threadIdx.x;
    const int c    = tid / warp_size;
    const int lane = tid % warp_size;

    const int64_t j0  = (int64_t) blockIdx.x * J;
    const int kbx0 = lane / (qi/vdr);
    const int kbx1 = lanes_per_warp_blocks + lane / (qi/vdr);
    const int kqs  = vdr * (lane % (qi/vdr));
    const bool has0 = kbx0 < blocks_per_row;
    const bool has1 = kbx1 < blocks_per_row;

    // 1. issue every global load up front: the weights of this warp's J rows, the epilogue's xn values,
    //    and lo for the quantize replay. None of them depends on the quantize, so they overlap it.
    int  wv0[J][vdr], wv1[J][vdr];
    half wd0[J], wd1[J];
#pragma unroll
    for (int jj = 0; jj < J; ++jj) {
#pragma unroll
        for (int i = 0; i < vdr; ++i) {
            wv0[jj][i] = 0;
            wv1[jj][i] = 0;
        }
        wd0[jj] = __float2half(0.0f);
        wd1[jj] = __float2half(0.0f);
        if (j0 + jj < ne0) {
            const int row = (int) (j0 + jj + c*ne0);
            const int kbx_offset = row*blocks_per_row;
            if (has0) {
                const block_q8_0 * bq = (const block_q8_0 *) vx + kbx_offset + kbx0;
#pragma unroll
                for (int i = 0; i < vdr; ++i) {
                    wv0[jj][i] = get_int_b2(bq->qs, kqs + i);
                }
                wd0[jj] = bq->d;
            }
            if (has1) {
                const block_q8_0 * bq = (const block_q8_0 *) vx + kbx_offset + kbx1;
#pragma unroll
                for (int i = 0; i < vdr; ++i) {
                    wv1[jj][i] = get_int_b2(bq->qs, kqs + i);
                }
                wd1[jj] = bq->d;
            }
        }
    }
    float xn_e[hc] = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (tid < J && j0 + tid < ne0) {
#pragma unroll
        for (int cidx = 0; cidx < hc; ++cidx) {
            xn_e[cidx] = xn[j0 + tid + cidx*ne0];
        }
    }
    constexpr int n_pass = 2;   // lo has at most 2*4*warp_size = 512 elements (MATRIX_ROW_PADDING)
    float lo_e[n_pass];
#pragma unroll
    for (int ps = 0; ps < n_pass; ++ps) {
        const int i = ps*4*warp_size + tid;
        lo_e[ps] = i < n_lo ? lo[i] : 0.0f;
    }

    // 2. quantize_q8_1 replay: 256-wide passes, one element per thread, width-32 reductions per block of 32.
    //    warp_reduce_max/sum<QK8_1> for both passes are issued level by level (each chain keeps its own order).
    if (!(dbg_skip & 1)) {
        float amax[n_pass], sum[n_pass];
#pragma unroll
        for (int ps = 0; ps < n_pass; ++ps) {
            amax[ps] = fabsf(lo_e[ps]);
            sum[ps]  = lo_e[ps];
        }
#pragma unroll
        for (int offset = QK8_1/2; offset > 0; offset >>= 1) {
#pragma unroll
            for (int ps = 0; ps < n_pass; ++ps) {
                amax[ps] = fmaxf(amax[ps], __shfl_xor_sync(0xffffffff, amax[ps], offset, QK8_1));
                sum[ps] += __shfl_xor_sync(0xffffffff, sum[ps], offset, QK8_1);
            }
        }
#pragma unroll
        for (int ps = 0; ps < n_pass; ++ps) {
            const int i   = ps*4*warp_size + tid;
            const int ib  = i / QK8_1;
            const int iqs = i % QK8_1;
            const float xi = lo_e[ps];
            const float  d = amax[ps] / 127.0f;
            const int8_t q = amax[ps] == 0.0f ? 0 : roundf(xi / d);
            if (ib < 2*lanes_per_warp_blocks) {
                y_sh[ib].qs[iqs] = q;
                if (iqs == 0) {
                    y_sh[ib].ds = make_half2(d, sum[ps]);
                }
            }
        }
    }
    __syncthreads();

    // 3. mmvq small_k replay for rows j + c*ne0: warp-0 lanes hold kbx = lane/4, warp-1 lanes kbx = 16 + lane/4,
    //    the second partial joins the first in the same lane before the xor-tree reduce (vec_dot_q8_0_q8_1 arithmetic)
    int  u0[vdr], u1[vdr];
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        u0[i] = get_int_b4(y_sh[kbx0].qs, kqs + i);   // kbx0 < 16, kbx1 < 32: always inside y_sh
        u1[i] = get_int_b4(y_sh[kbx1].qs, kqs + i);
    }
    const half ud0 = __low2half(y_sh[kbx0].ds);
    const half ud1 = __low2half(y_sh[kbx1].ds);
    float tmp[J];
#pragma unroll
    for (int jj = 0; jj < J; ++jj) {
        tmp[jj] = 0.0f;
        if (has0) {
            tmp[jj] += vec_dot_q8_0_q8_1_impl<float, vdr>(wv0[jj], u0, wd0[jj], ud0);
        }
        float tmp1 = 0.0f;
        if (has1) {
            tmp1 += vec_dot_q8_0_q8_1_impl<float, vdr>(wv1[jj], u1, wd1[jj], ud1);
        }
        tmp[jj] += tmp1;
    }
    // warp_reduce_sum<warp_size> for all J rows, level by level: each row's chain is the same xor tree in the
    // same order (bit-identical); issuing the J independent shuffles of a level together hides their latency
    if (!(dbg_skip & 2)) {
#pragma unroll
        for (int offset = warp_size/2; offset > 0; offset >>= 1) {
#pragma unroll
            for (int jj = 0; jj < J; ++jj) {
                tmp[jj] += __shfl_xor_sync(0xffffffff, tmp[jj], offset, warp_size);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int jj = 0; jj < J; ++jj) {
            u_sh[c][jj] = tmp[jj];
        }
    }
    __syncthreads();

    // 4. hc_mix_epilogue_f32 replay
    if (tid < J && j0 + tid < ne0 && !(dbg_skip & 4)) {
#pragma clang fp contract(off)
        float acc = 0.0f;
        for (int cidx = 0; cidx < hc; ++cidx) {
            const float g = xn_e[cidx] * (1.0f / (1.0f + expf(-u_sh[cidx][tid])));
            acc = (cidx == 0) ? g : acc + g;
        }
        dst[j0 + tid] = fmaf(scale, acc, bias);
    }
}

void ggml_cuda_op_hc_mix_mega(ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_down, const ggml_tensor * w_up, const ggml_tensor * xn,
        float scale_lo, float bias_lo, float scale_out, float bias_out, ggml_tensor * dst) {
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    constexpr int hc = 4;
    const int64_t K_down = w_down->ne[0];
    const int64_t N_down = w_down->ne[1];
    const int64_t K_up   = w_up->ne[0];
    const int64_t N_up   = w_up->ne[1];
    const int64_t ne0    = dst->ne[0];

    GGML_ASSERT(w_down->type == GGML_TYPE_Q8_0 && w_up->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(xn->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(w_down) && ggml_is_contiguous(w_up) && ggml_is_contiguous(xn) && ggml_is_contiguous(dst));
    GGML_ASSERT(xn->ne[0] == K_down && xn->ne[1] == 1 && ggml_nelements(xn) == K_down);
    GGML_ASSERT(dst->ne[1] == 1 && ggml_nelements(dst) == ne0 && K_down == hc*ne0 && N_up == hc*ne0);
    GGML_ASSERT(K_up == N_down && K_up % QK8_0 == 0 && K_down % QK8_0 == 0);
    const int nb_down = (int) (K_down / QK8_0);
    const int nb_up   = (int) (K_up / QK8_0);
    // the replayed mmvq variants: full loop for the down projection, small_k for the up projection
    GGML_ASSERT(nb_down >= 2*HC_MEGA_VDR*warp_size/HC_MEGA_QI && nb_up < 2*HC_MEGA_VDR*warp_size/HC_MEGA_QI);
    GGML_ASSERT(warp_size == 64);
    GGML_ASSERT(K_up <= 2*4*warp_size);   // hc_up_mix_q8_0 quantizes lo in two 256-wide passes

    cudaStream_t stream = ctx.stream();

    // q8_1 of xn exactly as ggml_cuda_mul_mat_vec_q would produce it
    const int64_t ne10_padded = GGML_PAD(K_down, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> xn_q8_1(ctx.pool(), ne10_padded * sizeof(block_q8_1)/QK8_1);
    quantize_row_q8_1_cuda((const float *) xn->data, nullptr, xn_q8_1.get(), GGML_TYPE_Q8_0,
        K_down, xn->nb[1]/sizeof(float), xn->nb[2]/sizeof(float), xn->nb[3]/sizeof(float), ne10_padded, 1, 1, 1, stream);

    ggml_cuda_pool_alloc<float> lo(ctx.pool(), N_down);

    {
        const dim3 block_dims(warp_size, 2, 1);
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(N_down, 1, 1), block_dims, 0, stream);
        if (nb_down == 320) {
            ggml_cuda_kernel_launch(hc_down_silu_q8_0<320>, lp,
                w_down->data, (const block_q8_1 *) xn_q8_1.get(), lo.get(), nb_down, scale_lo, bias_lo);
        } else {
            ggml_cuda_kernel_launch(hc_down_silu_q8_0<0>, lp,
                w_down->data, (const block_q8_1 *) xn_q8_1.get(), lo.get(), nb_down, scale_lo, bias_lo);
        }
    }
    {
        // outputs per block; GGML_HC_MEGA_J overrides for tuning
        static const int J_env = getenv("GGML_HC_MEGA_J") ? atoi(getenv("GGML_HC_MEGA_J")) : 0;
        const int J = J_env > 0 ? J_env : 8;
        static const int dbg_skip = getenv("GGML_HC_MEGA_SKIP") ? atoi(getenv("GGML_HC_MEGA_SKIP")) : 0;   // timing experiments only
        const dim3 block_dims(4*warp_size, 1, 1);
        auto launch = [&](auto jtag) {
            constexpr int JJ = decltype(jtag)::value;
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3((ne0 + JJ - 1)/JJ, 1, 1), block_dims, 0, stream);
            ggml_cuda_kernel_launch(hc_up_mix_q8_0<JJ>, lp,
                w_up->data, (const float *) lo.get(), (const float *) xn->data, (float *) dst->data,
                nb_up, (int) K_up, ne0, scale_out, bias_out, dbg_skip);
        };
        switch (J) {
            case 4:  launch(std::integral_constant<int, 4>{});  break;
            case 8:  launch(std::integral_constant<int, 8>{});  break;
            case 32: launch(std::integral_constant<int, 32>{}); break;
            case 64: launch(std::integral_constant<int, 64>{}); break;
            case 16: launch(std::integral_constant<int, 16>{}); break;
            default: launch(std::integral_constant<int, 8>{});  break;
        }
    }
}
