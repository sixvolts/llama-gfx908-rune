#pragma once

#include "common.cuh"

// Inside a fully unrolled top-down butterfly `for (offset = W/2; offset > 0; offset >>= 1)`, ggml_cuda_shfl_xor_td<W>(x, offset)
// returns at EVERY lane exactly what __shfl_xor_sync(0xffffffff, x, offset, W) would, so the reductions stay
// bit-identical to warp_reduce_* while (on CDNA) only the xor-32 level still goes through the LDS permute unit:
// xor 16 is ds_swizzle in bit mode (lane ^ 16 within 32), xor 8 is DPP row_ror:8 (exact), xor 2 / xor 1 are quad_perm
// moves (exact). xor 4 is row_ror:4, which reads lane (i +- 4) & 15 of the row: after the xor-16 and xor-8 levels
// lanes i and i ^ 8 hold equal values, so that is the value of lane i ^ 4. This relies on the top-down order.
template <int W>
static __device__ __forceinline__ float ggml_cuda_shfl_xor_td(const float x, const int offset) {
#if defined(GGML_USE_HIP) && defined(CDNA)
    static_assert(W == 32 || W == 64, "ggml_cuda_shfl_xor_td: 32- or 64-lane butterflies only");
    if (offset == 16) {
        return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(x), 0x401F));
    }
    if (offset == 8) {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x128, 0xF, 0xF, false));   // row_ror:8
    }
    if (offset == 4) {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x124, 0xF, 0xF, false));   // row_ror:4
    }
    if (offset == 2) {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x4E, 0xF, 0xF, false));    // quad_perm [2,3,0,1]
    }
    if (offset == 1) {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0xB1, 0xF, 0xF, false));    // quad_perm [1,0,3,2]
    }
#endif // defined(GGML_USE_HIP) && defined(CDNA)
    return __shfl_xor_sync(0xffffffff, x, offset, W);
}
