#pragma once

#include <cstdlib>

// rune: bit-exact decode fusions of the non-MoE GLM-5.3-Flash chains (fuse-dense.cu, kpool-compress.cu), all off by
// default. GGML_CUDA_FUSE_DENSE=1 turns every one on; a per-fusion switch (=1 on, =0 off) overrides the umbrella:
//   GGML_CUDA_FD_KPOOL     DSA indexer pool compressor (kpool-compress.cu), 9 launches -> 1
//   GGML_CUDA_FD_TOPK_MASK DSA sparse-gather cell list + mask after the pool top-k, 11 launches -> 1
//   GGML_CUDA_FD_CONV_MS   KDA conv-input assembly for several sequences (the kda_conv_rows kernel), 3 + 2n -> 1
//   GGML_CUDA_FD_KDA_GATE  KDA gate: q8_1(f_a) + f_b GEMV + add/mul/scale/sigmoid/scale, 7 launches -> 1
//   GGML_CUDA_FD_KDA_BETA  KDA beta GEMV + sigmoid, 2 launches -> 1
//   GGML_CUDA_FD_KDA_TAIL  KDA output: cont + rms_norm + q8_1(g_a) + g_b + sigmoid*mul + cont + q8_1, 7 launches -> 1
static inline bool ggml_cuda_fd_on(const char * name) {
    const char * e = getenv(name);
    if (e != nullptr) {
        return atoi(e) != 0;
    }
    static const bool all = getenv("GGML_CUDA_FUSE_DENSE") && atoi(getenv("GGML_CUDA_FUSE_DENSE")) != 0;
    return all;
}
