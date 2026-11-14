#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

enum { FD_KPOOL, FD_TOPK_MASK, FD_CONV_MS, FD_KDA_GATE, FD_KDA_BETA, FD_KDA_TAIL, FD_N };
// counts a fused launch (host side, so HIP graph replays are not counted; GGML_CUDA_FD_STATS=1)
void ggml_cuda_fd_count(int which);

// a plain (not split) device buffer of this backend (ggml_backend_buffer_is_cuda)
bool ggml_cuda_fd_buffer_is_cuda(ggml_backend_buffer_t buffer);

// rune fuse-dense (switches in fuse-dense.cuh). Each try returns the number of nodes to skip (0: not fused).
typedef bool (*ggml_cuda_fd_forward_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);   // ggml_cuda_compute_forward
int  ggml_cuda_fd_try_conv_ms(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, ggml_cuda_fd_forward_t fwd);
int  ggml_cuda_fd_try_kda_beta(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
int  ggml_cuda_fd_try_kda_gate(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
int  ggml_cuda_fd_try_kda_tail(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, ggml_cuda_fd_forward_t fwd);
int  ggml_cuda_fd_try_topk_mask(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
// graph_optimize: alloc deps that keep the fused kernels' inputs from being overwritten by their outputs
void ggml_cuda_fd_alloc_deps(const ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params);
