#include "common.cuh"

void ggml_cuda_sparse_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_sparse_attn_supported(int device, const ggml_tensor * dst);
