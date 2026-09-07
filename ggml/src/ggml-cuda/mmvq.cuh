#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

#include <memory>
#include <vector>
struct ggml_cuda_q8_cache_entry {
    const ggml_tensor * t = nullptr;
    const void * data = nullptr;
    int64_t ne[4] = {0, 0, 0, 0};
    size_t  nb[4] = {0, 0, 0, 0};
    size_t  bytes = 0;
    std::unique_ptr<ggml_cuda_pool_alloc<char>> buf;
};
struct ggml_cuda_q8_cache {
    bool active = false;
    std::vector<ggml_cuda_q8_cache_entry> entries;
    size_t hits = 0, misses = 0;
};
ggml_cuda_q8_cache & ggml_cuda_q8_cache_get(int device);
void ggml_cuda_q8_cache_begin(ggml_backend_cuda_context & ctx);   // call at the start of a graph evaluation
void ggml_cuda_q8_cache_end(ggml_backend_cuda_context & ctx);     // releases the cached buffers

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
