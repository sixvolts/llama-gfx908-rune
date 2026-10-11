#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

#include <memory>
#include <vector>
struct ggml_cuda_q8_cache_entry {
    const ggml_tensor * base = nullptr;   // tensor that produced the bytes (pure RESHAPE/VIEW chains resolved)
    const void * data = nullptr;          // src1->data
    size_t  offs   = 0;                   // src1->data - base->data
    int64_t ne10   = 0;
    int64_t nrows  = 0;
    size_t  row_nb = 0;                   // rows form one affine sequence: row r at data + r*row_nb
    size_t  bytes  = 0;
    bool    side   = false;                // written by a producer kernel (ggml_cuda_q8_cache_reserve)
    std::unique_ptr<ggml_cuda_pool_alloc<char>> buf;
};
struct ggml_cuda_q8_cache {
    bool active = false;
    std::vector<ggml_cuda_q8_cache_entry> entries;
    size_t hits = 0, misses = 0, side = 0;
};
ggml_cuda_q8_cache & ggml_cuda_q8_cache_get(int device);
void ggml_cuda_q8_cache_begin(ggml_backend_cuda_context & ctx);   // call at the start of a graph evaluation
void ggml_cuda_q8_cache_end(ggml_backend_cuda_context & ctx);     // releases the cached buffers
// Producer-side q8_1: returns a buffer for the q8_1 rows of t (rows padded to MATRIX_ROW_PADDING), registered in the
// cache so later GEMVs reading t take it; the caller must write it on ctx.stream() before them. nullptr if the cache is
// off, t is not cacheable, or GGML_CUDA_Q8_SIDE=0.
void * ggml_cuda_q8_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t);

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
