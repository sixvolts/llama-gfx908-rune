#pragma once

#include "ggml.h"
#include "llama.h"
#include "llama-graph.h"

#include <cstdint>

struct llama_ubatch;
class llama_kv_cache;
class llama_kv_cache_context;

// GLM-5-Next indexer pooling. no input may hold a negative index (ggml_set_rows asserts
// i1 >= 0, ggml_get_rows has no sentinel), so unusable entries are clamped and masked.

// n_kv/kpool (exact only while the sequences' cells are disjoint) plus 2 per seq for rebasing
uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs = 1);

// select_k of Glm5NextTextIndexer.forward: must run over POOLS, a cell cut takes partial pools
uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool);

// LLAMA_DSA_SPARSE: DSA attention over the selected cells only (top-k pools plus the query's tail, at most
// indexer_top_k + kpool - 1 per query) instead of masking all n_kv cells, so its cost and compute buffer stop growing
// with the context. 1 = the fused GGML_OP_SPARSE_ATTN, 2 = the reference built from get_rows/mul_mat/soft_max.
// Read once; 0 when unset.
int llama_kpool_sparse_attn_mode();
bool llama_kpool_sparse_attn();
// the NextN/MTP draft context's own switch: LLAMA_MTP_DSA_SPARSE=0 keeps the draft head dense while the trunk is
// sparse (A/B of the draft's own indexer); unset, it follows LLAMA_DSA_SPARSE
bool llama_kpool_sparse_attn_mtp();

// `kv` must be the ATTENTION (MLA) cache; the indexer cache shares its slot layout.
//   pool_cells  pool member -> cell, 0 if not resident
//   pool_bias   computed, NOT gathered at the last member, which an incomplete pool lacks
//   cand_mask   bounds top-k spills a partial seq_rm would let escape
// pool_reps / new_pool_cells / new_pool_reps are nullptr when the cache is off, and an entry
// is emitted only for filled == kpool: cell 0 is real, so writing its 0 slot would clobber
// sparse attention (LLAMA_DSA_SPARSE) replaces sel_mask/cand_mask (both nullptr then) with
//   tail_cells  I32 [kpool - 1, n_tps, n_stream]: the cells of the query's partial pool at or before it (0 if unused)
//   tail_mask   F32 [kpool - 1, n_tps, n_stream]: 0 for a used slot, -INFINITY otherwise
void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              int64_t          n_kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
              ggml_tensor    * tail_cells,
              ggml_tensor    * tail_mask,
              ggml_tensor    * pool_reps,
              ggml_tensor    * new_pool_cells,
              ggml_tensor    * new_pool_reps,
        // a global row is strm_of[s]*kv_size + cell; only read when pool_reps is set
        const uint32_t       * strm_of,
              int64_t          kv_size,
        // re-emit every complete pool, not only those this ubatch closed; set after a position mutation
              bool             rebuild,
        const llama_ubatch   * ubatch,
              uint32_t         kpool);

// one map per ubatch; valid only while every indexer layer sees the same candidate set
class llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(
            const llama_kv_cache_context * mctx_attn,
            const llama_kv_cache_context * mctx_idx,
            uint32_t kpool) : mctx_attn(mctx_attn), mctx_idx(mctx_idx), kpool(kpool) {}

    ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override;

    // set_input rewrites every tensor from the ubatch and the cache state, so the graph can be reused whenever the
    // shapes it was built with still hold (params.mctx must be the llama_memory_hybrid_context build_inp_kpool took)
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * k_idxs     = nullptr;   // I32 [n_tokens]
    ggml_tensor * pool_cells = nullptr;   // I32 [kpool*n_pools, n_stream]
    ggml_tensor * pool_bias  = nullptr;   // F32 [n_pools, n_tps, n_stream]

    // pooled-key cache: a pool's value lives in the row of its LAST member, a pure function of
    // cell content (seq_cp shares it, rebase leaves it alone)
    ggml_tensor * pool_reps      = nullptr; // I32 [n_pools, n_stream]  stream-local rep cell
    ggml_tensor * new_pool_cells = nullptr; // I32 [kpool*n_new_max, n_stream] members to (re)pool
    ggml_tensor * new_pool_reps  = nullptr; // I64 [n_new_max*n_stream] GLOBAL dest row

    // fixed for the decode phase: a shape tracking pools-closed-this-step would flip the graph
    // topology every kpool tokens and force CUDA-graph recapture
    uint32_t n_new_max = 0;

    // set at build time: re-emit every pool after a position mutation
    bool rebuild = false;

    // exact, since pool_bias only holds 0.0f or -INFINITY. nullptr if the fused path is off
    ggml_tensor * pool_bias_f16 = nullptr; // F16 [n_pools, n_tps, 1, n_stream]

    ggml_tensor * sel_mask   = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]
    ggml_tensor * cand_mask  = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]

    // sparse attention: gather the selected cells instead of masking all n_kv (sel_mask/cand_mask are nullptr then)
    ggml_tensor * tail_cells = nullptr;   // I32 [kpool - 1, n_tps, n_stream]
    ggml_tensor * tail_mask  = nullptr;   // F32 [kpool - 1, n_tps, n_stream]

    // per-graph caches of the views the indexer takes of pool_cells / pool_bias (one split input per pipeline stage
    // instead of one per DSA layer)
    ggml_tensor * pool_cells_3d = nullptr;
    ggml_tensor * pool_bias_4d  = nullptr;

    const llama_kv_cache_context * mctx_attn;
    const llama_kv_cache_context * mctx_idx;

    const uint32_t kpool;
};
