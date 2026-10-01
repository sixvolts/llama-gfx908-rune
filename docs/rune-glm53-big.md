# GLM-5.3 (745B, glm-dsa) e-waste edition on rune

Companion to `rune-glm53-flash-plan.md`. The full-size GLM-5.3 (256 routed experts, MLA + DSA lightning indexer,
`glm-dsa` arch) rebuilt with the GLM-5.2 "e-waste edition" recipe (K-quants for the experts, Q3_K_M mix) and run on
all ten MI100s with the NextN/MTP block as a self-contained drafter. Dates are 2026-10-01 unless noted.

## 1. Build

- Source: `zai-org/GLM-5.3`, FP8 e4m3 with 128x128 block scales (756 GB, 141 shards), pinned to revision
  `aca966e4`, sha256-verified per file onto the NAS (`/mnt/nas/models/glm/GLM-5.3`).
- Intermediate: **F32**, not BF16. The block scales are arbitrary floats (194 of 95,040 are powers of two), so a BF16
  intermediate re-rounds 88% of the dequantized weights (typically 0.1%, up to 2^-8 relative); F32 is exactly
  `fp8 * scale`, and `llama-quantize` works in f32 anyway. 3.0 TB on the NAS, 77 shards, 2h53m to convert.
- Recipe: the per-tensor types of the published 5.2 Q3_K_M file, read from its GGUF headers over HTTP range requests
  (`scripts/rune/gguf_remote_types.py` in `~/glm53-big`), written as anchored `--tensor-type-file` overrides. Two
  deviations from the 5.2 file: the MTP block's experts at Q4_K instead of Q2_K and its `nextn.eh_proj` at Q6_K
  (+2.1 GiB), because this build runs the MTP head as a drafter. Rationale: at the head's shape the MoE block's time
  on an MI100 is within 30 us between every 4-bit type (q4_0 222 us, iq4_nl 231, q4_K 253, q2_K 255; q3_K is the
  slowest at 413), while relative RMS error on the real blk.78 experts is q2_K 0.30, q3_K 0.15, q4_K 0.072, q5_K 0.036.
- Result: `~/models/GLM-5.3-GGUF/Q3_K_M/GLM-5.3-Q3_K_M.gguf`, 297.45 GiB, 3.39 bpw, 1,524 tensors. The 5.2 file has
  1,809: the old converter wrote indexer tensors into all 79 layers; the checkpoint only has them on the 22
  `indexer_types: full` layers, and this converter records `indexer_types` so the shared layers reuse the previous
  full layer's selection (what the reference implementation does).
- Quality: wikitext-2 100x512 PPL 2.8092 +/- 0.034 (f16 KV, `-fa off`); the 5.2 Q3_K_M README figure was 2.8348 with
  a q8_0 KV cache.
- Drafter: `scripts/rune/glm53_export_mtp.py` works unchanged for glm-dsa:
  `~/models/GLM-5.3-GGUF/MTP/GLM-5.3-MTP-Q3_K_M-sc.gguf` (7.24 GB, blk.78 + token_embd/output_norm/output, byte-identical
  to the source tensors).

## 2. Placement: no CPU spill at tensor granularity

`--fit` cannot see the drafter and, when a layer does not fit, moves the remainder to the CPU instead of to another
card (9-12 GiB of experts ended up in host memory). `~/glm53-big/plan_placement.py` packs the trunk itself:

- Layer homes are sequential (`-ts`); a layer's attention, KV cache (1152 B/token MLA latent, +256 B/token indexer
  keys on full-indexer layers) and small tensors live on its home.
- Expert tensors (1008 / 1320 / 1728 MiB each) are what fragments a 32 GB card: each device picks, by subset-sum
  over the three sizes, how many of each fill it best, realized from its own layers first and then the following
  ones (`-ot` rules; `--lookahead 1` leaves ~23 tensors off their home). A tensor off its home costs that layer a
  round trip to the device holding it.
- `--ctx`, `--reserve` (runtime + compute per card) and `--extra idx:MiB` (the drafter on ROCm9) size the budgets.

8k single slot with the drafter: all weights on GPU, cards at 100-200 MiB free. With the dense attention of section
3 that was the ceiling; with sparse attention, 64k + drafter and 128k trunk-only fit (`placement-A/C.txt`).

## 3. The context-headroom bug: dense attention over every cached cell

`glm-dsa.cpp` called upstream's `build_attn(llm_graph_input_attn_k_dsa)`, which writes the indexer's top-k into an
n_kv-wide mask and then runs the ordinary dense KQ over every cached cell on every layer. The fused sparse kernel
(`ggml_sparse_attn`, `build_attn_sparse*`, `LLAMA_DSA_SPARSE`) was wired only into `glm5next` (Flash). So on the
745B each card's compute scratch was 64 heads x n_kv x n_ubatch x 4 B per layer: ~537 MB at 8k/256 (the ~660 MiB
observed), 4.3 GB per card at 128k/128 and 17 GB at 128k/512, and decode attended over all 8k cells instead of 2048.

The reference implementation (transformers `GlmMoeDsaAttention`) treats layers 0-2 as ordinary "full" indexer layers
that do their own top-k, so there is no dense-lead special case to preserve.

Port:

- `ggml-cuda/sparse-attn.cu`: a `<576, 512>` instantiation next to Flash's `<512, 512>`. The absorbed-MLA query is
  512 latent + 64 rope; the kernel's constraints (D % 32, DV % 128, DV <= D, V = the first DV values of the K row) hold.
  `test-backend-ops` `SPARSE_ATTN(d=576,...)` added: 36/36 pass against the CPU reference.
- `llm_graph_context::build_attn_sparse_topk`: the indexer's top-k list (`[n_top_k, n_tps, 1, n_stream]`) goes to the
  fused op directly, with `get_rows(kq_mask, idx)` as the per-cell mask so a slot that top-k filled from a masked
  score stays masked. `LLAMA_DSA_SPARSE=2` takes the gather reference (K rows gathered, f16 products like `-fa off`).
- `glm-dsa.cpp` switches on `llama_kpool_sparse_attn()`; unset, the dense path is unchanged.
- The drafter: `graph_mtp` runs the block's own lightning indexer (blk.78 ships trained indexer weights: same
  statistics as the trunk's full layers) and the same sparse builder; its context gets the DSA cache
  (`llama-model.cpp`). The reference shares the trunk's last selection with the MTP step instead; a draft only
  proposes and the trunk's lossless verification keeps the output exact, so this can move acceptance, not results.
  `LLAMA_MTP_DSA_SPARSE=0` keeps the draft head dense for A/B.

Measured (8k, placed, MTP n=2, single stream):

| | dense (before) | sparse |
|---|---|---|
| decode, greedy | 18.9 t/s | 19.6 t/s |
| decode, temp 1.0 | 18.4-18.5 t/s | 19.6 t/s |
| acceptance | 0.82-0.84 | 0.82-0.84 |
| trunk compute per card | ~660 MiB (ub 256) | 92-121 MiB (ub 128) |
| drafter compute (ROCm9) | 311 MiB | 60-79 MiB |
| drafter own indexer vs dense, 5.4k-token prompt | accept 0.80 | 0.78, identical text |

Drafting 3 tokens instead of 2: 18.3 t/s, prose 15.6 t/s (accept 0.56). Kept 2.

Quality gate (wikitext, c=4096 x 12 chunks, dense ub128 as the base). The first pass read sparse mean KLD 0.043,
PPL +3.2%, but the per-chunk table had every chunk at KLD 0.0001-0.003 except one, and the "numerics floor" run
(dense, ub 129) was catastrophic on that same chunk only (PPL ~3600): an intermittent prefill corruption shared
with the dense path, section 3a. With prompt batches no longer replaying HIP graphs (the gate of 3a):

| vs the dense ub128 base | mean KLD | max KLD | 99.9% | same top-1 |
|---|---|---|---|---|
| dense ub128, re-run (numerics floor) | 0.0093 | 4.18 | 0.31 | 96.96% |
| **sparse ub128** | **0.0098** | 1.35 | 0.33 | 96.81% |

Every chunk sits at 0.001-0.003 in both. The sparse path is at the floor.

### 3a. Intermittent prefill corruption: HIP graph replay across prompt ubatches

Floor matrix (`~/glm53-big/kl-floor.sh`, all dense unless noted, vs the dense ub128 base):

| run | result |
|---|---|
| identical re-run of the base config (ub 128) | chunk 7 corrupted, KLD inf |
| ub 129, fused indexer on / off | chunk 5 corrupted both times (PPL ~3600) |
| ub 130 | chunk 10 corrupted |
| sparse ub 128 / ub 129 | chunk 5 corrupted (mildly / fully) |
| **ub 64** | **clean: mean KLD 0.009, max 0.87** |
| **ub 128 with `GGML_CUDA_DISABLE_GRAPHS=1`** | **clean: mean KLD 0.009, max 4.2** |

Prompt ubatches are captured as HIP graphs here (upstream no longer gates graphs on batch size). The cache key is
shape-complete, so every prompt ubatch (a new n_kv) is a new key; the 64-entry LRU thrashes ("evicting LRU graph
entry" floods the log) and entries are reused chunk to chunk. Four of six graph-enabled prefill runs corrupted one
whole 4k-token chunk each; ub 64 (64 shapes per chunk, nothing is ever reused) and graphs-off never did.
`GGML_CUDA_GRAPH_DIAG=1` (full node-property compare on every call instead of the uid shortcut) still corrupted a
chunk (KLD 0.41), so the replayed instance matches every recorded property and is wrong anyway: the stale state is
something a capture bakes in that the properties do not cover, most likely pool scratch allocated inside a kernel.
The fix applied regardless is the gate upstream used to have: `GGML_CUDA_GRAPH_MAX_BATCH` (default 32 rows, read
from the weight matmuls' activations) keeps decode and speculative-verify batches graphed and runs prompt batches
eagerly. Flash production shares this graph code; its KL gates ran with single-ubatch 512-token chunks (graphs reused
across all 100 chunks) without a catastrophe, but that is not a proof - worth a graphs-off KL comparison there.

## 4. Memory arithmetic for long context

Per token: 78 x 1152 B (MLA latent, f16) + 21 x 256 B (indexer keys) = ~95 KB, so 12.5 GB at 128k. Weights 298.4 GB,
drafter ~7 GB sparse. 128k single-slot with the drafter needs the measured (not guessed) per-card reserve; the first
128k attempt at reserve 1100 MiB failed on ROCm0's compute buffer (979 MiB at ub 128, 3.9 GB at ub 512 - the n_kv-wide
f32 KQ masks and indexer scores scale with n_kv x n_ubatch, and ROCm0 hosts five full-indexer layers). Multi-slot x
128k is out of reach for the 745B on 10 x 32 GB with an f16 K cache; a q8_0 K-cache variant of the fused kernel would
halve the KV term.

What the prompt scratch is (`GGML_GALLOC_DEBUG=2`): at 128k/ub512 the largest allocations on every card are copies
of the n_kv-wide f32 KQ mask input, [131072 x 512] = 256 MiB each, one per scheduler split on that device (eight on
ROCm0: every expert tensor placed off its home adds a round trip and so two more splits, each with its own copy).
At ub 128 the same eight copies are 64 MiB each, most of ROCm0's 915 MiB. At 32k/ub2048 the MoE intermediates
dominate instead, `ffn_moe_down` / `ffn_moe_weighted` [6144 x 8 x 2048] f32 = 384 MiB each, ~4.3 GB on ROCm0.
Fixes, in order: a position-based per-cell mask for the sparse path (gather the listed cells' positions and compare
with the query's; an empty cell gets a sentinel), which removes the n_kv x n_ubatch MLA mask input altogether; one
input copy per device per graph in the scheduler instead of one per split; the indexer's own n_kv-wide mask stays
(its kernel takes it) and can go to f16; for ub 1024-2048 prefill, fuse the down-projection with the weighted
reduction so the [6144 x 8 x n] intermediate never exists.

## 5. Open items

- Prefill is ~10x off: 62-64 t/s at ub 128, 54 t/s with pipeline parallelism forced on (`-ot` overrides disable
  it unless `LLAMA_PIPELINE_PARALLEL=1`, llama-context.cpp) plus the 32k KV pad, 75 t/s at ub 256 (13.6k-token
  prompt, 32k placement), ub 512 does not fit (4.5 GB prompt scratch on ROCm0). Reading every expert once per ubatch
  bounds ub 128 at ~430 t/s, so the prompt path itself is slow, not the pipeline; the 256-expert MoE does take the
  MMQ path for Q2_K/Q3_K on gfx908 (`n_experts > 64`). rocprofv3 (2 x 2048 tokens, ub 256): kernel time 37 s per
  4096 tokens, of which `mul_mat_q` Q2_K 40% (9.8 ms per 1 GB expert tensor, ~108 GB/s), Q3_K 26% (~184 GB/s),
  Q4_K 8% (~380 GB/s), sparse attention 5%. At ~8 rows per expert the MMQ inner loop is VALU-bound on the K-quant
  unpack, not bandwidth or MFMA. Levers, in order: a larger prompt ubatch (an expert's weights are unpacked once per
  64-row tile, so ub 2048 is up to ~8x cheaper per token than ub 256, bounded by bandwidth) once the prompt scratch
  is shrunk; a cheaper Q2_K/Q3_K unpack in MMQ (the same lever as Q3_K decode); requantizing the hot gate/up to
  Q4_K (+~10 GB, no room today).
- q3_K experts are 1.6x slower per MoE block than q4_K on the MI100 (413 vs 253 us at the head's shape) despite fewer
  bytes; the 43 hot layers' gate/up are q3_K. The next decode lever after the attention fix.
- The chunk-5 instability above.
