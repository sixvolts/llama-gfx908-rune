# GLM-5.3-Flash decode, round 3 (2026-10-06): 9th build -> branch glm53/decode-r3

Decode on rune is GPU-kernel-bound and serial: a verify step (1 + 2 draft tokens) runs the 9 trunk stages one after
the other and then the MTP drafter, ~46-48 ms of summed kernel time per step. This round targets the sparse-attention
selection path (lightning indexer + top-k over KV pools, 14 calls per step above ~2k context) and the MoE down
projection, which fell back to an older kernel in the rune Q4_K quant. Plans and reviews: ~/glm53-prefill/PLAN-r3-topk.md,
PLAN-r3-moe.md, results/r3/REVIEW-*.md; measurements results/r3/NOTES.md.

## Changes (each has an off switch)

| Commit | Change | Class | Off |
|---|---|---|---|
| bd1af0671 | top-k radix select picks the digit with a wave suffix scan (was one thread walking 256 bins) | bit-exact | `GGML_CUDA_TOPK_SCAN=0` |
| 2683aacb2 | Q4_K K=2048 MoE down projection on the expert-dedup kernel (fell back to `mul_mat_vec_q_moe`) | tolerance | `GGML_MOE_V2_Q4K_DOWN=0` |
| e4e6f636e, 754b56a67 | decode-shaped f32 lightning indexer: 4 lanes per score evaluate quarters of the vec kernel's 32-leaf tree, combined by DPP quad permutes; one query per block | bit-exact | `GGML_CUDA_LI_DECODE=0` |
| 274cd8a02, 78d689a98 | fused one-workgroup top-k for short rows (<= 3072 columns); parallel tie gather in the radix path (one launch fewer) | bit-exact | `GGML_CUDA_TOPK_FUSED=0`, `GGML_CUDA_TOPK_TIEPAR=0` |
| 35f0325fc | tests: TOP_K at k = 512 over 1025..32780 columns and 1..33 rows; MUL_MAT_ID Q4_K K=2048 at 1..8 tokens | tests | - |

## Validation
- Bit-exact items: op-level byte A/B harnesses (results/r3/tools/topk_ab.cpp, li_ab.cpp) against the 9th-build library:
  top-k 175 shapes (incl. 512-row prefill shapes, 65547/98306 columns, NaN, -inf k-th key, all ties), indexer 116 shapes
  (1..8 queries, 290..32770 pools, 32/64 heads, 5 mask patterns, 2 streams) - all byte-identical. Server decode oracle
  (greedy, top-20 logprobs, 3 concurrent streams + a 32k-depth single stream): the 9th build is byte-stable across
  sessions, and the round-3 build with the down-projection change off is byte-identical to it.
- Tolerance item (MoE down): perplexity KL with 3-token ubatches (decode-shaped) vs the 9th build: 0.0103 / 0.0108 in
  two runs; the decode noise reference (4-token vs 3-token ubatches, same build) is 0.0104 +- 0.0020; top-1 97.3 % in
  both. Prefill numerics are unchanged (ub512 KL vs the 9th build 0.000000).
- test-backend-ops: TOP_K 565/565, MUL_MAT_ID 1041/1041, LIGHTNING_INDEXER 147/147. Fable reviews of every commit:
  no blocking findings (should-fix items applied).

## Measurements (prod config, MTP on, same session, two rounds)

Per call (op harness, gfx908, us): top-k at 8258 columns x 3 rows 97.0 -> 77.0, 32770 x 3 157.5 -> 98.6, 32770 x 18
208.3 -> 113.5; indexer at 8258 pools x 3 queries 131.7 -> 72.5, 32770 x 3 406.9 -> 160.3, 1 query 103.5 -> 48.0.

Decode per verify step (greedy, 128 tokens from a cached prefix at the given depth; ms per step = tokens per step / t/s):

| depth | 9th build | round 3 | |
|---|---|---|---|
| 2k | 49.6 ms (45.3 t/s) | 46.9 ms (45.9 t/s) | -5.4 % |
| 8k | 51.6 ms (50.1 t/s) | 48.7 ms (53.6 t/s) | -5.5 % |
| 32k | 55.4 ms (42.0 t/s) | 52.5 ms (42.0 t/s) | -5.2 % |
| 96k | 68.4 ms (28.8 t/s) | 64.4 ms (32.6 t/s) | -5.9 % |

(t/s also moves with MTP acceptance, which the MoE down change perturbs per prompt; the per-step time does not.)
Mixed single-stream workload 54.2 / 54.4 -> 56.3 / 56.6 t/s. 6-stream aggregate runs ranged 34-47 t/s for the same
build in this session (noise). Prefill 8k 2955 / 2854 -> 2958 / 2818, 32k 3487 / 3319 -> 3324 / 3257 t/s (within the
session spread; the parallel tie gather also runs in prefill, `GGML_CUDA_TOPK_TIEPAR=0` restores the old gather).

## Dead ends and findings
- The fused one-workgroup top-k only wins on short rows: each phase is a serial chain of L2 round trips on one CU
  (8258 columns 102 vs 76 us for the multi-block radix path, 32770 columns 190-222 vs 94-166 us) - gated to <= 3072.
- The first decode indexer held all queries of a verify batch in one block and was slower than the vec kernel at 3
  queries (189 vs 131 us); one query per block fixed it.
- The -ub 3 perplexity path (thousands of pipelined 3-token ubatches per decode call) shows a sporadic run-to-run
  difference in ~1 of 3 runs with HIP graphs on (isolated positions with very different logits, median KL 0), on every
  build including the unmodified one; not seen with graphs off (2 runs), not seen at ub512 or in server decode. Open.

## Next levers
- sparse_attn + combine in decode (~1.4 ms per step, same per-block latency pattern as the old indexer).
- The radix top-k still spends ~9 launches per call; folding init / count into select (bit-exact) saves ~2 more.
- MoE prefill: LDS swizzle (PLAN-r3-moe.md M2).
