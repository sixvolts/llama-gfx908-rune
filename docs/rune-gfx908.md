# Qwen3.8-Flash-Next on gfx908 (MI100) — the "rune" changes

This branch is `danielhanchen/llama.cpp` @ `qwen4exp/mtp` (base d1a9235) plus the work done on **rune**
(10x MI100 in two XGMI hives + 2 PCIe orphans, ROCm 6.4.3 host) to serve Qwen3.8-Flash-Next UD-Q4_K_XL.
Every change below is either bit-exact against the code it replaces (byte-identical outputs, verified with
standalone oracles) or scheduling-only.

## Results (single node, 4x MI100 layer split, `-fa on`)

| metric | before | after |
|---|---:|---:|
| decode, single stream | 34.5 t/s | 45.6 |
| decode, single stream, MTP head | 48.9 | 62–63 |
| decode, 4 streams, no MTP | 59 aggregate | 96 |
| decode, 4 streams, MTP | 74 aggregate | 93 |
| prompt processing, trunk only (pp8192, `-b 8192`) | 747 t/s | 2052 |
| prompt processing with the MTP head attached (4k prompt, server) | ~600 | 1276 |
| decode at 32k context | 20 | 31 |

## What changed

### ggml-cuda (HIP, gfx908)
- `hc-fused.cu/.cuh` — fused HyperConnection kernels: `hc_combine`, `hc_mix_epilogue`, `scale_silu`, and the
  low-rank megakernel (`hc_down_silu_q8_0`, `hc_up_mix_q8_0`) replaying mmvq's lane partition and reductions
  so results are byte-identical. Hooks live in `ggml_cuda_try_fuse` (toggles: `GGML_HC_FUSE_DISABLE`,
  `GGML_HC_MEGA_DISABLE`, `GGML_HC_MEGA_J`).
- Grouped RMS-norm fusion through a reshape (`RMS_NORM -> RESHAPE -> MUL`).
- `mmvf.cu` — 8-way unrolled F32 column loop (order-preserving); CDNA1 column cap for F32/BF16 raised 3 -> 16
  (chunks of <= 8 columns): batched decode no longer falls into a single-workgroup rocBLAS SGEMM.
- `mmvq.cu` — per-graph q8_1 quantize cache for GEMVs sharing an activation (`GGML_CUDA_Q8_CACHE=0` off).
- `cpy.cu` — copy kernel for small contiguous same-type copies instead of `hipMemcpyAsync`.
- Contraction rules for bit-exact fusions: `#pragma clang fp contract(off)` where the original ops were
  separate kernels; explicit `fmaf` where the original was contracted.

### ggml-backend / llama
- Scheduler: one wait per split for host-resident inputs, async input copies (`GGML_SCHED_SYNC_INPUTS=1` restores).
- Constant graph topology: every ubatch graph gathers >= 1 output row, so the allocator never reallocates
  (a reallocation drains every device) between prompt ubatches.
- Pipeline parallelism kept on despite tensor overrides: `LLAMA_PIPELINE_PARALLEL=1`.
- `set_embeddings_nextn` re-reserves the graphs when the mode changes on a live context (the MTP driver
  enables it after creation; without this every prompt ubatch reallocated).
- Unmasked nextn rows are double-buffered per decode with a per-decode event on the exporting device:
  `llama_nextn_seq`, `llama_synchronize_nextn(seq)`, `llama_get_embeddings_nextn_seq(seq)` (src/llama-ext.h).
- `llama_context::synchronize()` returns early when no async work is pending (`LLAMA_SYNC_ALWAYS=1` restores).

### common / server
- MTP draft: hidden rows fetched with one pointer (the per-row getter synchronized the whole scheduler).
- `common_speculative_process_seq()` — draft catch-up waits only for the exporting device.
- Server: the draft catch-up is deferred by one prompt view and prompt views are capped at
  `LLAMA_SPEC_PROMPT_CHUNK` (default 2048) when a draft is attached, so the head's catch-up overlaps the
  trunk's next chunk.

## Production launch (rune)

```
HIP_VISIBLE_DEVICES=0,2,3,4,6 LLAMA_PIPELINE_PARALLEL=1 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -ngl 99 -fa on -t 16 \
  -dev ROCm0,ROCm1,ROCm2,ROCm3 --load-mode none -lzm off -ot 'per_layer_token_embd\.weight=CPU' \
  -md MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 -ngld 99 -devd ROCm4 \
  -c 786432 -np 6 -ub 512 -b 8192
```

Notes: `-lzm off` keeps the 28.8 GB n-gram table in RAM (lazy mode page-faults it from disk every token);
the *non-shared* MTP head is required to place the head on its own device (the shared one borrows the
trunk's embedding/output tensors); draft length 2 is the sweet spot; flash attention works on gfx908 for
this model (perplexity-checked at 8k chunks).

## Diagnostics added
`GGML_GALLOC_DEBUG=1` (allocator reallocation reasons, stderr), `GGML_SCHED_SYNC_TRACE=N` (backtraces of
scheduler synchronizes), `COMMON_SAMPLER_TRACE=1` (sampler phase times), `GGML_SCHED_DEBUG_REALLOC` (upstream).

## 2026-09-15: prefill campaign (5th production build)

Ground truth came from a `rocprofv3` kernel trace of 512-token prefill ubatches: the MoE expert GEMM (`mul_mat_q` with ids) was
49% of kernel time, dense projections through rocBLAS 23%, the gated delta-net recurrence 13%. Hardware counters ruled out LDS
bank conflicts; the MoE kernel was bound by exposed global-load round trips (one block per CU, load -> barrier -> MFMA -> barrier).

Bit-exact (verified byte-identical against the previous production library with `golden/mmq_ab`):
- `mmq.cuh`: both K halves of the activation tile are loaded together (second LDS buffer, two fewer barriers per K step).
- `mmq-load-tiles.cuh`: `mmq_x_prefetch<Q4_K|Q5_1|Q8_0>` split the tile loaders into register load / LDS store halves so the
  next K step's weights and activations are fetched during the current step's MFMAs (`__builtin_amdgcn_sched_barrier`).
- `mmq-vec-dot.cuh`: the MFMA column loop is dispatched once per tile to 1/2/4/full straight-line variants sized by the tile's
  valid rows (`j_lim`); a runtime bound or an early break inside the loop spills the accumulators and is slower.
- `mmid.cu`: the ids helper scans each expert's tokens with 8 warps over in-order slices (same output order).
Result: MoE GEMM 3914 -> 2908 ms, ids helper 174 -> 56 ms per 2x2048-token trace.

Tolerance-only (gated by KL divergence against saved reference logits, `llama-perplexity --kl-divergence`; the previous build
is itself non-deterministic at prefill with max KL 0.18 run-to-run, the new build lands at max KL 0.17 against it and is
deterministic run-to-run):
- `gated_delta_net.cu`: `gated_delta_net_lpc_cuda` for batches of >= 8 tokens: 16 lanes per state column, 4-step DPP sums
  (`__builtin_amdgcn_update_dpp`), one wave per block, per-lane float4 k/q register ring 8 tokens ahead, `__launch_bounds__(64, 1)`
  (at 2 waves/SIMD the ring spills to AGPRs and every spill of a pending load forces a `vmcnt(0)` drain). GDN 1026 -> 452 ms.
  `GGML_GDN_LPC=0` disables, `=2` forces for all batch sizes.
- `ggml-cuda.cu`: gfx908 dense Q8_0 GEMMs with >= 6144 rows and > 128 columns run on MMQ instead of rocBLAS (Tensile picks a
  64x32 macro-tile at ~28 TFLOPS; MMQ is 1.4-1.5x faster). `GGML_MMQ_DENSE_MIN_M=0` disables.
- `mmvf.cu`: thin F32 projections (M <= 64) run through the F32 vector kernel with swapped roles (each activation column is a
  channel, the M weight rows are broadcast vector columns). `GGML_MMVF_SMALL_M=0` disables.

Measured on the production layout (Hive A + head on the orphan, 6 x 128k, `-b 8192`): pp8192 2052 -> 2704 t/s, server 5.5k
prompt 1438 -> 1716 t/s, decode unchanged (60 t/s chat with MTP, 45 t/s without). Dead ends: sizing the MoE J tile from the
mean rows per expert (routing is heavily skewed; -8%), I=64/occupancy-2 tiles (-19%), per-load branches to skip unused
activation rows (-10%), forcing all dense GEMMs onto MMQ (-3%), prefetching inputs in the warp-per-column GDN kernel (issue-bound).
Diagnostics: `GGML_MMQ_DUMP_IDS=1` (rows-per-expert histogram), `GGML_CUDA_DUMP_MM=1` (shapes routed to rocBLAS).

## 2026-09-23: server-side fixes from the kernel review (6th production build)

- `gated_delta_net.cu`: with MTP the server asks the GDN op for K>1 rollback snapshots, and the per-token snapshot check
  inside the straight-line D-token group made the compiler drain the load ring every token (2.47x slower). Tokens that
  never snapshot now run in a separate branch-free loop. Byte-identical output (A/B vs the previous library at n = 3..2048,
  K = 1 and 3); K=3 at 512 tokens 3.27 -> 1.68 ms, at 2048 tokens 10.8 -> 4.5 ms.
- Prompt cache (configuration): `--no-cache-idle-slots --cache-ram 65536`. In non-unified KV mode idle slots keep their
  state on the GPU and a slot is always saved before it is reused, so re-saving every idle slot on every new task only
  cost a multi-GB copy per request (and thrashed the default 8 GiB cache). Follow-up turn at 24k context: TTFT 1.56 -> 0.49 s.
- `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` (run.sh): THP is `madvise` on this host, so the server's large host
  allocations (prompt-cache entries, 160 MB context checkpoints) paid a page fault per 4 KB; with huge pages the cache-entry
  allocation drops 0.8-1.1 s -> 0.44 s and a 64-token turn at 24k depth 504 -> 407 ms.
- `server-context.cpp`: `LLAMA_SERVER_BUSY_PROMPT_CAP=<tokens>` (default off) caps prompt tokens per iteration while
  other slots generate. Measured trade-off (30k prompt arriving while a slot generates): off 17.9 s TTFT / generating slot
  1.5 t/s, 4.9 s worst stall; 2048: 22.8 s / 2.6 t/s / 1.7 s; 1024: 26.5 s / 4.0 t/s / 1.0 s.
Result on the production layout: 5.5k prompt 1716 -> 1988 t/s, decode unchanged (59 t/s chat).

## 2026-09-23: decode path, first pass (7th production build)

- `topk-moe.cu` (CDNA only): the 10-round expert selection uses DPP row moves plus one `ds_swizzle` per reduction
  instead of 5 butterfly shuffles on value and index. Byte-identical ids/weights (300 randomized cases incl. heavy
  ties); 14.6 -> 10.1 us per call (x48 per token).
- `concat.cu`: non-contiguous concats with short rows (the GDN conv-state concat for multi-token decode, [3+n, 10240]
  with a transposed src1) use one thread per element instead of one 256-thread block per row. Pure copy.
- `ggml-cuda.cu`: the HIP graph cache key mixes the node count and first/last node shapes into nodes[0]; graphs of
  different shapes from the same context (the MTP head's 3-row catch-up and 1-row drafts, the trunk at different slot
  counts) no longer evict each other. hipGraphExecUpdate per 192-token MTP run 91 -> 16.
- `llama-context.cpp`: the "at least one output row" workaround is limited to pipeline-parallel contexts, so the
  head's zero-output catch-up decode no longer runs a 0.77 ms lm_head on a dummy row.
Measured (production layout, llama-cli): MTP 59.4 -> 61.8 t/s, no-MTP tg64 44.3 -> 45.2 t/s.
`--backend-sampling` (GPU target sampling) saves ~1.4 ms per MTP step (36.0 vs 37.4 ms) but halves long-prompt prefill
with the head attached (1954 -> 956 t/s at 5.5k); not enabled yet.
- `common/sampling.cpp` (8th build): top-K prefilter for the CPU sampler chain. When top-k (<= 62) comes first among the
  active samplers and everything before it is a no-op (neutral penalties, DRY off, top-n-sigma off, only -inf logit
  biases / model suppress tokens), cur_p is built from the 64 largest logits (vectorized block-max scan) instead of the
  full 248k vocabulary. Falls back to the full path on any tie among the top k+1 or any non-finite logit, so the sampled
  token is identical (3000/3000 randomized trials incl. ties/NaN; seeded server generations hash-identical).
  0.48 -> 0.19 ms per sampled row; MTP step 37.2 -> 36.1 ms. COMMON_SAMPLER_PREFILTER=0 disables it.
- `ggml-alloc.c`: GGML_GALLOC_DEBUG also prints the consumers of a tensor that forces a reallocation.
- Not adopted: `--backend-sampling` (GPU target sampling). Its per-row sampler subgraphs change the graph node count
  between decode and prompt ubatches, so after any decode each prompt ubatch reallocates (a KV-sized QSA input grows by
  512 per ubatch) and pipeline parallelism drains: 5.5k prefill 1954 -> 956 t/s.

## 2026-09-23: MTP verify path (9th production build)

The MTP n=2 step verifies 3 tokens in one trunk pass; several decode-only paths fell back to generic kernels at nt=3.
- `hc-fused.cu` / `ggml-cuda.cu`: the HC low-rank megakernel (down GEMV + scale/silu + per-block q8_1 replay + up GEMV +
  sigmoid mix epilogue) now takes 1..4 token columns; each column accumulates exactly as the ncols=nt mmvq path it
  replaces. Its butterflies (`hc_shfl_xor`) use one ds_swizzle and four DPP moves per 32-lane level set instead of
  ds_bpermute (same pairings, so every lane gets the same value as `__shfl_xor_sync`; row_ror:4 is only exact because the
  order is top-down), the replay skips the q8_1 block sum (Q8_0 x Q8_1 never reads it), and the epilogue runs one
  thread per (column, output). Byte-identical to the unfused graph (golden/hc_bitcheck, nt 1..4, 5 shapes, all J).
  hc_up_mix at nt=3 26.9 -> 17.4 us; at nt=1 11.5 -> 10.2 us. In-place fusion is limited to nt=1.
  GGML_HC_MEGA_MT_DISABLE=1 restores the nt=1-only matcher; GGML_HC_MEGA_J_MT picks J (default 4) for nt > 1.
- `gated_delta_net.cu`: for 2..4 tokens (S_v=128) the warp-per-column kernel preloads every token's k/q/v/g/beta before
  the recurrence (constant-trip loop with `continue`, so it fully unrolls and stays in VGPRs). Byte-identical vs the
  previous library at n = 1..5, K = 1 and 3; test-backend-ops GATED_DELTA_NET 36/36; 3 tokens 25.3 -> 22.6 us.
  GGML_GDN_PRELOAD=0 disables.
Measured (server, production layout, seeded chat A/B back to back): MTP step 36.23 -> 35.69 ms (65.6 -> 66.7 t/s pooled);
seeded generations hash-identical to the 8th build. Without the DPP butterflies the nt=3 megakernel was a net loss
(+0.8 ms/step): inside a HIP graph the unfused small kernels cost little, so the fused path has to win on latency.

## 2026-09-23: few-token MoE kernel with expert dedup (10th production build)

`mmvq-moe.cu` (tolerance class): MUL_MAT_ID for 1..4 tokens with Q4_K gate/up (fused SWIGLU) and Q5_1 down at the
Flash-Next shapes (K = 2560 / 640). mul_mat_vec_q_moe ran one warp per (token, slot) through vec_dot_*_q8_1, re-unpacking
every weight fragment per token: at 3 tokens it was VALU/latency bound (7.6M VALU instructions for gate/up, ~49% VALU
busy, ~640 GB/s) and re-read each shared expert per token. Now a lane owns whole quant blocks (Q4_K: a 64-weight pair with
16-byte loads; Q5_1: one block), unpacks them once and applies them to every token routed to that expert; activations are
staged in LDS, work items are streamed with a one-item register prefetch, and the batch routing is read with one load +
ballot so each distinct expert is computed by one block (verify batches in real chat: 69-77% of pairs are distinct).
Each 32-weight block is scaled after its integer dot product (the Q4_K min term uses the exact fp32 d8*sum(q8) as the
reference does), so the summation order differs: 3.5e-5..1e-4 relative RMS vs the reference at op level.
Per layer on MI100 (golden/moe_bench, random routing): nt=1 65 -> 44 us, nt=3 147 -> 114 us (73 us when the 3 tokens
share experts), nt=4 189 -> 145 us. Q5_K gate/up (1 layer), Q8_0 down (5 layers) and the Q8_0 MTP head keep the old path.
KL gate (perplexity with -ub 3 so the decode kernels run, 8x512 tokens of ppl_long.txt, base = reference kernels):
mean KLD 0.0186, top-1 94.8%, PPL ratio 0.9997 +- 0.005; for scale, the prefill kernels (-ub 512) vs the same base give
mean KLD 0.031, top-1 93.5%; the reference against itself 0.
Server (production layout, seeded chat): MTP step 36.08 -> 34.12 ms, 65.9 -> 70.5 t/s pooled.
GGML_MOE_V2=0 disables; GGML_MOE_V2_Q4K / GGML_MOE_V2_Q51 = 10*waves_per_block + row_groups (default 41);
GGML_MOE_DEDUP_STATS=1 prints distinct-expert counts per device at exit. `reduce-dpp.cuh`: the exact DPP butterfly
helper, shared with hc-fused.cu.

## 2026-09-23: orphan GPU and speculative-loop measurements (experimental, not in production)

- `ggml-cuda.cu` (c1fb8c054): device pairs without peer access (the PCIe-only orphans) copy through a pinned host ring
  instead of hipMemcpyPeerAsync, which faulted. This makes a 5-stage layer split with the orphan possible
  (`-dev ROCm0..ROCm4 -ts 10,10,10,10,8`, head on ROCm4 as before).
- 5-stage results on Hive A: trunk VRAM 28.9-31.8 -> 24.1-26.2 GB/GPU at 6 x 128k; 7.4k prompt 2095 -> 2207 t/s; 114k
  prompt 1319 -> 1393 t/s; MTP step unchanged. Decode logits identical to the 4-GPU split; the prefill path depends on the
  split points at KLD 0.006 (same tolerance class as prefill vs decode kernels, 0.031). 6 x 256k loads but leaves
  ~0.1 GB free on GPUs 2 and 4 (the 12 attention layers, which hold the KV, land 2/3/2/3/2); ~46 KB of VRAM per context
  token. At 220k depth: prompt 987 t/s average, MTP decode 22.5 t/s (31 t/s at 114k).
- `--spec-draft-n-max 3`: 70.1 vs 70.5 t/s, no gain (output hash-identical: MTP is lossless).
- LLAMA_SPEC_TIMING=1 (server): median host timeline of the MTP loop. Single stream, 10th build: step 33.6 ms =
  target verify 28.7 + draft 3.45 (2 head decodes: 0.31 ms enqueue + 1.39 ms GPU incl. the 0.77 ms lm_head each) +
  catch-up 0.55 + sample/accept/rest 0.96 + build 0.09. Draft sampling already runs on the GPU (top-k 10).

## 2026-09-23: HIP graph cache correctness under changing batch shapes (not yet in production)

`ggml-cuda.cu`: the graph cache was keyed on the first node's struct address (+ shapes) and compared node properties
including struct pointers. When batch shapes alternate (verify batches of 2/3/4 rows with adaptive MTP drafts; in
production, whenever the number of busy slots changes) every shape change rebuilt the llama graph, the scheduler re-created
the split's nodes at new addresses and picked another input copy, and the cache restarted its 2-call warmup (2 eager runs
+ a ~7 ms capture of 2341 nodes). Worse, the seeded output under alternating shapes (5649af6d...) differed from fully eager
execution (GGML_CUDA_DISABLE_GRAPHS=1: 90b15b34...): a captured instance was being reused for a computation it did not
match. Now the key is the data layout (first/last node data pointers, the first node's input data pointers, n_nodes and
every node's shape/op) and the property comparison ignores struct pointers, names and the data pointer of empty views.
Graph runs equal eager runs bit-for-bit and are deterministic across runs; under alternating shapes graph reuse went
29% -> 75% of launches (verify 39.9 -> 35.2 ms/step; fixed shape 28.7). GGML_CUDA_GRAPH_WARMUP=0 captures on the first
call; GGML_CUDA_GRAPH_STATS=1 / GGML_CUDA_GRAPH_DEBUG=1 report reuse counts and the first mismatching node.
Validation rule for any graph-cache change: seeded chats (scratch chatab2.py) must hash-equal GGML_CUDA_DISABLE_GRAPHS=1.
The remaining cost of a shape change is the llama graph rebuild path (~0.75 ms build + ~0.8 ms sched alloc + ~1.6 ms
extra enqueue), which is why per-step adaptive draft lengths measure +1.3% today and +5.9% without it.

## 2026-09-23: dense Q8_0 GEMV for 1..4 columns (mmvq-q8.cu)

The 3-row MTP verify spends ~7.5 ms/step in ~300 dense Q8_0 GEMVs (attn_qkv/gate/q/k, ssm_out, shared-expert
gate/up/down, output) at 160-700 GB/s: mul_mat_vec_q reads the 34-byte Q8_0 blocks with 16-bit loads and its runtime
K loop keeps little in flight. `mmvq_q8_0_v2<NB, ncols, RPB>` replays mul_mat_vec_q's GCN partition exactly (nwarps 2,
kbx = tid/4 + 32*it, kqs = 2*(tid%4), same _impl, warp-1 partials through shared memory, warp_reduce_sum<64>) with the
K loop unrolled at compile time (K = 640/2560/6144), every load issued first and one 8-byte load per lane and block
(2-byte aligned; CDNA accepts it). Bit-identical to the reference on all 8 model shapes x 1..4 columns (golden/gemv_bench).
Rows per block is a free knob (a row's order is one lane sequence): 1 for one column, 4 for 2..4 columns on >= 2048 rows.
Isolated: 6144x2560 at 3 cols 28.3 -> 24.0 us, at 4 cols 33.4 -> 26.6; output layer at 1 col 804 -> 717 us.
Server: MTP step 34.26 -> 33.62 ms (2 drafts), 39.87 -> 39.13 (3 drafts); seeded outputs unchanged.
GGML_MMVQ_Q8_V2=0 disables; GGML_MMVQ_Q8_RPB=<n> overrides rows per block.

## 2026-09-23: MTP head trims (top-k, HC combine with gathered rows)

- `top-k.cu`: for k <= 32 on long rows (the draft sampler's top-10 over the 248k vocabulary) a two-kernel partial
  top-k (64 blocks keep their slice's k largest by k rounds of block argmax, one block merges) replaces the radix select's
  11 launches: 127 -> 46 us per call on MI100, exactly the k largest values (ties: lowest index). GGML_CUDA_TOPK_SMALL=0
  restores the radix path.
- `ggml-cuda.cu`: the HC combine fusion also matches REPEAT, GET_ROWS, SCALE, UNARY, SCALE, RESHAPE, MUL, ADD (the row
  gather that appears when only some rows are output: the trunk's last layer and every layer of the MTP head); the gather
  runs as its own op, the other seven fuse as before. GGML_HC_FUSE_DEBUG=1 reports any hc_combine/mtp_hc_ ADD left
  unfused; GGML_CUDA_GRAPH_DUMP=<n_nodes> prints the first graph with that many nodes.
- Tolerance-class dense GEMV (lane owns whole Q8_0 blocks, LDS activations, prefetch) was tried and dropped: slower than
  the bit-exact unrolled kernel on every model shape (e.g. 6144x2560 at 3 columns 28.7 vs 24.0 us).
Server (seeded chats, hash-identical): draft phase 3.27 -> 3.13 ms, MTP step 33.62 -> 33.34 ms.

Addendum (pool epoch): captured graphs also bake in the addresses of the context's memory-pool temporaries (quantized
activations, split-K partials), which no node property covers. A layout key that matches a rebuilt graph could therefore
replay a capture whose pool buffers had since moved: under 3 concurrent streams the candidate diverged from eager runs and
faulted ("Memory access fault by GPU node-6"). `ggml_cuda_pool::epoch` now counts every map/unmap of pool memory, each
captured graph records the epoch it was captured under, and a mismatch forces a re-capture (even on the uid shortcut).
Oracles: 3 concurrent seeded streams (slots joining and leaving) match GGML_CUDA_DISABLE_GRAPHS=1 three times in a row;
the single-stream alternating-shape case matches eager too. GGML_CUDA_GRAPH_KEY_MODE=0 / GGML_CUDA_GRAPH_PROPS_MODE=0..2
restore the old key / property comparison for bisection.
