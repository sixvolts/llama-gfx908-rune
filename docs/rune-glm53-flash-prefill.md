# GLM-5.3-Flash prefill campaign on rune (2026-10-04/05): 8th build

Goal: raise prompt-processing (prefill) throughput of the production GLM-5.3-Flash server (10x MI100 gfx908, 9 pipeline
stages + MTP drafter, 6 slots x 128k) toward 1000 tok/s without changing the model file, model quality, decode speed or
slot capacity. Four lanes worked in parallel (pipeline, gemm, seqmix, agentic), each lane's result was reviewed
adversarially and fixed, and this branch (`glm53/prefill-final`) merges all four.

**Result (same session, interleaved with the 7th build, wall-clock t/s including HTTP):** 2048 tokens 443 -> 975,
8192 556 -> 1892, 32768 630 -> 1826, 65536 581 -> 1517, 100000 544 -> 1352, two concurrent 100k prompts 544 -> 1471
aggregate. **1000 t/s is passed at every length from 8k up; 2k reaches 975 wall / 1255 server-side.** Agentic 3-session
replay: sum TTFT 530 -> 211 s. Decode unchanged within noise, KL 0.006872 (tolerance budget 0.0167), 6 x 128k fits
with 4.8 GiB free on the fullest GPU.

## 1. What changed

Merged branches (no conflicts): `glm53/prefill-pipeline` 1e3e79f76, `glm53/prefill-seqmix` 6f1182e45,
`glm53/prefill-gemm` 90175852a, `glm53/prefill-agentic` 6db87938e, all based on prod 432673179.
Classes: **config** = flags/env, **bit-exact** = KL 0.000000 vs the same-ub reference, **tolerance** = numerics move,
KL-gated below the model's own ub256-vs-ub512 floor.

### Pipeline (why the 9 stages never overlapped, and the fix)
| Commit | Change | Class | Off switch |
|---|---|---|---|
| a1d3db581 | glm5next: expand the layer input before the KDA state gathers - removes the 0-input back-splits at stage boundaries (18 -> 10 splits per prompt graph) whose host `event_synchronize` waited for the next GPU's previous ubatch | bit-exact | `LLAMA_GLM5_KDA_ORDER=0` |
| 5abd69d08 | sched: rotate pipeline copies on reused prompt graphs (graph reuse skipped `sched_alloc_graph`, so `cur_copy` never advanced and every split waited for the previous ubatch on its GPU); HIP-graph keys per input copy | bit-exact | `LLAMA_PP_ROTATE_MIN=0`, `GGML_CUDA_GRAPH_KEY_INPUTS=0` |
| b9676d62f | runtime pipeline depth `GGML_SCHED_COPIES` (cap 16; build default stays 4) and host-staged crossing ring `GGML_CUDA_STAGE_RING` | config | unset (defaults 4 / 4) |
| 9099b7a26 | stage host user inputs once per rotated ubatch (pinned snapshot, async H2D per split) | bit-exact | `GGML_SCHED_STAGE_INPUTS=0` |
| 7e0683135 | one view of the KQ mask / pool inputs per graph instead of per DSA layer (-2.3/-2.6 GiB on gpu4/gpu9 at 8 copies; this is what keeps 6 x 128k) | bit-exact | `LLAMA_GRAPH_VIEW_CACHE=0` |
| a9416c730, 821fca853, 1e3e79f76 | skip the source-stream wait when inputs are staged; review fix: host-wait every backend's last use of a copy before a rotated ubatch writes it (ordering by construction) | bit-exact | `GGML_SCHED_NO_SRC_WAIT=1`, `GGML_SCHED_ROTATE_WAIT=0` (A/B only) |
| 600655ba7 | diagnostics `GGML_SCHED_TIMING`, `LLAMA_GRAPH_TIMING` | config | off by default |
| fc0030a03 | `LLAMA_KQ_MASK_F16=1` (f16 mask without FA) | bit-exact, experimental | off by default; no long-context KL - keep off |
| d0065ce36 | `LLAMA_SCHED_COPIES_SINGLE=1` (scheduler copies for the drafter) | config, REJECTED | off by default; costs -4 % single-stream decode |

### GEMM (MoE MMQ was 48.5 % of kernel time)
| Commit | Change | Class | Off switch |
|---|---|---|---|
| d2b8fcb5d | Q4_K MMQ prefetch: scale unpack in registers (prod kernel spilled 52 B/lane to scratch, so every prefetched load was waited on right after issue) | bit-exact | revert |
| 01091f1ea | MoE MMQ tiled schedule on CDNA (compact device list of non-empty (expert, column tile), no stream-k) | **tolerance** (8k KL 0.006872 / top-1 97.31 %; 32k 0.005555 / 97.27 %) | `GGML_MMQ_MOE_TILED=0` (see caveat in sec. 7) |
| c8a1be43b | Q8_1-layout MFMA vec_dot k01 unroll | bit-exact | revert |
| fdc6acc2a | CDNA Q4_K tiles: 64 rows, 256 threads, 2 blocks/CU; float2 dm in LDS; 16-byte loader | bit-exact | compile-time `-DGGML_MMQ_Q4K_I128=1` etc. |
| a64ec1135 + 90175852a | thin Q8_0 projections (M <= 1024) to MMQ - now OPT-IN, default off (200-1000x the per-op error of rocBLAS for ~1.7 % trunk time) | tolerance, not adopted | `GGML_MMQ_DENSE_THIN_M` default 0 |
| 2f7f69d65 | diagnostics `GGML_CUDA_DUMP_MM_ALL`, `GGML_MMQ_DUMP_IDS_FILE` | config | off by default |

### Sequence mixing (all bit-exact, byte-identical op outputs)
| Commit | Change | Off switch |
|---|---|---|
| 63778feb1, 6ca812a6c | tiled lightning indexer (bit-exact tree order) that skips tiles hidden by the causal mask - the vec kernel did work for the whole 32k-padded n_kv; 13.2 -> 2.1 ms per DSA layer per ubatch at 32k | `GGML_CUDA_LI_TILED=0`, `GGML_CUDA_LI_TILED_STREAM=0` |
| 2f1e21903 | tiled dim-0 concat for a transposed src1 | `GGML_CUDA_CONCAT_TILED=0` |
| 692fe86a0 | KDA conv-input assembly fused at prefill (concat 1.18 ms + copies -> 0.16 ms per layer) | `GGML_CUDA_KDA_CONV_ROWS_PREFILL=0` |
| f21942fe4 | KDA recurrence reads a precomputed exp(g) (1.35x) | `GGML_GDN_EXPG=0` |
| 4c5c861b4 | sparse DSA attention: two head groups per block, double-buffered V^T (1.45x) | `GGML_CUDA_SPARSE_HG2=0` |
| bafa7f747 | dsv4_hc_post for hc=4, one thread per (element, token) (2.27x) | `GGML_CUDA_HC_POST4=0` |
| 6f1182e45 | eval/perf tests for the above | - |

### Server / agentic (tools/server only)
| Commit | Change | Class | Switch |
|---|---|---|---|
| dc15a36bd | `LLAMA_CKPT_TAIL_ALIGN=1`: take the far end-of-prompt checkpoint at a natural batch boundary instead of forcing an isolated 512-token ubatch at n-516 (it ran unpipelined, ~1.2 s per request) | tolerance (last batch's ubatch partition; 48-prompt top-40 KL 0.0171 vs same-metric floor 0.0225) | default off; ON in run.sh.next |
| 723dbea91, b5e62e84b | `LLAMA_PROMPT_SJF=<t/s>`: fill prompt batches by estimated finish time; do not top up a batch in which a prompt completes (short turn behind a long one: 46 -> 9 s TTFT) | tolerance (batch composition, only with >1 pending prompt) | default off; ON (1000) in run.sh.next |
| bdb97ef2f, 4c34dc5b3, 828ff5f1b, 8af4dae07, 6db87938e | `LLAMA_PREFIX_SHARE=<min tokens>`: shared system-prompt + tools snapshot in the host prompt cache (new conversation loads it in ~50 ms instead of re-prefilling ~8.8k tokens); review fixes: drafter state stored, tie-break prefers the snapshot, empty-slot takeover, `LLAMA_PREFIX_SHARE_MAX` (default 4) | bit-exact (state copy; 4/4 x 300 greedy tokens identical to cold) | default off; ON (4096) in run.sh.next |
| 066fd745d (+8af4dae07) | `LLAMA_SLOT_CACHE_LCP=1`: reload a parked conversation from the host cache when it reuses more | config, experimental | opt-in, OFF in run.sh.next |

## 2. Production configuration of the 8th build (`~/prod/glm53/run.sh.next`)

Unchanged from the 7th build: layout (`-dev ROCm0..8 -ts 7,4,4,5,5,5,5,5,7`, drafter on ROCm9), `-ub 512`,
`-c 786432 -np 6`, `-fa off`, `LLAMA_KV_PAD_PREFILL=32768`, `GGML_CUDA_GRAPH_FULL_BATCH=512`, sampling, cache flags.
Changed: `-b 2048` -> `-b 8192` and

```
export GGML_SCHED_COPIES=8               # 8 ubatches in flight across the 9 stages (needs the rotation commits)
export GGML_CUDA_STAGE_RING=8            # host-staged island crossings (gpu4->gpu5, gpu9->gpu6) can run 8 deep
export LLAMA_SERVER_BUSY_PROMPT_CAP=2048 # while a slot generates, prompt batches carry <= 2048 prompt tokens (fairness)
export LLAMA_CKPT_TAIL_ALIGN=1
export LLAMA_PROMPT_SJF=1000
export LLAMA_PREFIX_SHARE=4096
```
Kernel switches need nothing (new paths default on; thin-M default off). Build: the standard gfx908 cmake line,
`GGML_SCHED_MAX_COPIES` left at its default (depth is a runtime setting now).

## 3. Measurements (2026-10-05, one session, interleaved A/B: base, final, final, base, ..., then repeats)

Base = frozen 7th-build bins + prod config; final = this branch + the config above (results/final/ in the campaign
directory: A-*, A2-*, B-*, kl.out, trace-final-report.txt, c6-*.out). Every server fresh, warm-up request first.

### Cold prefill (bench_prefill.py, cache_prompt=false; 3 runs per side: 2 in job A + 1 in job A2; median)
| Prompt | Base server t/s (runs) | Final server t/s (runs) | Base wall | Final wall | Gain (wall) |
|---|---|---|---|---|---|
| 2048 | 462.6 (464/463/450) | 1254.8 (1255/1242/1311) | 443.0 | 974.8 | 2.20x |
| 8192 | 564.0 (566/564/558) | 2122.8 (2023/2125/2123) | 556.2 | 1891.6 | 3.40x |
| 32768 | 632.1 (632/635/621) | 1876.4 (1876/1759/1948) | 629.5 | 1825.7 | 2.90x |
| 65536 | 582.3 (582/597/575) | 1534.5 (1511/1751/1534) | 581.0 | 1516.8 | 2.61x |
| 100000 | 545.2 (545/561/529) | 1360.9 (1361/1484/1359) | 544.3 | 1351.8 | 2.48x |
| 2 x 100000 concurrent (aggregate) | 543.5 (549/538) | 1471.0 (1457/1485) | | | 2.71x |

"Wall" includes HTTP and the server's host-cache save of the slot being reused (`prompt cache update`, 0.2-0.6 s,
depends on what that slot held; it is the same code in both builds). On fresh servers with free slots (diag runs) wall
equals server time: 2048 1383, 32768 1919 t/s. The 6.4k prefill inside ab_workload: 523.6 / 510.3 -> 1020.8 / 1036.2.

Contribution check (same final bins, prod geometry `-b 2048`, 4 copies, tail-align on, diag-prodgeo): 32768 1152,
2048 1089 t/s - i.e. kernels + tail-align alone give ~1.8x at 32k; the pipeline geometry adds the remaining ~1.67x.

### Agentic
| Workload | Base | Final |
|---|---|---|
| warm 8192-token prefix + d new tokens, TTFT (median of 3), d = 1024 / 2048 / 4096 / 8192 | 3.18 / 4.64 / 7.76 / 13.98 s | 1.38 / 1.74 / 2.62 / 4.02 s |
| 3 sessions x 8 turns tool-loop replay (agentic_replay.py): sum TTFT | 530.3 s | 211.3 s (-60 %) |
| same: median TTFT of turns > 0 | 24.39 s | 9.35 s |
| same: prompt tokens / sum TTFT ("effective" t/s) | 1541 | 3859 |
| same: tokens actually prefilled | 169,936 (20.8 %) | 152,070 (18.6 %) |
| first turn of sessions 2 and 3 (shared 8.8k system prompt) | full prefill | 49-52 tokens prefilled (snapshot) |
| generating stream during a 32k prefill: max / p99 inter-token gap; prefill t/s meanwhile | 4.89 / 3.35 s; 597 | 2.26 / 1.71 s; 1296 |

The replay is not a fixed workload (assistant turns are sampled); the warm-prefix and cold numbers are the clean A/Bs.

### Decode (must not regress)
| | Base A | Base A2 | Final A | Final A2 |
|---|---|---|---|---|
| ab_workload single stream (MTP), accept | 54.60, 0.806 | 53.86, 0.797 | 55.03, 0.820 | 54.09, 0.806 |
| conc sweep 1 / 3 / 6 streams x 400 tok (sum of per-stream decode t/s) | 45.2 / 64.2 / 79.2 | 45.3 / 63.9 / 79.2 | 43.2 / 63.6 / 78.3 | 45.6 / 63.6 / 77.9 |
| ab_workload conc6 aggregate | 56.3 | 56.6 | 45.8 | 44.4 |

The conc6 gap triggered a dedicated check (conc6.py, 4 reps per fresh server): base 46.7 / 47.5 / 55.5 / 53.0, final
46.6 / 51.9 / 51.5 / 46.6, final without SJF 49.9 / 48.1 / 56.7 / 58.0, final bins at prod geometry 49.9 / 55.9 / 54.5 /
52.6 t/s - the spread inside one config (47-58) covers every difference, so ab_workload's conc6 is noise here. The
6-stream conc sweep (fixed 400 tokens) reads 1.4 % lower on the final in both reps; treat as at most ~1 %.

### Quality (KL gate, wikitext-2, vs kl/base-prod7-ub512-*.kld)
| Gate | Mean KLD | Max | Same top-1 | PPL(Q) / PPL(base) |
|---|---|---|---|---|
| ctx 8192 x 6 chunks, run 1 | 0.006872 | 1.2615 | 97.306 % | 2.687054 / 2.688253 |
| ctx 8192, run 2 | 0.006872 (identical) | | 97.306 % | |
| ctx 8192, `GGML_CUDA_DISABLE_GRAPHS=1` vs graphs-off base | 0.006872 (identical) | | 97.306 % | |
| ctx 32768 x 2 chunks | 0.005555 | 0.4387 | 97.272 % | 3.359468 / 3.361381 |
| floor (prod ub256 vs ub512) | 0.016712 (8k); 0.019587 (32k) | 2.27 | 95.94 % (8k); 95.34 % (32k) | |

The combined build reproduces the gemm lane's tiled-MoE-only result in every printed digit at both contexts, so every
other merged change is bit-exact in combination, and the only numerics change is `GGML_MMQ_MOE_TILED`. (kl.sh cannot
see the server-side tail-align / SJF batch partitions; those were gated separately by the agentic lane.)
test-backend-ops on the merged build: LIGHTNING_INDEXER 147/147, CONCAT 185/185, GATED_DELTA_NET 38/38, SPARSE_ATTN
38/38, DSV4_HC_POST 5/5, KDA_CONV_STATE 90/90, MUL_MAT (q4_K/q5_K/q8_0) 143/143, MUL_MAT_ID 258/258.

### VRAM (MiB of 32752, peak sampled every second over the whole job incl. 100k and 2 x 100k prefills)
| GPU | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 21796 | 7568 | 20589 | 20589 | 27302 | 24762 | 25401 | 24762 | 24770 | 27304 |
| final | 23390 | 7557 | 22274 | 22274 | 27792 | 26448 | 27090 | 26448 | 26456 | 27792 |

Fullest GPUs (gpu4 / gpu9, the two 2-DSA stages): 27792 MiB = 4.8 GiB free; 6 x 128k unchanged. Host: the staged-input
snapshots pin up to ~256 MiB per copy at 128k context (~2 GiB at 8 copies) and never shrink.

## 4. Where the time goes now, and the physics

rocprofv3 trace of the final build (trace-final-report.txt; tracing slows this build much more than the old one -
32k ran 1107 t/s traced vs 1876 untraced - so use it for kernel times, not for overlap):

| per 512-token ubatch at 32k | 7th build | 8th build |
|---|---|---|
| trunk GPU time summed over 9 stages | 1312 ms | 802 ms (-39 %) |
| slowest stage | 165 ms (S7) | 106 ms (S0) |
| stage ms S0..S8 | 156/118/120/162/145/150/147/165/149 | 106/73/73/95/89/88/90/96/92 |
| MTP drafter | 40.5 ms, 0.4 ms overlapped with the trunk | 21.2 ms, 73 % overlapped |
| wall per ubatch (untraced) | 822 ms | 273 ms |
| mean busy stages (untraced, = GPU time / wall) | 1.6 | 2.9 |

Kernel shares now (32k): MoE MMQ 43 %, rocBLAS 15.5 %, dense MMQ 10.3 %, KDA 7.7 %, sparse attention 4.7 %,
dequant/convert 3.8 %, indexer 3.0 % (was 11.9 %), norm 2.6 %, elementwise 2.6 %, HC 1.8 %, concat 0.1 % (was 3.6 %).

Ceilings for this model on this machine (35.9 GFLOP per token at 32k; results/baseline/physics.py):

| Bound | t/s at 32k | vs measured 1876 |
|---|---|---|
| Compute, 100 % MFMA on all 9 stages, perfect overlap | ~46,300 | 25x |
| Compute, realistic 30 % MFMA, perfect overlap | ~13,900 | 7.4x |
| HBM: all 181 GB of weights read once per ubatch, pipelined | ~23,300 | 12x |
| PCIe Gen3 host-staged island crossings as a pipeline stage | ~51,000 | 27x |
| KDA serial recurrence | ~36,000 | 19x |
| Pipeline with today's kernels, unbounded overlap (512 / slowest stage 106 ms) | ~4,840 | 2.6x |
| Pipeline, 8 copies, drained every -b 8192 (8192 / (802 + 15 x 106) ms) | ~3,400 | 1.8x |
| Single host thread: measured ~250-270 ms of enqueue + input work per ubatch | ~1,900-2,050 | ~1.05x |
| Today's kernels with no overlap at all (512 / 802 ms) | ~640 | |

So 1000 t/s was a software-pipeline problem, not a hardware one; it is now passed by 1.35-2.1x. The binding limit at
8k-32k is the **single host thread** that walks 10 splits per ubatch (set_inputs, ~3-9 ms graph launch per stage, the
gpu9->gpu6 host-staged crossing, the server loop) - the measured wall per ubatch (273 ms at 32k) sits at the host cost
the pipeline lane measured, while the GPUs are busy only ~2.9 of 9 stage-slots. At 100k the indexer, the n_kv x n_ub
mask snapshot copies (256 MiB per ubatch at 128k) and the larger masks raise both host and GPU time (376 ms per ubatch).

## 5. Dead ends (do not redo)
- More than 8 pipeline copies (12, 12 + f16 mask): no gain (host-bound); copies 12 without the view cache OOMs at reserve
  and silently falls back to no pipelining (376 t/s).
- `LLAMA_SPEC_PROMPT_CHUNK` 4096 / 8192: worse than 2048. `-b 16384`: within noise, doubles the generating-slot stall.
- `-ub 1024` with 4 copies: no gain while host-bound (tolerance class, not gated).
- `LLAMA_SCHED_COPIES_SINGLE=1` (drafter copies): +1-2 % prefill, -4 % single-stream decode.
- Stage rebalance (`-ts`): 11 DSA layers over 9 stages - the prod split is already at the max-stage floor.
- MMQ: prefetch depth 2-3 (no gain / 6 % slower), depth 2 + unroll (2.6x slower, VGPR cap 128 -> AGPR shuffling),
  MFMA batching (= unroll), Q8_0 k01 unroll (slower), all dense Q8_0 to MMQ (rocBLAS faster for M >= 1536), thin-M MMQ
  (KL cost, restricted variants do not recover it - the DSA top-k cascade is chaotic, not additive).
- KDA at 3 waves/SIMD, exp(g) prefetch depth 8, register-window ssm_conv, 4 head groups per sparse-attention block
  (spills): all slower or no gain. f16/MFMA indexer: tolerance-class by design (the model keeps pooled keys in f32).
- Moving the n-4 checkpoint after the prompt; denser checkpoints; SJF ordering without the no-top-up rule.

## 6. Open issues and next levers (ranked)
1. **Host enqueue** is the limit now: one enqueue thread per stage (or per hive), cheaper set_inputs (build masks on
   the GPU from positions instead of n_kv x n_ub host masks + 256 MiB snapshot memcpy at 128k), fewer launches per
   stage, `GGML_CUDA_STAGE_RING=16` for the gpu9->gpu6 crossing (gpu6 still waits ~135 ms per ubatch in its inputs).
2. Keep the pipeline full across server batches (spec_flush drains at every n_batch; keep spec_pending while the slot
   is still prefilling, correct around update_dft checkpoints).
3. Kernels, which raise the host-free bounds and decide 100k: MoE MMQ is now VALU-epilogue bound (~2.5-3x above its HBM
   floor); rocBLAS + dequant/convert 19 % (a precise thin-projection kernel); KDA recurrence chunked form (tolerance);
   DSA non-contiguous copies (~0.8 ms per DSA layer); rms_norm 1024-thread tail.
4. Host-side state copies: `prompt_save` deep-copies every 145.6 MiB checkpoint (0.2-0.7 s per new request when the
   chosen slot is occupied; compaction moves 1.6-3.5 s for 8k states) - pinned / by-reference checkpoint transfer.
5. MTP drafter still costs ~8-10 % of prefill; `LLAMA_SCHED_COPIES_SINGLE` regresses decode - a variant that applies
   copies only to the drafter's prompt-sized ubatches is untested.
6. Non-blocking review notes left open: name-keyed staging table (two unnamed KDA state views collide -> one host sync
   per stage per ubatch), pinned staging memory never shrinks, f16 KQ mask lacks a long-context KL, no concurrent KL for
   SJF, `LLAMA_SLOT_CACHE_LCP` exact but not soak-tested under concurrency.

## 7. Reverting
- Whole build: `~/prod/glm53/bin.prev4` (7th build) + `run.sh.bak-<date>-7th` (see the promote command).
- Config only: `-b 2048` and unset `GGML_SCHED_COPIES` / `GGML_CUDA_STAGE_RING` / `LLAMA_SERVER_BUSY_PROMPT_CAP`
  returns the 7th-build pipeline behaviour on these bins (rotation still runs with 4 copies, bit-exact).
- Each switch in sec. 1 turns its path off independently. Caveat: `GGML_MMQ_MOE_TILED=0` falls back to stream-k with the
  new 64-row Q4_K tiles, which is not byte-identical to the 7th build; prod-identical MoE numerics need the 7th-build
  bins (or a rebuild with `-DGGML_MMQ_Q4K_I128=1` plus `GGML_MMQ_MOE_TILED=0`, not built or verified).
- Server switches (`LLAMA_CKPT_TAIL_ALIGN`, `LLAMA_PROMPT_SJF`, `LLAMA_PREFIX_SHARE`) default to upstream behaviour
  when unset.
