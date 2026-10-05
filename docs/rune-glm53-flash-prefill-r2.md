# GLM-5.3-Flash prefill, round 2 (2026-10-05): 8th build -> branch glm53/prefill2-final

Goal: ~3k tok/s prefill on the production configuration (10x MI100, 9 pipeline stages + MTP drafter, 6 x 128k).
Round 1 (docs/rune-glm53-flash-prefill.md) took prefill from ~550 to ~1.9-2.1k by making prompt ubatches overlap
across the stages; round 2 attacks what then bound it: the host thread, the staged GPU crossing, the drafter, kernels.

## Changes (each has an off switch; config = run.sh env)

| Commit | Change | Class | Off |
|---|---|---|---|
| 733a080af, 364e545c1 | `GGML_HOST_TRACE=<file>`: host timeline of the decode path (scheduler splits, waits, process_ubatch stages, server decode/draft calls, graph launch kind) | diagnostics | unset |
| 428a541a7 | staged island-crossing ring capacity 16 -> 64; prod sets `GGML_CUDA_STAGE_RING=16` | config | `GGML_CUDA_STAGE_RING=8` |
| 393ff819a | glm5next sparse gather builds its mask at the listed cells from cell/query positions; the n_kv x n_ubatch KQ mask (64 MB per prompt ubatch at the 32k pad, filled on the host and copied to every stage) is gone | bit-exact (KL 0.000000 @8k/32k) | `LLAMA_POS_MASK=0` |
| 9809d1082, 11a3ba52a | MTP draft catch-up deferred by up to d views (deep copies, `LLAMA_NEXTN_RING` export regions in their own fixed host buffer); pending views survive the end of a server batch; prod d=3, ring 4 | config (outputs unchanged: greedy text identical, acceptance identical) | `LLAMA_SPEC_DEFER=1` |
| 42e180943 | `LLAMA_CKPT_ASYNC=1` context checkpoints with queued state copies | experimental, OFF | unset |
| 3af246f5a | KDA recurrence: a block's 4 waves share each head's k/q/exp(g) through LDS (967 -> 547 us/call) | bit-exact | `GGML_GDN_LDS=0` |
| 9d47581fd | rms_norm loads batched (16384-wide 155 -> 129 us) | bit-exact | `-DGGML_CUDA_NORM_UNROLL=1` |
| 135edad31 | row-contiguous permute copy fast path (DSA absorbed q 472 -> 180 us) | bit-exact | `GGML_CUDA_CPY_ROWS=0` |
| ac546215b | MoE weighted reduction unrolled for 8 experts | bit-exact | revert |
| 7eee16c2e | thin Q8_0 projections (M <= 512, K <= 128, DSA k_b/v_b) on f16 MFMA with rocBLAS's f16 inputs | tolerance (KL 0.0105 @8k, 0.0076 @32k) | `GGML_CUDA_MM_THIN_F16=0` |
| 11a3ba52a | review fixes: nextn export ring in its own fixed buffer; NEXT_RESPONSE no longer drains the deferred views, they are flushed before checkpoints instead; draft failures abort slots; alignment guards in the row copy and KDA LDS paths | bit-exact for the target (greedy text and acceptance identical) | as above |

Not adopted: drafter scheduler copies (`LLAMA_SCHED_COPIES_SINGLE=1`: no prefill gain, multi-stream decode loss); a
worker thread for the draft catch-up (event waits from a second thread fail while the server thread captures a HIP
graph on that stream: "operation not permitted on an event last recorded in a capturing stream"; and no gain);
larger drafter views (`LLAMA_SPEC_PROMPT_CHUNK` 4096/8192: slower); async checkpoints (no wall-clock gain, worse with
MTP); MoE MMQ register-weights rewrite (bit-exact but 9-15% slower); f16 kernel for large rocBLAS shapes (slower);
f32-MFMA router (KL 0.0151, rejected).

## What bound prefill and how it moved (host trace, 32k prompt)
- 8th build: every ~8 ubatches the server thread blocked 0.3-0.7 s on the gpu9 -> gpu6 host-staged crossing: several
  tensors cross per ubatch, so 8 ring slots held ~2 ubatches and the host could run only ~2 ubatches ahead of gpu6.
  16 slots: NO_MTP 32k 2069 -> ~3.2k t/s.
- Then per ubatch: set_inputs ~22 ms + ~7 ms per stage of synchronous mask copies (64 MB each). Position mask: set_inputs
  ~5 ms, ~2.7 ms per stage.
- With the drafter: the server sends 2048-token views and ran the draft for view k right after issuing view k+1,
  waiting ~0.4-1.1 s for view k's h_nextn export each time (32k: 3.8 s in target decode calls, 8.8 s in draft catch-up).
  Deferral by 3 views removes the export waits inside a server batch.
- The first deferral build still drained the 9 stages at every 8192-token batch boundary: each `update_slots` posts a
  NEXT_RESPONSE task, and the "catch the draft up before any task" rule flushed the pending views on it, waiting for the
  batch's last export (the review found this). NEXT_RESPONSE touches no slot or draft state and no longer flushes; the
  export ring moved out of `buf_output` (re-laid-out by every decode's output reserve) into a fixed buffer, so a region
  stays valid until it is reused. Drafter catch-up on the server thread: 32k 3.3-4.0 -> 1.8 s, 100k ~11 -> 6.7 s.
  Pending views are instead flushed before each context checkpoint, where the target's state copy waits for the
  target anyway: the draft then catches up while the target finishes its last ubatches, not after the end-of-prompt
  checkpoint and the tail decode (8k prompts were 6-9% slower without this).
- HIP graph instances are not reused across requests on different slots (keys carry the slot's KV data pointers): each
  request runs ~11 eager + ~6 captured launches per stage before replays; measured not to be the 8k limit.

## Measurements (prod config, MTP on, same session, two interleaved rounds A/B; server t/s, median of 2)
| prompt | 8th build | round-2 gated build | |
|---|---|---|---|
| 2k | 1229 / 1181 | 1384 / 1457 | +18% |
| 8k | 2012 / 1996 | 2827 / 2646 | +38% |
| 32k | 1869 / 1786 | 2807 / 2828 | +53% |
| 64k | 1648 / 1651 | 2323 / 2339 (2nd runs 2737/2765) | +41% |
| 100k | 1413 / 1419 | 2213 / 2211 (2nd runs 2505/2499) | +56% |
| 2 x 100k concurrent | 1507 / 1487 agg | 2386 / 2402 agg | +60% |

Without the drafter the same build reaches ~3.0-3.3k at 32k. Decode (ab_workload): single 53.7/53.9 -> 52.1/55.0 t/s,
6 streams 40.3/38.0 -> 44.3/40.6 (noise). VRAM after the runs: fullest GPUs (gpu4/gpu9) 27.8 -> 25.7 GB (no mask
buffers). KL vs the 8th build: 8k 0.010544 (top-1 96.66%), 32k 0.007587 (96.99%); vs the 7th build (cumulative since
round 1): 8k 0.010277 (96.63%); floors 0.0167 @8k / 0.0196 @32k. test-backend-ops: GATED_DELTA_NET 38/38, RMS_NORM
51/51, CPY 249/249, CONCAT 185/185, SPARSE_ATTN 38/38, LIGHTNING_INDEXER 147/147, GET_ROWS 215/215, MUL_MAT 1314/1314.

Review fixes (11a3ba52a) vs the gated build, same session, prod config, MTP on; server t/s, rounds A / B:

| prompt | round-2 gated build | + review fixes | |
|---|---|---|---|
| 2k | 1420 / 1459 | 1560 / 1537 | +8% |
| 8k | 2863 / 2872 | 2940 / 2991 | +4% |
| 32k | 2821 / 2810 | 3228 / 3173 | +13% |
| 100k (repeat run) | 2474 / 2465 | 2884 / 2944 | +18% |
| 100k (first run at the pad) | 1600 / 1603 | 1795 / 1762 | +11% |
| 2 x 100k concurrent | 2447 / 2425 agg | 2953 / 2904 agg | +20% |

Greedy oracle (6.7k and 27k chat prompts, 160 tokens): identical text and acceptance (98/121 both). Decode: single
53.9/53.2 -> 54.0/54.3 t/s, 6 streams 38.8/45.6 -> 53.5/50.4 (6-stream runs are noisy; no decode path changed).
MTP-on 32k now matches the earlier NO_MTP figures (3.0-3.3k, not re-measured in this session). test-backend-ops CPY
249/249, GATED_DELTA_NET 38/38.

Known: the first long request on a fresh server is slower (graph capture / pool growth); the first request at each new
32k KV-pad step is ~25% slower than repeats (100k first run 1920 vs 2500).

## Next levers
- The drafter's catch-up still runs on the server thread (32k: ~1.8 s of 10 s); at 100k the first run at a new KV-pad
  step is ~40% slower than repeats (graph capture + pool growth per pad step).
- MoE MMQ (still ~43% of GPU time): 4-token-granularity tiling to cut padded MFMA/epilogue work (~29%).
- Graph reuse across slots (pointer-independent keys + exec update) for short prompts and first requests.
- `-ub 1024` (tolerance; needs KL budget: thin-f16 already uses ~63% of the 8k floor).
