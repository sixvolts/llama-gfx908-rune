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
