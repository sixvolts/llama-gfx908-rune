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
