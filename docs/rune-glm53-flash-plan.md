# GLM-5.3-Flash on rune (10x MI100): feasibility, layout and physics estimates

Research note, 2026-09-23. Nothing here has been run on the hives yet; every number below is either read from the
model files / code or derived from measured Flash-Next behaviour on the same hardware.

## 1. What the model is (from the GGUF metadata and `config.json`)

| | GLM-5.3-Flash (`glm5next`) | Qwen3.8-Flash-Next (what runs today) |
|---|---|---|
| params | 321B total, ~18B active ("288x10B") | ~80B total, ~3.5B active |
| blocks | 46 = 45 trunk + 1 MTP layer (`blk.45`, in the main GGUF) | 48 + separate MTP head |
| layer mix | 34 KDA linear-attention + 11 DSA sparse-attention (every 4th: 3,7,...,43); MTP layer is DSA | 36 GDN + 12 full attention |
| hidden / streams | 4096, mHC hyper-connections hc=4 (Sinkhorn, 20 iters, 24-wide mixer) | 2560, HC hc=4 (sigmoid mix) |
| MoE | 288 experts, top-8 + 1 shared, 2048-wide, `noaux_tc` sigmoid routing, first 3 layers dense (12288) | 512 experts, top-10 + shared, ~640-wide |
| attention | MLA (q_lora 1536, kv_lora 512, 64 heads x 256, no rope dims) + lightning indexer (32 heads x 128, top-2048, kpool 4) | GQA + dsv4 indexer |
| KDA | 64 heads x 128, ssm_conv 4, low-rank f/g gates (4096->128->8192) | GDN |
| vocab / ctx | 154880 / 1M trained, 300k evaluated | 248320 / 262k |
| UD-Q4_K_XL | 200 GB: experts Q4_K (gate/up) + Q5_K (down), everything else Q8_0/F32, output Q8_0 (674 MB) | 46 GB |

Per-token weight bytes at UD-Q4_K_XL, from the tensor index (single token, decode):
KDA attention 5.0 GB (34 x ~148 MB: q/k/v/out are Q8_0 [4096<->8192]) + DSA attention 1.5 GB + routed experts 5.1 GB
(8 x 42 x 15.2 MB) + shared experts/routers 1.3 GB + dense FFNs 0.5 GB + lm_head 0.67 GB = **14.1 GB/token**
(Flash-Next: 6.5 GB measured). The KDA projections alone are as big as the routed experts.

## 2. Software status

- Support lives in **ggml-org/llama.cpp PR #27754** ("model: add GLM-5-Next (GLM-5.3-Flash)", danielhanchen, open,
  2026-08-26). Our production tree (`7baa212de`, branch qwen4exp) shares a merge-base with it (`95ef7fc16`): the PR
  is 245 commits ahead, we are 29 ahead. Dry-run merge: **4 conflicts** (`mmq.cu`, `mmq.cuh`, `mmid.cu`,
  `qwen4exp.cpp`), all in files we tuned - a rebase of our 29 commits onto the PR branch is the right move (as done for
  the Inkling 0.28 rebase). ggml-cuda.cu / common.cuh merge cleanly.
- What of our MI100 work carries over:
  - mHC: the PR uses the DSV4 fused ops `GGML_OP_DSV4_HC_PRE/COMB/POST` (`dsv4-hc.cu`, HIP-capable, enabled by default).
    Without them the Sinkhorn mixer would be ~7 launches x 20 iterations x 2 mixers x 46 layers = ~13k launches per token
    (~60 ms on MI100) - verify the fused path is taken on gfx908 on day one. Our `hc-fused.cu` (sigmoid HC of
    Flash-Next) does not apply.
  - KDA runs on `GGML_OP_GATED_DELTA_NET` with the `KDA` template flag already present in our `gated_delta_net.cu`; our
    NPRE preload (verify batches) and lane-per-column prefill kernels are GDN-only and need KDA ports.
  - MoE decode: our few-token dedup kernel covers Q4_K and Q5_1 experts; GLM's down experts are **Q5_K** -> a Q5_K
    variant is needed for the verify batch (or the upstream mmvq path is used). MMQ prefill gets the padding fix.
  - Lightning indexer: the CUDA kernel's WMMA path is compiled out on HIP; gfx908 runs the F32 `_vec` kernel (a few
    TFLOPS). Decode cost is small; at 100k+ prefill it is 10-30% of the work until an MFMA version exists.
  - DSA attention must run with **`-fa off`** (the FA F32->F16 cast breaks MLA precision, PR note): attention is
    mul_mat + soft_max over 2048(+3) gathered keys per DSA layer, so its cost is context-independent.
  - Multi-slot serving needs the non-unified KV cache (the PR refuses a unified cache with several sequences because
    key pooling would leak across sequences) - the server default.
- DFlash2 drafter (`incoai/GLM-5.3-Flash-DFlash2`): a 5-layer, hidden-4096 **Qwen3-backbone** (32/8 heads, head 128,
  sliding window 2048, ~1B params, BF16 safetensors), block size 8, target layers [5, 14, 24, 33, 42], two-tap conv
  (kernel 2, group 16), selector rank 256 / top-16. Our fork's `draft-dflash` already implements the conv + selector
  (the DFlash2 features) for Qwen3 backbones, and the PR's glm5next graph exposes the per-layer inputs the drafter
  reads (`t_layer_inp[il] = build_hc_mean(...)`). Conversion: `convert_hf_to_gguf.py incoai/GLM-5.3-Flash-DFlash2
  --target-model-dir zai-org/GLM-5.3-Flash`. Reported on other targets: 4.8-6.0 accepted tokens per step, 2.7-3.4x on
  dense targets; no GLM-5.3-Flash numbers published yet.

## 3. Layout on rune

The model does not fit one hive (4 x 30.5 GB usable = 122 GB). Options:

| layout | usable VRAM | weights | headroom | notes |
|---|---|---|---|---|
| 8-stage pipeline A(0,2,3,4) -> B(5,7,8,9) | 244 GB | Q4_K_XL 200 GB | ~44 GB: ~3 GB/GPU compute buffers + KV | tight but feasible; orphans 1/6 free for the drafter |
| 10-stage (orphans 6 and 1 as extra stages) | 305 GB | Q4_K_XL 200 / Q5_K_XL 240 | ~105 / ~65 GB | each orphan adds 2 host-staged PCIe hops per step; drafter shares an orphan |

KV/state budget: MLA compressed KV + indexer keys ~15 KB/token across the 11 DSA layers (4 x 128k slots = 7.6 GB;
300k x 1 = 4.4 GB), KDA recurrent state ~143 MB per sequence (34 layers x 64 x 128 x 128 f32). Comfortably inside
either layout; the compute buffers for 288-expert MoE at `-ub 512` are the thing to measure.

**Cross-hive rule.** The hives are separate XGMI islands; cross-hive peer copies collapsed hive A's fabric under load
(uncorrectable ATHUB/XGMI errors, GPU reset). The fork's `ggml_cuda_can_access_peer` trusts `cudaDeviceCanAccessPeer`,
which reports every pair as accessible on this driver, so an 8-stage split would issue `hipMemcpyPeerAsync` across the
hives. Before any two-hive run: a peer allowlist (env, e.g. `GGML_CUDA_PEER_GROUPS=0,2,3,4;5,7,8,9`) that keeps XGMI
peer copies inside a hive and routes everything else through the existing pinned host-staged path (`c1fb8c054`), and
the same gate on `cudaDeviceEnablePeerAccess` at load. Cost of the crossing: the mHC residual is 4 x 4096 floats per
token (64 KB), so ~0.1-0.2 ms per decode step and ~1% of a 512-token prefill ubatch.

## 4. Physics estimates (single 8-stage pipeline, UD-Q4_K_XL)

Assumptions: ~0.9 TB/s achievable HBM2 per MI100 (we measure 0.88 on the big Q8_0 GEMVs), pipeline stages run one
after another for a single stream, ~4.5 us per kernel launch even inside HIP graphs, ~3000 launches per token
(Flash-Next: 3128 for 48 layers). "Efficiency" is the fraction of the bandwidth floor the whole step reaches:
Flash-Next reaches 26-31% because its kernels are tiny (640-wide experts, thin projections); GLM's GEMVs are 5-10x
larger, so 40-60% is the working range.

Decode, one stream, no speculation: floor 14.1 GB / 0.9 TB/s = **15.6 ms -> 64 t/s**; realistic **26-38 t/s**
(40-60%), most likely ~32 t/s once the HIP kernels are tuned, 15-25 t/s on first bring-up. Consistency check: the PR
reports 63 t/s on a B200 (8 TB/s) at IQ1_S - i.e. launch-bound at ~16 ms/token, which matches ~3000 launches at ~2 us
per launch on a B200 graph and ~14 ms of launch floor on MI100 that sits under our bandwidth time.

**Speculative decoding is bandwidth-capped by the expert union.** Every extra verify row routes to 8 more experts out
of 288 (nearly disjoint), and each expert is 15 MB; the non-expert bytes (9 GB) stay fixed:

| verify rows | distinct experts | step bytes | floor | at 50% + draft | accepted/step | t/s |
|---|---|---|---|---|---|---|
| 1 (no spec) | 8 | 14.1 GB | 15.6 ms | 31 ms | 1 | 32 |
| 3 (MTP n=2, p=0.75) | 23 | 23.9 GB | 26.5 ms | 56 ms | 2.3 | **41** |
| 5 (DFlash2 block 4, p=0.85) | 38 | 33.1 GB | 36.8 ms | 77 ms | 3.7 | **48** |
| 9 (DFlash2 block 8, p=0.80) | 65 | 50.2 GB | 55.7 ms | 115 ms | 4.3 | 38 |

So on MI100 the DFlash2 block should be truncated to 4-5 (the fork's `--spec-draft-n-max` and the confidence cut do
this), giving ~+50% over raw versus MTP's ~+28%; block 8 is a net loss here even though it wins on B200/H100 class
bandwidth. Flash-Next's verify rows are cheap (+7.7%/row) precisely because its experts are 5x smaller - that
economics does not transfer. The drafter itself (5 layers, ~1.4 GB at Q8 + block lm_head) is ~3 ms/step on an idle
orphan; its inputs are 5 x 4096 floats per row, host-staged.

Concurrency (agents), 50% efficiency: 4 streams ~63 t/s aggregate (16/stream), 8 streams ~78 t/s (10/stream); with
per-stream MTP 66 / 83 t/s. The union saturates (8 streams x 3 rows already touch half the experts), so aggregate
throughput flattens around 80-90 t/s.

Prefill: 33.6 GFLOP/token (2 x 16.8B). Flash-Next's server prefill is 3.5 TFLOP/s effective per GPU (MoE GEMMs see
~14 rows per expert per 512-token ubatch; the KDA/indexer/HC kernels are launch- and latency-bound). 8 GPUs x 3.5 =
**~830 t/s** mid-length, ~1200 t/s if a larger ubatch lifts MMQ to 5 TFLOP/s per GPU (VRAM permitting), 10 stages
~1050. Expect 300-600 t/s on first bring-up. The DSA attention keeps attention cost flat with context; the indexer's
F32 kernel and the 300k-token pooled key scans are what degrade long prompts until an MFMA indexer exists.

Bottom line versus today's Flash-Next service (73 t/s single-stream with MTP, ~2000 t/s prefill): roughly **0.45x
the decode speed and 0.4-0.5x the prefill**, at a model that is 4-5x larger in active compute - the MI100s are
bandwidth machines and this model reads 2.2x the bytes per token.

## 5. Plan (in order, with the gates)

1. Rebase our 29 commits onto `pr-27754` (4 conflicts), build for gfx908, run the Flash-Next oracle suite on Hive A to
   prove nothing regressed there (half a day).
2. Peer-access allowlist + host-staged cross-hive copies; prove it under sustained load with `dmesg` armed (the
   2026-07-29 lesson: one successful transfer proves nothing).
3. Download UD-Q4_K_XL (200 GB, 608 GB free on the NVMe) and the DFlash2 safetensors; convert the drafter with
   `--target-model-dir`.
4. Bring-up on the 8-stage layout with `-fa off`, PPL/KL against the BF16 reference logits, seeded-hash oracles,
   measure decode/prefill; check the fused HC ops fire and the launch count per token.
5. HIP tuning in the order the profile dictates - expected: Q5_K few-token expert kernel, KDA ports of the GDN preload /
   lane-per-column kernels, MFMA lightning indexer, MMQ tile configs for 4096/8192/16384 shapes, HC fused-op checks.
6. DFlash2 on orphan GPU 1 or 6; block sweep 4/5/6/8 with the confidence cut; compare with the built-in MTP layer.
7. Decide the production layout (8 vs 10 stages, Q4 vs Q5, slots x context) from the measured VRAM and the agents'
   real traffic; Hive B's Flash-Next service cannot coexist with it (both hives are needed).

## 6. Prep status (2026-09-24, no GPU used - both hives stayed in production)

Branches: `flash-next/13th-build` = the exact production commit (7baa212de), `glm53/snapshot-base` = the tuning tip
before this work, `glm53/bringup` = worktree `/home/sixvolts/llama.cpp-glm53`: PR #27754 (86ebfef2c) + our 30 commits
replayed + the peer policy. Build `/home/sixvolts/llama.cpp-glm53/build` (same CMake options as production).

- **Rebase.** Conflicts were all where upstream had moved first: qwen4exp tensor shapes (hc norms now `[n_embd, hc]`
  with `TENSOR_ALLOW_RESHAPE`; kept upstream's shapes with our MTP skip/trunk flags) and MMQ, where upstream added the
  same per-expert J hint as ours under the name `ncols_opt` (kept upstream's, our `GGML_MMQ_MOE_J_MULT` override on
  top). Everything compiles. Upstream also changed qwen4exp numerics since our fork point (GDN norm `max` -> `rsqrt`
  #28068, rms_norm+mul fusion #28896, fp32 accumulation in the HIP MFMA flash attention #28576), so Flash-Next on
  this branch is NOT expected to hash-match the 13th build: the regression gate there is KL/PPL + speed, and our HC
  fusion matchers must be re-checked against the new node order (a miss costs speed, not correctness).
- **Peer policy** (`f28ce566e`): on HIP a device pair uses peer copies only with a direct XGMI link (sysfs confirms
  the islands {0,2,3,4} and {5,7,8,9}; GPUs 1 and 6 are PCIe-only). All cross-island traffic takes the pinned
  host-staged path. `GGML_CUDA_PEER_POLICY=query|xgmi|none`. Untested on hardware - the soak script is the gate.
- **Layout correction.** On 8 GPUs the UD-Q4_K_XL weights (194.4 GB without the MTP layer) do not fit with a 3 GB
  compute reserve: MoE layers are ~4.6 GB, and the best contiguous split leaves the fullest GPUs 0.2-1.4 GB short.
  9 stages (Hive A, Hive B, then GPU 6) fit with >= 2.9 GB free per GPU at 4 x 128k (`-ts 7,4,4,5,5,5,5,5,5`), leaving
  GPU 1 for the drafter. 8 GPUs are possible only with per-tensor `-ot` balancing (~2 GB free) - not worth it.
- **DFlash2 drafter**: converted with `--target-model-dir` (tokenizer from the target, metadata verified: target
  layers [6,15,25,34,43], block 8, conv 2/16, selector 256/16, SWA 2048). The checkpoint has no embeddings and no
  lm_head - it borrows the target's at runtime, which from GPU 1 would mean reading the target's 674 MB lm_head across
  devices every draft step - so the target's Q8_0 `token_embd`/`output` were appended
  (`scripts/rune/gguf_add_tensors.py`) and the whole draft quantized to Q8_0: `GLM-5.3-Flash-DFlash2-Q8_0-sc.gguf`,
  2.5 GB, self-contained. SGLang's reference captures the mean of the 4 mHC streams at the input of layer k+1, which
  is exactly what the PR's graph exports for `target_layers` - acceptance should match SGLang's.
  License: CC BY-NC-ND 4.0 (private evaluation use only; do not redistribute the converted files).
- **Q6_K mix**: `llama-quantize` re-derives the type of every tensor an override does not match, so the requant uses
  an explicit map of all 1412 tensors (`glm53_plan.py typemap --q6k`): 294 Q8_0 projections -> Q6_K, the other 1118
  copied byte-for-byte (unsloth's Q4_K/Q5_K/Q6_K experts untouched), unsloth imatrix, -1.87 GB.
  Output: `UD-Q4_K_XL-Q6mix/` (5.6 min on CPU; verified: 1118 tensors byte-identical to the source, exactly the 294
  planned ones retyped; decode bytes 14.12 -> 12.25 GB/token). Gate before use: KL vs UD-Q4_K_XL, and the Q6_K GEMV must reach Q8_0's bandwidth
  efficiency on MI100 (mmvq Q6_K is upstream code we have not tuned) or the byte saving does not turn into speed.
- HIP coverage (static): lightning indexer runs its vec kernel for the 32-head config (F32/F16/Q8_0 keys), fused mHC
  ops take F32, KDA runs on our GDN kernels (their preload and lane-per-column variants already carry the KDA template
  branch), glm5next supports recurrent rollback for speculative decoding (`n_rs_seq` = draft depth).

Scripts (`scripts/rune/`): `glm53-env.sh` (layout + flags), `glm53-serve.sh [spec|nospec] [q4xl|q6mix]`,
`glm53-peer-soak.sh [minutes]` (sustained cross-island load with the kernel log watched; stops the server on any
fabric error), `glm53_plan.py bytes|layout|typemap`, `gguf_add_tensors.py`. Both serve scripts refuse to start while
`flash-next` or `flash-next-b` is active.

Bring-up order once both hives can be released: soak (20 min) -> nospec bring-up (load, VRAM per GPU, PPL/KL vs the
reference, tokens/s, HIP-graph and fused-op checks) -> DFlash2 depth sweep 3/4/5/6 -> Q6mix gate -> kernel work from
the profile.

## 7. First measurements on the two hives (2026-09-24, Qwen services paused)

Layout: 9 stages (A 0,2,3,4 -> B 5,7,8,9 -> GPU 6), DFlash2 Q8_0 drafter on GPU 1, `-fa off`, 1 slot x 32k, ubatch 512.
Safety: 3-min and 20-min cross-island soaks (the 20-min one with the drafter: 75 requests, 236k tokens) passed with no
fabric/hardware errors; staged pairs exactly the cross-island ones (3->4, 7->8, 0->4..8); hops cost 20-40 us.

- Compute buffers are the memory constraint, not weights: the DSA "sparse" attention is mask-based dense attention,
  so with `-fa off` every stage materializes [n_kv x ubatch x 64 heads] scores (~288 B per pair: 128k x 512 ->
  19.7 GB per GPU). With FA the buffers drop to 2.4 GB, but on gfx908 the 512-wide MLA runs the tile kernel with
  packed-fp16 Q.K (the precision problem the PR avoids). Hence 32k x 1 for now.
- `-ts` must count layers the way llama.cpp assigns them (il/47 against the cumulative split, the last share holding
  the skipped MTP layer and the output): `-ts 7,4,4,5,5,5,5,5,7` reproduces the planner's split.
- Graph reuse was never happening (`llm_graph_input_kpool` lacked `can_reuse`): a fixed 10.9 ms host rebuild per
  token. Fixed (b29ce33bc, outputs identical): raw decode 19.7 -> 26.0 t/s.
- Decode anatomy (trace, 1 token): 9 stages strictly sequential, ~3,700 kernels/token. Dense Q8_0 GEMVs are already at
  0.76-0.88 TB/s on GLM's shapes (upstream mmvq; our tuned kernel adds nothing here), MoE experts ~72% of bandwidth;
  the remainder is launch-bound small kernels (quantize, mHC pre/comb/post, norms, elementwise, 4.5 us each) plus
  the mHC Sinkhorn kernel (19 us x 90), a 46 us recurrent-state get_rows x 34 and the indexer (87 us x 11).
- DFlash2 depth sweep (greedy, 3 prompts): raw 26.0 | d2 34.2/42.1/35.8 | d3 33.3/40.8/33.3 | d4 31.4/38.2/28.7 |
  d5 29.8/36.2/27.2 | d6 27.8/32.4/25.5. Depth 2 wins (+44%), as the expert-union model predicts. Acceptance ~40-60%
  per draft at d4.
- With the drafter (d2): prompt 1.6k/6.5k/13k/22.7k -> prefill 227/389/388/318 t/s, decode 32.5/24.7/20.7/13.9 t/s.
  Without the drafter prefill was ~640 t/s at 6k (the drafter adds the 5-layer hidden-state export and its own
  prompt pass). Decode falls with context because attention and the indexer scan the whole cache (dense masked
  attention): the case for true gather-based sparse attention.

## 8. Reuse from llama-halo-hybrid, steps 1-3 (2026-09-24)

Port order from the review of github.com/sixvolts/llama-halo-hybrid (Strix Halo GLM tuning): (1) ssm_a / TOP_K ties /
DFlash causal fix, (2) scheduler and input-copy host paths, (3) MTP export + lossless rejection sampling, (4) sparse
attention with a gfx908 kernel, (5) the fusion wave.

- Step 1 (e4aa7799f): `ssm_a` loads from the NOSCAN tensor name, and TOP_K resolves ties in column order
  (test-backend-ops TOP_K 525/525 on gfx908). Not ported: Halo's DFlash causal-SWA override applies only when a config
  leaves `is_causal` unset, and the GLM DFlash2 config sets `is_causal=false`, so the bidirectional block is correct.
- Step 2: skipped on evidence. After graph reuse, decode is GPU-bound (stage busy time 40.7 of a 42.1 ms period) with
  about 9 copies per token, so the host paths Halo tuned have nothing left to give here.
- Step 3: the model's own MTP (NextN) block as the drafter, with lossless rejection sampling.
  - `glm53_export_mtp.py` writes blk.45 plus the global tensors as a self-contained 5.9 GB draft file that runs on
    GPU 1. With a separate `-md` file the target no longer loads its own blk.45: GPU 6 holds the same VRAM as without
    speculation.
  - Rejection sampling (Halo 2c85bcfa6, reworked here): the MTP head samples its draft from its own top-10
    distribution at the request's temperature, with top-p/min-p mirrored, and the target accepts a draft token with
    probability min(1, p/q), else it resamples from max(0, p - q). Streams are seeded from the request's resolved seed
    through splitmix (raw small seeds biased the first tests toward accepting). RS turns itself off, falling back to
    the compare verifier, for mirostat, adaptive-p, an active grammar, backend sampling and synthetic drafts. After a
    checkpoint restore, the replay accepts the tokens RS already emitted as they are. `LLAMA_SPEC_RS=0` disables it.
  - Lossless check (`glm53_rs_dist.py`, 400 seeds per condition, token distribution at the first verified noun):
    RS against compare mode gives p = 0.25 (full distribution) and p = 0.84 (top-p 0.95, min-p 0.05). Against a
    nospec server the truncated case fails (p < 0.001), and that is not RS: the verify batch's logits differ slightly
    from single-token decode, and two tokens at p 0.036/0.034 sit on the min-p cut (0.034). Every verify keeps them
    and every plain decode drops them. Compare mode shows the same, so it is the reference for this test.

Sampled decode (T = 1.0, top-p 0.95, 4 prompts x 2 seeds, 2.5k tokens, `glm53_sampled.py`):

| drafter | verifier | decode t/s | draft acceptance |
|---|---|---|---|
| none | - | 26.1 | - |
| DFlash2, depth 2 | compare | 35.9 | 0.62 |
| DFlash2, depth 3 | compare | 34.2 | 0.50 |
| MTP, depth 2 | compare | 37.0 | 0.66 |
| MTP, depth 2 | RS | 38.9 | 0.71 |
| MTP, depth 3 | RS | 37.1 | 0.59 |

Greedy (3 prompts, t/s per prompt): MTP depth 2 36.6/41.8/38.4 against DFlash2 depth 2 34.2/42.1/35.8, with identical
output hashes (both verify the same batch shape). MTP depth 2 with RS is the best drafter configuration: +49% over no
speculation on sampled chat, +8% over DFlash2, and its drafter needs no second model. `GLM_SPEC=mtp` selects it.

## 9. Step 4: sparse DSA attention on gfx908 (2026-09-24)

The PR computes DSA attention as dense attention under a mask. Every query scores all n_kv cells, and the host fills
two n_kv-wide masks per ubatch. With `-fa off` that materializes [n_kv x ubatch x 64] scores, so the compute buffer
limited us to 1 x 32k, and decode fell from 24 to 11.5 t/s between 1.6k and 22.7k of context. A query only ever
attends the cells of its top-k pools plus its own partial pool, at most top_k + kpool - 1 = 2051 cells. The sparse
path now computes exactly that set.

- Inputs (`LLAMA_DSA_SPARSE` set): the pooled-indexer input gives each query its tail, the visible cells of its own
  partial pool (at most kpool - 1), as a cell list and a 0/-inf mask. They replace the n_kv-wide `sel_mask` and
  `cand_mask`. A selected pool is usable iff its `pool_bias` is 0, so its members' mask is that bias gathered at the
  top-k picks. The KQ mask is also gathered at every listed cell, as in the dense path.
- `GGML_OP_SPARSE_ATTN` (`ggml_sparse_attn`): attention of each query over its own cell list, one KV head, V = a prefix
  of the K row. CPU reference in ggml-cpu. The gfx908 kernel (`ggml-cuda/sparse-attn.cu`) handles one query x 16 heads
  x 64-cell tiles per block. It gathers the K rows straight from the cache into registers and runs QK on
  `v_mfma_f32_16x16x16f16`, with Q held in registers as f16. The online softmax runs across the 4 waves, and PV runs on
  the same MFMA with V^T staged through LDS in 128-wide chunks. Precision matches the dense `-fa off` path: f16 K, Q
  and P with f32 accumulation. For few queries (decode, verify), the cell range is split across blocks and a combine
  kernel merges the partial results.
- `LLAMA_DSA_SPARSE=1` selects the fused op, `=2` a reference built from get_rows/mul_mat/soft_max, and unset or 0
  keeps the dense path.

Validation:
- test-backend-ops SPARSE_ATTN: 18/18 against the CPU reference (decode, verify and 64-query batches, 1 and 2 streams,
  partial tiles, a fully masked query).
- The dense path is unchanged: greedy hashes are identical to before (7f3bc855 3ea253ae 3667a96e).
- KL gate (`glm53-kl.sh`, wikitext-2, 6 chunks of 8k, so the top-k is active). The model's numerics floor is not 0:
  changing only the dense path's ubatch moves the logits as much as the sparse paths do, because a tiny difference flips
  near-tied top-k pool picks in later DSA layers.

| against dense, ubatch 512 | mean KLD | same top token | PPL |
|---|---|---|---|
| dense again | 0.000000 | 99.996% | 2.6852 |
| dense, ubatch 256 (numerics only) | 0.00976 | 96.46% | 2.6851 |
| sparse, reference ops | 0.00945 | 96.73% | 2.6832 |
| sparse, fused kernel | 0.00965 | 96.70% | 2.6814 |

Both sparse paths sit at the numerics floor, so later tolerance-class GLM changes should be gated against the ubatch-256
figure, not against 0.

Speed without speculation, one slot (prefill / decode t/s):

| prompt tokens | dense, 32k ctx | fused, 32k ctx | fused, 128k ctx |
|---|---|---|---|
| 1,641 | 280 / 24.1 | 309 / 25.6 | |
| 6,471 | 462 / 19.4 | 518 / 25.0 | |
| 13,001 | 423 / 15.2 | 561 / 24.6 | |
| 22,726 | 342 / 11.5 | 574 / 24.2 | 435 / 24.3 |
| 54,054 | | | 497 / 23.7 |
| 108,043 | | | 449 / 22.2 |

Decode no longer falls with the cache (22.2 t/s at 108k), and long prefill is up to 1.7x faster. VRAM at 32k drops by
about 4.5 GB per GPU. At 128k the fullest GPU holds 27.1 of 34.3 GB. The dense path cannot run 128k with ubatch 512.

With the MTP drafter (depth 2, lossless RS) the sparse path applies to the drafter's DSA layer too:
- Sampled chat (T = 1.0, top-p 0.95): 40.9 t/s, acceptance 0.74, against 38.9 with dense attention.
- Greedy, 3 prompts: 37.6/41.7/41.5 t/s.
- 128k context, prefill / decode t/s: 1.6k prompt 289 / 36.8, 22.7k 682 / 37.9, 108k 352 / 30.9.

Two prefill effects are open. The first prompt that reaches a new cache width prefills slower than a repeat of the
same length (22.7k: 422 against 682 t/s in one server run), a one-time warm-up of per-shape state, not steady state.
And the drafter's own prompt pass costs about a fifth of prefill at 108k (352 against 449 t/s without it).
`LLAMA_DSA_SPARSE=1` is now the default in `glm53-env.sh`.

## 10. Production readiness (2026-09-29/30, GLM replaces both Qwen stacks)

- **Correctness:** perplexity depended on ubatch size (4k ctx: ubatch 512 2.04, ubatch 8 2.21-2.22, nondeterministic).
  Root cause was ours: the Flash-Next host-path tuning copied scheduler inputs with an async H2D from pinned memory
  and skipped the pre-set_inputs sync on graph reuse, so the next ubatch's tokens/positions/masks raced into the
  current one. Fixed (9052d1413): ubatch 8 2.0306 against ubatch 512 2.0324. Residual run-to-run KL at 4k (indexer
  active) ~0.0005, 20x below the numerics floor.
- **Slots:** 6 x 128k fits the 9-stage layout with the MTP drafter on GPU 1; peak VRAM under 6 concurrent 22k-token
  requests 30.1 of 34.3 GB on the fullest cards, GPU 1 at 8.2 GB. No layers need to move onto the drafter card.
- **Adaptive speculation** (912d872ee, LLAMA_SPEC_MAX_GEN=2): MTP drafts only while <= 2 slots generate.
- **Decode micro-batching** (splitting a concurrent decode step into 2-6 ubatches for pipeline overlap): tried and
  dropped. 6 streams +7%, 3-4 streams -2% to -38%: the stages do not overlap on rune, so each micro-batch only
  re-reads the weights. Making pipeline overlap work is the open lever for concurrent throughput.
- **Fusion wave** (9db82ab54, 5beb30e68): single-stream decode without speculation 27.5 -> 32.2 t/s.

Production-style configuration (6 x 128k, MTP depth 2 + adaptive speculation, sparse DSA, fusion wave), aggregate
decode, short prompts:

| concurrent streams | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| t/s per stream | 43.9 | 25.7 | 19.4 | 15.9 | 11.5 |
| aggregate t/s | 43.9 | 51.5 | 58.2 | 63.6 | 69.0 |

## 11. Prefill: the stages never overlapped (2026-09-30)

Production prefill was flat at ~390 t/s from 1.6k to 22.7k tokens and fell to 313 at 108k, with `-b 512`, `-b 2048`
and `-b 8192` all identical. A rocprofv3 trace of one warm 3k-token prompt showed exactly one GPU busy 94.7% of the
time: stage 0 started ubatch u+1 only after stage 8 finished ubatch u. Stack samples put the host in
`ggml_backend_sched_alloc_graph -> ggml_backend_cuda_synchronize` in 9 of 10 samples: every ubatch re-reserved the
graph, and a re-reserve drains every backend first (the split inputs may move).

Cause: `n_kv` is the used cells padded to 256, so it grows by one ubatch on every prompt ubatch, and the pooled DSA
indexer's inputs and intermediates (`kpool_pool_reps`, `pool_bias`, scores) are all sized by it. Each ubatch outgrew
the previous reservation (`GGML_GALLOC_DEBUG=1`: kpool_pool_reps 322 -> 450 -> 578 ...). A decode step resets the
reservation, so repeating a prompt does not help; the old "first prompt at a new width is slower" note was this.

Fix: `LLAMA_KV_PAD_PREFILL=n` pads prompt-sized ubatches (> 32 tokens per stream) to a multiple of n cells, so a
prompt changes shape once per n cells. Decode keeps the 256 padding. Empty cells cannot leak in: every pool starts at
-inf bias and only occupied cells of the right sequence are unmasked.

| prompt tokens (MTP on, 6 x 128k) | before | n = 32768 |
|---|---|---|
| 1,639 (warm) | 372 | 415 |
| 6,471 | 395 | 580 |
| 22,722 | 390 | 633 |
| 54,054 | 358 | 582 |
| 108,044 | 313 | 517 |

Without the drafter 22.7k reaches 720 t/s (415 before). n = 16384 is better at 1.6k (449) and worse at 22.7k (604);
geometric buckets (pad next_pow2/4, /8) reach only 560 / 449 at 22.7k because each re-reserve costs a full drain.
KL gate (4k ctx, 2 chunks, against an unpadded base): padded 0.0103, the ubatch-256 numerics-only reference 0.0120,
unpadded again 0.0015; PPL 2.0388 unpadded / 2.0390 padded. The shift is rocBLAS tiling for a different pool count
flipping near-tied top-k picks, the same class as a ubatch change.

The same drain probably explains the failed decode micro-batching in section 10 (every micro-batch shape change
re-reserves); worth re-testing with stable shapes.

### Busy slots must be contiguous (9ab7ea56d)

Checking concurrency after the prefill fix exposed an older problem: after a few requests, 3 streams decoded at
33 t/s aggregate instead of 58.5 (4 streams 39 vs 64), with no graph reuse. Without a unified KV cache,
`split_equal(sequential)` only puts consecutive sequence ids in one ubatch (the K/V views span one contiguous stream
range), so busy slots {0,4,5} made every decode step two ubatches, two passes over the weights. LRU and LCP-similarity
slot choice produce such gaps routinely. The server now gives a new task the idle slot that leaves the fewest runs of
busy slots, and moves a fragmenting LCP match into such a slot through the host prompt cache (~0.3 s for 650 tokens).
`LLAMA_SLOT_COMPACT=0` reverts. After the same long-prompt history: 1/2/3/4/6 streams 41.6/50.6/58.2/63.0/69.0 t/s.

Moves are verified after the load (the new slot must hold at least the source's prefix, else the source slot is used),
skipped when the prompt cache would refuse the entry (f_keep < 0.25) and bounded by `LLAMA_SLOT_MOVE_MAX` (65536
tokens): 320 / 411 / 612 ms for 605 / 6k / 21k tokens, ~300 ms of it the fixed recurrent state (a1b976449).

### Run-to-run variance: the radix top-k's order (904105df8)

Full prefills of the same >2k-token prompt were not deterministic: greedy text differed from the first sentence,
first-token logprobs by up to 0.85, and two identical perplexity runs by KL 0.0094 with the prefill padding (0.0015
without). The radix top-k placed the entries above the k-th key with an atomic counter, so the selected set was
stable but its order was not, and the sparse attention / indexer accumulate in list order. With the padding, early
ubatches have more pools than k (most at -inf), so a real top-k ran where the unpadded path selected everything.
A count pass plus per-block ballot compaction makes the order deterministic: identical runs now give mean KLD 0 and
100% same top token (padded, unpadded, HC mix on). rocBLAS atomics (`ROCBLAS_DEFAULT_ATOMICS_MODE=0`) made no
difference. Most of the earlier "padding KL 0.0103" was this noise; measured deterministically it is still 0.0103
against the unpadded path (PPL 2.0359 vs 2.0351), below the ubatch-256 reference 0.0124 (PPL 2.0414).

## 12. MTP verify kernels: 45.9 -> 50.5 t/s single stream (2026-09-30)

Step anatomy (LLAMA_SPEC_TIMING, sampled 8 x 320 tok): 52.7 ms = draft 4.2 + target verify 46.6 + ~2 host, 2.46 tokens
per step. Stage handoffs inside a verify are only 1.2 ms; the verify is ~47 ms of kernels: MoE 18.6, dense Q8_0 mat-vec
(3 columns) 15.5, ~1800 small kernels ~13. Depth 3 (41.8 t/s) and confidence-gated drafting (37.9 / 43.2) lose because
variable verify widths defeat graph reuse; depth 2 fixed stays.

- Dense (26a96ef42, bit-exact): the Qwen-era mmvq_q8_0_v2 only covered K = 640/2560/6144; GLM's K (hidden 4096) now
  dispatch there. mul_mat_vec_q at 3 columns loses 25% vs 1 column (Q8_0's 34-byte blocks); v2 is 12-27% faster at
  3 columns for K <= 8192 (one column only for 12288/16384). Dense 15.5 -> 13.8 ms/step, 46.0 -> 46.8 t/s.
- MoE (1d203ffd5, tolerance): the verify ran one warp per (token, slot), re-reading shared experts; distinct experts are
  77.7% of pairs. The dedup kernel now takes Q4_K gate/up at K = 4096 with the SWIGLU_CLAMP epilogue and Q5_K down
  (qh unpacked to 5-bit bytes once): 46.8 -> 49.1 -> 50.5 t/s. KL at ubatch 3 vs GGML_MOE_V2=0: 0.0180 (PPL 2.0289 vs
  2.0429), numerics-only reference ubatch 4 vs 3: 0.0318 (PPL 2.0597).

| streams (aggregate t/s) | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| 4th build | 47.4 | ~51 | 58.8 | 63.0 | 68.7 |
| 5th build | 51.8 | 55.3 | 63.6 | 71.4 | 68.8 |

Six streams (6-column MoE) still use the old kernel: the dedup kernel stops at 4 tokens. Next levers: extend it to 8,
drafter catch-up merged into the first draft (~1.4 ms/step), the ~1800 small kernels.
