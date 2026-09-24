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
