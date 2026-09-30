# GLM-5.3-Flash on rune: shared launch settings (sourced by glm53-serve.sh / glm53-peer-soak.sh).
# Needs BOTH hives: the flash-next and flash-next-b services must be stopped first (the scripts refuse otherwise).
#
# Layout (scripts/rune/glm53_plan.py layout, UD-Q4_K_XL, 4 x 128k, MTP layer not loaded): 9 pipeline stages
#   Hive A 0,2,3,4 (XGMI) -> Hive B 5,7,8,9 (XGMI) -> GPU 6 (PCIe), DFlash2 drafter on GPU 1 (PCIe).
#   Island crossings (A->B, B->6) and the drafter's inputs go through the host-staged copy path: the peer policy
#   (GGML_CUDA_PEER_POLICY, default xgmi) never issues peer copies outside an XGMI island.
#   -ts counts layers the way llama.cpp assigns them: layer il (0..45) and the output layer (46) go to device
#   upper_bound(cumsum(ts)/sum(ts), il/47), so the last device's share includes the (skipped) MTP layer and the output.
# DSA attention: LLAMA_DSA_SPARSE=1 (default here) attends only each query's selected cells (<= 2051) with the fused
#   GGML_OP_SPARSE_ATTN kernel, so neither its cost nor its compute buffer grows with the context (1 x 128k fits, the
#   fullest GPU at 27.1 of 34.3 GB). LLAMA_DSA_SPARSE=0 restores the PR's mask-based dense attention, which
#   materializes [n_kv x n_ubatch x 64] scores (~288 B x ctx_per_slot x ubatch per GPU: 32k x 512 -> 5.1 GB).
GLM_BIN=${GLM_BIN:-/home/sixvolts/llama.cpp-glm53/build/bin}
GLM_MODEL=${GLM_MODEL:-/home/sixvolts/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
GLM_MODEL_Q6MIX=/home/sixvolts/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL-Q6mix/GLM-5.3-Flash-UD-Q4_K_XL-Q6mix-00001-of-00006.gguf
GLM_DRAFT=${GLM_DRAFT:-/home/sixvolts/models/GLM-5.3-Flash-DFlash2/GLM-5.3-Flash-DFlash2-Q8_0-sc.gguf}
GLM_PORT=${GLM_PORT:-18099}
GLM_DRAFT_N=${GLM_DRAFT_N:-2}          # measured: 2 > 3 > 4 > 5 > 6 (each verify row adds ~8 of 288 experts' weights)

export HIP_VISIBLE_DEVICES=0,2,3,4,5,7,8,9,6,1   # ROCm0..8 = trunk stages, ROCm9 = drafter
export LLAMA_DSA_SPARSE=${LLAMA_DSA_SPARSE:-1}
# adaptive speculation: draft only while <= 2 slots generate (6 slots, aggregate t/s, no drafter / MTP d2 always:
# 1 stream 31.7 / 44.2, 2: 50.4 / 51.9, 3: 60.6 / 36.6, 4: 66.0 / 41.5, 6: 70.8 / 43.7; MTP held back at 6: 68.9)
export LLAMA_SPEC_MAX_GEN=${LLAMA_SPEC_MAX_GEN:-2}
export LLAMA_PIPELINE_PARALLEL=1
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
export LD_LIBRARY_PATH=$GLM_BIN

GLM_LAYOUT=(
  -ngl 99 -dev ROCm0,ROCm1,ROCm2,ROCm3,ROCm4,ROCm5,ROCm6,ROCm7,ROCm8 -ts 7,4,4,5,5,5,5,5,7
  -fa off                                   # PR #27754: FA's F32->F16 cast breaks MLA precision
  --load-mode none -lzm off -t 16
)
GLM_ARGS=(
  "${GLM_LAYOUT[@]}"
  -c "${GLM_CTX:-32768}" -np "${GLM_NP:-1}" -ub "${GLM_UB:-512}" -b 2048
  --no-cache-idle-slots --cache-ram 65536
  --temp 1.0 --top-p 0.95
  --host 127.0.0.1 --port "$GLM_PORT" --alias glm-5.3-flash --metrics
)
# GLM_SPEC=dflash (DFlash2 block drafter) | mtp (the model's own NextN block, exported self-contained by
# glm53_export_mtp.py; with lossless rejection sampling at temperature > 0). Both drafters run on GPU 1 (ROCm9).
GLM_SPEC=${GLM_SPEC:-dflash}
GLM_MTP=${GLM_MTP:-/home/sixvolts/models/GLM-5.3-Flash-GGUF/MTP/GLM-5.3-Flash-MTP-UD-Q4_K_XL-sc.gguf}
if [ "$GLM_SPEC" = mtp ]; then
  GLM_SPEC_ARGS=(-md "$GLM_MTP" --spec-type draft-mtp --spec-draft-n-max "$GLM_DRAFT_N" -ngld 99 -devd ROCm9)
else
  GLM_SPEC_ARGS=(-md "$GLM_DRAFT" --spec-type draft-dflash --spec-draft-n-max "$GLM_DRAFT_N" -ngld 99 -devd ROCm9)
fi

glm53_require_hives_free() {
  local busy=""
  for u in flash-next flash-next-b; do
    systemctl is-active --quiet "$u" && busy="$busy $u"
  done
  if [ -n "$busy" ]; then
    echo "refusing: production services still running:$busy (stop them first; this needs both hives)" >&2
    return 1
  fi
}
