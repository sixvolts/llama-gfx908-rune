# GLM-5.3-Flash on rune: shared launch settings (sourced by glm53-serve.sh / glm53-peer-soak.sh).
# Needs BOTH hives: the flash-next and flash-next-b services must be stopped first (the scripts refuse otherwise).
#
# Layout (scripts/rune/glm53_plan.py layout, UD-Q4_K_XL, 4 x 128k, MTP layer not loaded): 9 pipeline stages
#   Hive A 0,2,3,4 (XGMI) -> Hive B 5,7,8,9 (XGMI) -> GPU 6 (PCIe), DFlash2 drafter on GPU 1 (PCIe).
#   Island crossings (A->B, B->6) and the drafter's inputs go through the host-staged copy path: the peer policy
#   (GGML_CUDA_PEER_POLICY, default xgmi) never issues peer copies outside an XGMI island.
#   Planner: tightest stage 2.9 GB free with a 3 GB compute reserve; ~0.3 GB/GPU more for KDA rollback snapshots.
GLM_BIN=${GLM_BIN:-/home/sixvolts/llama.cpp-glm53/build/bin}
GLM_MODEL=${GLM_MODEL:-/home/sixvolts/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
GLM_MODEL_Q6MIX=/home/sixvolts/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL-Q6mix/GLM-5.3-Flash-UD-Q4_K_XL-Q6mix-00001-of-00006.gguf
GLM_DRAFT=${GLM_DRAFT:-/home/sixvolts/models/GLM-5.3-Flash-DFlash2/GLM-5.3-Flash-DFlash2-Q8_0-sc.gguf}
GLM_PORT=${GLM_PORT:-18099}
GLM_DRAFT_N=${GLM_DRAFT_N:-4}          # physics: block 4-5 beats 8 on MI100 (expert-union cost of each verify row)

export HIP_VISIBLE_DEVICES=0,2,3,4,5,7,8,9,6,1   # ROCm0..8 = trunk stages, ROCm9 = drafter
export LLAMA_PIPELINE_PARALLEL=1
export GLIBC_TUNABLES=glibc.malloc.hugetlb=1
export LD_LIBRARY_PATH=$GLM_BIN

GLM_ARGS=(
  -ngl 99 -dev ROCm0,ROCm1,ROCm2,ROCm3,ROCm4,ROCm5,ROCm6,ROCm7,ROCm8 -ts 7,4,4,5,5,5,5,5,5
  -fa off                                   # PR #27754: FA's F32->F16 cast breaks MLA precision
  --load-mode none -lzm off -t 16
  -c 524288 -np 4 -ub 512 -b 2048
  --no-cache-idle-slots --cache-ram 65536
  --temp 1.0 --top-p 0.95
  --host 127.0.0.1 --port "$GLM_PORT" --alias glm-5.3-flash --metrics
)
GLM_SPEC_ARGS=(
  -md "$GLM_DRAFT" --spec-type draft-dflash --spec-draft-n-max "$GLM_DRAFT_N" -ngld 99 -devd ROCm9
)

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
