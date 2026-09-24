#!/usr/bin/env bash
# KL gate for tolerance-class changes: llama-perplexity on wikitext-2 with the server's layout, storing the base run's
# logits and scoring a candidate's KL divergence against them.
#   glm53-kl.sh base <base.kld> <log> [ENV=VAL ...]     reference logits (e.g. the dense DSA path)
#   glm53-kl.sh cmp  <base.kld> <log> [ENV=VAL ...]     candidate vs the reference (e.g. LLAMA_DSA_SPARSE=1)
# GLM_KL_CTX (8192: past the 2048-cell top-k, so the DSA selection is active), GLM_KL_CHUNKS (6), GLM_UB (512).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/glm53-env.sh"
mode=$1; kld=$2; log=$3; shift 3
glm53_require_hives_free || exit 1
CTX=${GLM_KL_CTX:-8192}
ARGS=("${GLM_LAYOUT[@]}" -m "$GLM_MODEL" -c "$CTX" -b "$CTX" -ub "${GLM_UB:-512}" --chunks "${GLM_KL_CHUNKS:-6}"
      -f "${GLM_KL_TEXT:-/home/sixvolts/wiki.test.raw}" --kl-divergence-base "$kld")
[ "$mode" = cmp ] && ARGS+=(--kl-divergence)
env "$@" "$GLM_BIN/llama-perplexity" "${ARGS[@]}" > "$log" 2>&1
rc=$?
grep -E "Final estimate|Mean PPL|Mean *KLD|Maximum KLD|99.9% *KLD|99.0% *KLD|Median *KLD|Same top p|RMS Δp" "$log"
exit $rc
