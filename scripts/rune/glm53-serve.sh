#!/usr/bin/env bash
# GLM-5.3-Flash bring-up server on both hives (see glm53-env.sh for the layout). Foreground; Ctrl-C stops it.
#   glm53-serve.sh [spec|nospec] [q4xl|q6mix] [extra llama-server args...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/glm53-env.sh"
MODE=${1:-spec}; shift || true
QUANT=${1:-q4xl}; shift || true
glm53_require_hives_free || exit 1
MODEL=$GLM_MODEL
[ "$QUANT" = q6mix ] && MODEL=$GLM_MODEL_Q6MIX
SPEC=()
[ "$MODE" = spec ] && SPEC=("${GLM_SPEC_ARGS[@]}")
exec "$GLM_BIN/llama-server" -m "$MODEL" "${GLM_ARGS[@]}" "${SPEC[@]}" "$@"
