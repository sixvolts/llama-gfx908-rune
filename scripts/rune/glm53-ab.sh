#!/usr/bin/env bash
# A/B runs of the GLM test server: for each "tag:ENV=VAL,ENV=VAL[:extra server args]" argument start the server
# (nospec unless GLM_MODE=spec), run the greedy hash/speed client, stop the server (by its PID, args verified).
#   glm53-ab.sh <outdir> "base:LLAMA_GRAPH_REUSE_DISABLE=1" "reuse:" "d3::--spec-draft-n-max 3" ...
# GLM_CLIENT picks the client script (default glm53_client.py), GLM_CLIENT_ARGS adds arguments after the port (@TAG@
# becomes the run's tag), GLM_VRAM=1 records rocm-smi VRAM use once the server is healthy.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/glm53-env.sh"
OUT=$1; shift
mkdir -p "$OUT"
glm53_require_hives_free || exit 1
SPEC=()
[ "${GLM_MODE:-nospec}" = spec ] && SPEC=("${GLM_SPEC_ARGS[@]}")
for arg in "$@"; do
  tag=${arg%%:*}; rest=${arg#*:}; envs=${rest%%:*}; extra=""
  [[ "$rest" == *:* ]] && extra=${rest#*:}
  read -ra EXTRA <<< "$extra"
  ENVV=(); IFS=',' read -ra kv <<< "$envs"; for e in "${kv[@]}"; do [ -n "$e" ] && ENVV+=("$e"); done
  echo "== $tag ${ENVV[*]} ${EXTRA[*]}"
  env "${ENVV[@]}" "$GLM_BIN/llama-server" -m "$GLM_MODEL" "${GLM_ARGS[@]}" "${SPEC[@]}" "${EXTRA[@]}" -lv 3 > "$OUT/server_$tag.log" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 300); do
    curl -s -m 2 "localhost:$GLM_PORT/health" | grep -q '"ok"' && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break
    sleep 2
  done
  if [ $ok = 1 ]; then
    [ "${GLM_VRAM:-0}" = 1 ] && rocm-smi --showmeminfo vram --csv > "$OUT/vram_$tag.csv" 2>&1
    read -ra CARGS <<< "${GLM_CLIENT_ARGS:-}"
    CARGS=("${CARGS[@]//@TAG@/$tag}")
    python3 "$HERE/${GLM_CLIENT:-glm53_client.py}" "$GLM_PORT" "${CARGS[@]}" 2>&1 | tee "$OUT/client_$tag.txt"
  else
    echo "server failed"; tail -5 "$OUT/server_$tag.log"
  fi
  if kill -0 $SRV 2>/dev/null && ps -o args= -p $SRV | grep -q -- "--port $GLM_PORT"; then
    kill -INT $SRV; for i in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
    kill -0 $SRV 2>/dev/null && kill -9 $SRV
  fi
  sleep 3
done
echo AB_DONE
