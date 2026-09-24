#!/usr/bin/env bash
# rocprofv3 kernel + memory-copy trace of GLM-5.3-Flash decode on the two-hive layout (one greedy chat).
#   glm53-trace.sh <outdir> [spec|nospec] [max_tokens=192]
# Kills only the server it launched (by PID, args verified). Needs both hives free.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/glm53-env.sh"
OUT=$1; MODE=${2:-nospec}; NTOK=${3:-192}
glm53_require_hives_free || exit 1
mkdir -p "$OUT"
SPEC=()
[ "$MODE" = spec ] && SPEC=("${GLM_SPEC_ARGS[@]}")
rocprofv3 --kernel-trace --memory-copy-trace -d "$OUT/trace" --output-format csv -- \
  "$GLM_BIN/llama-server" -m "$GLM_MODEL" "${GLM_ARGS[@]}" "${SPEC[@]}" > "$OUT/server.log" 2>&1 &
RP=$!
for i in $(seq 1 300); do
  curl -s -m 2 "localhost:$GLM_PORT/health" | grep -q '"ok"' && break
  kill -0 $RP 2>/dev/null || { echo "server died"; tail -20 "$OUT/server.log"; exit 1; }
  sleep 2
done
python3 - "$GLM_PORT" "$NTOK" <<'PY'
import json, sys, urllib.request
port, n = sys.argv[1], int(sys.argv[2])
body = {"model": "q", "messages": [{"role": "user", "content": "Explain how a jet engine produces thrust, covering intake, compression, combustion and exhaust, then compare turbojet and turbofan designs."}],
        "max_tokens": n, "seed": 1, "temperature": 0}
d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
t = d["timings"]; print(f"traced: {t['predicted_per_second']:.1f} t/s over {t['predicted_n']} tokens, drafts {t.get('draft_n_accepted')}/{t.get('draft_n')}")
PY
SP=$(pgrep -f "llama-server .*--port $GLM_PORT" | head -1)
if [ -n "$SP" ] && ps -o args= -p "$SP" | grep -q "llama.cpp-glm53/build/bin/llama-server .*--port $GLM_PORT"; then
  kill -INT "$SP"; for i in $(seq 1 60); do kill -0 $RP 2>/dev/null || break; sleep 2; done
  kill -0 "$SP" 2>/dev/null && kill -9 "$SP"
else
  echo "REFUSING to signal pid '$SP' (not the GLM test server)"
fi
wait $RP 2>/dev/null
ls "$OUT"/trace/*/ 2>/dev/null | head; echo TRACE_DONE
