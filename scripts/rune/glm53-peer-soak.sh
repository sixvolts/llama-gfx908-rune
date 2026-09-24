#!/usr/bin/env bash
# Two-hive safety soak for the GLM-5.3-Flash layout: sustained prefill + decode across both XGMI islands (and the two
# PCIe-only GPUs) while the kernel log is watched for fabric errors. Any hit stops the test server at once.
# Lesson behind it (2026-07-29): one clean cross-hive transfer proved nothing; sustained cross-island peer traffic
# collapsed hive A's fabric in ~70 s. Here all cross-island copies are host-staged (peer policy xgmi) - this proves it.
#   glm53-peer-soak.sh [minutes=20]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/glm53-env.sh"
MIN=${1:-20}
OUT=${OUT:-/tmp/glm53-soak-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
glm53_require_hives_free || exit 1

FABRIC_RE='xgmi|athub|uncorrectable|poison|ras.*error|gpu reset|ring .* timeout|amdgpu.*(fault|hang)|Memory access fault'
# follow only NEW kernel messages (never clear the ring buffer: it is evidence for other investigations)
( sudo dmesg -W 2>/dev/null | grep --line-buffered -v 'Failed to map peer' | grep -iE --line-buffered "$FABRIC_RE" > "$OUT/dmesg-hits.txt" ) &
WATCH=$!

"$GLM_BIN/llama-server" -m "$GLM_MODEL" "${GLM_ARGS[@]}" > "$OUT/server.log" 2>&1 &
SRV=$!
echo "server pid $SRV, logs in $OUT"
stop_server() {
  if kill -0 "$SRV" 2>/dev/null && ps -o args= -p "$SRV" | grep -q -- "--port $GLM_PORT"; then
    kill -INT "$SRV"; for i in $(seq 1 30); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
    kill -0 "$SRV" 2>/dev/null && kill -9 "$SRV"
  fi
  pkill -P "$WATCH" 2>/dev/null; kill "$WATCH" 2>/dev/null
}
trap stop_server EXIT

for i in $(seq 1 300); do
  curl -s -m 2 "localhost:$GLM_PORT/health" | grep -q '"ok"' && break
  kill -0 "$SRV" 2>/dev/null || { echo "server died during load"; tail -20 "$OUT/server.log"; exit 1; }
  [ -s "$OUT/dmesg-hits.txt" ] && { echo "FABRIC ERROR during load:"; cat "$OUT/dmesg-hits.txt"; exit 2; }
  sleep 2
done
grep -iE 'peer access but share no direct XGMI|staged through host' "$OUT/server.log" | sort -u | head -20

# load: 3 concurrent clients alternating a ~8k-token prompt (prefill crosses every island boundary with 32 MB per
# ubatch) and 512-token decodes, for MIN minutes
python3 - "$GLM_PORT" "$MIN" "$OUT" <<'PY' &
import json, random, sys, threading, time, urllib.request
port, minutes, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
words = "turbine pressure ratio compressor stage nozzle combustion airflow bypass thrust altitude density".split()
deadline = time.time() + minutes * 60
stats = {"req": 0, "tok": 0, "err": 0}
lock = threading.Lock()
def worker(k):
    rnd = random.Random(k)
    while time.time() < deadline:
        long = rnd.random() < 0.5
        text = " ".join(rnd.choice(words) for _ in range(6000 if long else 50))
        body = {"model": "q", "messages": [{"role": "user", "content": text + "\nSummarize."}], "max_tokens": 64 if long else 512,
                "cache_prompt": False, "seed": k}
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200))
            with lock:
                stats["req"] += 1; stats["tok"] += d["timings"]["predicted_n"] + d["timings"]["prompt_n"]
        except Exception as e:
            with lock:
                stats["err"] += 1
            time.sleep(5)
ts = [threading.Thread(target=worker, args=(k,)) for k in range(3)]
[t.start() for t in ts]; [t.join() for t in ts]
json.dump(stats, open(f"{out}/load.json", "w")); print("load done", stats)
PY
LOAD=$!

while kill -0 "$LOAD" 2>/dev/null; do
  if [ -s "$OUT/dmesg-hits.txt" ]; then
    echo "FABRIC ERROR under load - stopping the server now:"; cat "$OUT/dmesg-hits.txt"; kill "$LOAD" 2>/dev/null; exit 2
  fi
  kill -0 "$SRV" 2>/dev/null || { echo "server died under load"; tail -30 "$OUT/server.log"; kill "$LOAD"; exit 1; }
  sleep 2
done
wait "$LOAD"
cat "$OUT/load.json" 2>/dev/null
[ -s "$OUT/dmesg-hits.txt" ] && { echo "FABRIC ERRORS:"; cat "$OUT/dmesg-hits.txt"; exit 2; }
echo "SOAK_PASS: $MIN min, no fabric errors"
