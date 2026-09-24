#!/usr/bin/env bash
# Two-hive safety soak for the GLM-5.3-Flash layout: sustained prefill + decode across both XGMI islands, GPU 6 (last
# trunk stage) and GPU 1 (DFlash2 drafter) while the kernel log is watched. Any hit kills the test server at once.
# Lesson behind it (2026-07-29): one clean cross-hive transfer proved nothing; sustained cross-island peer traffic
# collapsed hive A's fabric in ~70 s. Here every cross-island copy is host-staged (peer policy xgmi) - this proves it.
#   glm53-peer-soak.sh [minutes=20] [nospec]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/glm53-env.sh"
MIN=${1:-20}
MODE=${2:-spec}
OUT=${OUT:-/tmp/glm53-soak-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
glm53_require_hives_free || exit 1
if curl -s -m 2 "localhost:$GLM_PORT/health" >/dev/null 2>&1; then
  echo "refusing: something already listens on port $GLM_PORT" >&2; exit 1
fi
sudo -n true 2>/dev/null || { echo "refusing: needs passwordless sudo for dmesg" >&2; exit 1; }

# amdgpu RAS/XGMI/page faults, ring timeouts and resets, plus firmware-first PCIe/memory errors (this board keeps AER
# in firmware: they arrive as GHES / "[Hardware Error]" / mce lines)
FABRIC_RE='xgmi|athub|uncorrect|poison|ras.*(error|event)|gpu reset|reset (begin|succeeded|failed)|ring .* timeout|amdgpu.*(fault|hang|lost)|Memory access fault|Hardware Error|GHES|mce: |device lost|DPC|AER'
# follow only NEW kernel messages (never clear the ring buffer: it is evidence for other investigations)
( sudo -n dmesg -W 2>&1 | tee "$OUT/dmesg-raw.txt" | grep --line-buffered -v 'Failed to map peer' \
    | grep -iE --line-buffered "$FABRIC_RE" > "$OUT/dmesg-hits.txt" ) &
WATCH=$!
MARK="glm53-soak-$$-$(date +%s)"
echo "$MARK" | sudo -n tee /dev/kmsg >/dev/null
for i in $(seq 1 20); do grep -q "$MARK" "$OUT/dmesg-raw.txt" 2>/dev/null && break; sleep 0.5; done
grep -q "$MARK" "$OUT/dmesg-raw.txt" 2>/dev/null || { echo "refusing: kernel-log watcher is not receiving messages"; kill "$WATCH"; exit 1; }

SPEC=()
[ "$MODE" = spec ] && SPEC=("${GLM_SPEC_ARGS[@]}")
"$GLM_BIN/llama-server" -m "$GLM_MODEL" "${GLM_ARGS[@]}" "${SPEC[@]}" > "$OUT/server.log" 2>&1 &
SRV=$!
LOAD=""
echo "server pid $SRV, logs in $OUT"
kill_server_now() {   # on a fabric error: no graceful shutdown
  if kill -0 "$SRV" 2>/dev/null && ps -o args= -p "$SRV" | grep -q -- "--port $GLM_PORT"; then kill -9 "$SRV"; fi
}
cleanup() {
  [ -n "$LOAD" ] && kill "$LOAD" 2>/dev/null
  if kill -0 "$SRV" 2>/dev/null && ps -o args= -p "$SRV" | grep -q -- "--port $GLM_PORT"; then
    kill -INT "$SRV"; for i in $(seq 1 30); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
    kill -0 "$SRV" 2>/dev/null && kill -9 "$SRV"
  fi
  pkill -P "$WATCH" 2>/dev/null; kill "$WATCH" 2>/dev/null
}
trap cleanup EXIT
fabric_check() {
  if [ -s "$OUT/dmesg-hits.txt" ]; then
    kill_server_now; echo "FABRIC/HW ERROR - server killed:"; cat "$OUT/dmesg-hits.txt"; exit 2
  fi
  kill -0 "$WATCH" 2>/dev/null || { kill_server_now; echo "kernel-log watcher died - aborting"; exit 3; }
}

ready=0
for i in $(seq 1 300); do
  fabric_check
  kill -0 "$SRV" 2>/dev/null || { echo "server died during load"; tail -20 "$OUT/server.log"; exit 1; }
  curl -s -m 2 "localhost:$GLM_PORT/health" | grep -q '"ok"' && { ready=1; break; }
  sleep 2
done
[ $ready = 1 ] || { echo "server not healthy after 600 s"; tail -20 "$OUT/server.log"; exit 1; }

# which device pairs are staged (ROCm0-3 = island A, ROCm4-7 = island B, ROCm8 = GPU 6, ROCm9 = GPU 1): an
# intra-island pair here means the XGMI detection failed (safe, but slow) - report it loudly
grep -oE 'devices [0-9]+ -> [0-9]+: no peer copies' "$OUT/server.log" | sort -u | tee "$OUT/staged-pairs.txt"
python3 - "$OUT/staged-pairs.txt" <<'PY'
import re, sys
isl = lambda d: 0 if d < 4 else 1 if d < 8 else 2 + d
bad = [l for l in open(sys.argv[1]) if (m := re.search(r'devices (\d+) -> (\d+)', l)) and isl(int(m[1])) == isl(int(m[2]))]
print("WARNING: intra-island pairs staged (XGMI not detected):", bad) if bad else print("staged pairs: all cross-island, as intended")
PY

# load: 3 concurrent clients alternating ~8k-token prompts (each 512-token ubatch moves 32 MiB across A->B and B->6)
# and 512-token decodes (drafter on GPU 1 every step), for MIN minutes
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
        except Exception:
            with lock:
                stats["err"] += 1
            time.sleep(5)
ts = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(3)]
[t.start() for t in ts]; [t.join() for t in ts]
json.dump(stats, open(f"{out}/load.json", "w")); print("load done", stats)
PY
LOAD=$!

while kill -0 "$LOAD" 2>/dev/null; do
  fabric_check
  kill -0 "$SRV" 2>/dev/null || { echo "server died under load"; tail -30 "$OUT/server.log"; exit 1; }
  sleep 1
done
wait "$LOAD"; LOAD=""
fabric_check
cat "$OUT/load.json" 2>/dev/null
echo "SOAK_PASS: $MIN min, no fabric/hardware errors in the kernel log"
