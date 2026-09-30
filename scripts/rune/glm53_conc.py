#!/usr/bin/env python3
# N concurrent long-context requests against the GLM test server (fresh prompts, no cache): per-request prefill/decode
# and the aggregate decode rate over the window where all N are generating. Samples per-GPU VRAM (rocm-smi) meanwhile.
#   glm53_conc.py <port> [n=6] [words=20000] [max_tokens=512] [vram_csv]
import json, random, subprocess, sys, threading, time, urllib.request
port = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 20000; MT = int(sys.argv[4]) if len(sys.argv) > 4 else 512
VCSV = sys.argv[5] if len(sys.argv) > 5 else None
words = "the quick brown fox jumps over the lazy dog while engineers measure turbine efficiency across altitude pressure and temperature regimes".split()
res = [None]*N; done = threading.Event(); peak = {}
def one(i):
    rnd = random.Random(100 + i)
    text = " ".join(rnd.choice(words) + ("," if rnd.random() < 0.08 else "") for _ in range(NW))
    body = {"model": "q", "messages": [{"role": "user", "content": f"[{i}] " + text + "\n\nWrite a long essay about the text above."}],
            "max_tokens": MT, "temperature": 1.0, "top_p": 0.95, "seed": i, "cache_prompt": False}
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=7200))
    res[i] = (t0, time.time(), d["timings"])
def vram():
    while not done.is_set():
        out = subprocess.run("rocm-smi --showmeminfo vram --csv", shell=True, capture_output=True, text=True).stdout
        for line in out.splitlines()[1:]:
            f = line.split(",")
            if len(f) >= 3 and f[2].isdigit():
                peak[f[0]] = max(peak.get(f[0], 0), int(f[2]))
        time.sleep(2)
v = threading.Thread(target=vram, daemon=True); v.start()
ts = [threading.Thread(target=one, args=(i,)) for i in range(N)]
T0 = time.time()
for t in ts: t.start()
for t in ts: t.join()
done.set(); T1 = time.time()
for i, (a, b, t) in enumerate(res):
    print(f"req {i}: prompt {t['prompt_n']} tok prefill {t['prompt_per_second']:.0f} t/s ({t['prompt_ms']/1000:.1f} s) | "
          f"decode {t['predicted_n']} tok {t['predicted_per_second']:.1f} t/s | wall {b-a:.0f} s"
          + (f" | drafts {t['draft_n_accepted']}/{t['draft_n']}" if t.get('draft_n') else ""))
tot_dec = sum(r[2]['predicted_n'] for r in res); tot_pre = sum(r[2]['prompt_n'] for r in res)
print(f"AGG {N} requests in {T1-T0:.0f} s: {tot_pre} prompt + {tot_dec} generated tokens; "
      f"sum of per-request decode rates {sum(r[2]['predicted_per_second'] for r in res):.1f} t/s")
print("VRAM peak (GB): " + " ".join(f"{k}={v/1e9:.1f}" for k, v in sorted(peak.items())))
if VCSV:
    open(VCSV, "w").write("\n".join(f"{k},{v}" for k, v in sorted(peak.items())) + "\n")
