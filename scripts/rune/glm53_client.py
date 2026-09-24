#!/usr/bin/env python3
# Greedy seeded prompts against the GLM test server: output hashes (the oracle for bit-exact changes) and decode speed.
import hashlib, json, sys, urllib.request

port = sys.argv[1] if len(sys.argv) > 1 else "18099"
P = [("Explain how a jet engine produces thrust, covering intake, compression, combustion and exhaust, then compare turbojet and turbofan designs.", 256),
     ("Write a Python function that parses an ISO-8601 duration string like P3DT4H5M6S and returns seconds, with two tests.", 256),
     ("Summarize the causes and consequences of the 1929 stock market crash in five paragraphs.", 256)]
hs, tps, acc = [], [], []
for p, n in P:
    body = {"model": "q", "messages": [{"role": "user", "content": p}], "max_tokens": n, "seed": 1, "temperature": 0}
    d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    m = d["choices"][0]["message"]; t = d["timings"]
    hs.append(hashlib.md5(((m.get("reasoning_content") or "") + "|" + (m.get("content") or "")).encode()).hexdigest()[:8])
    tps.append(t["predicted_per_second"])
    if t.get("draft_n"):
        acc.append(f"{t['draft_n_accepted']}/{t['draft_n']}")
print(f"hashes {' '.join(hs)} | decode t/s {' '.join(f'{x:.1f}' for x in tps)}" + (f" | drafts {' '.join(acc)}" if acc else ""))
