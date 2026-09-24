#!/usr/bin/env python3
# Sampled traffic at GLM-5.3-Flash's recommended sampler (T=1.0, top-p 0.95): four prompt kinds x two seeds, reports
# per-request decode t/s and draft acceptance and the aggregate (sum of tokens / sum of decode time).
import json, sys, urllib.request
port = sys.argv[1] if len(sys.argv) > 1 else "18099"
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
P = [("prose", "Explain how a jet engine produces thrust, covering intake, compression, combustion and exhaust, then compare turbojet and turbofan designs."),
     ("code", "Write a Python function that parses an ISO-8601 duration string like P3DT4H5M6S and returns seconds, with two tests."),
     ("chat", "I have three days in Lisbon in March. Suggest a relaxed itinerary with food recommendations."),
     ("reason", "A farmer has 17 sheep, all but 9 run away, then he buys twice as many as remain. How many sheep does he have? Explain.")]
tot_n = tot_ms = acc = drf = 0
rows = []
for kind, p in P:
    for seed in (1, 2):
        body = {"model": "q", "messages": [{"role": "user", "content": p}], "max_tokens": 320, "seed": seed,
                "temperature": temp, "top_p": 0.95}
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
            json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
        t = d["timings"]
        tot_n += t["predicted_n"]; tot_ms += t["predicted_ms"]
        a, n = t.get("draft_n_accepted") or 0, t.get("draft_n") or 0
        acc += a; drf += n
        rows.append(f"{kind}/s{seed} {t['predicted_per_second']:.1f}" + (f" ({a}/{n})" if n else ""))
print(" | ".join(rows))
print(f"AGG decode {1000*tot_n/tot_ms:.1f} t/s over {tot_n} tokens" + (f", acceptance {acc}/{drf} = {acc/drf:.2f}" if drf else ""))
