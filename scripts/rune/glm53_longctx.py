#!/usr/bin/env python3
# prefill and decode speed vs prompt length on the GLM test server (fresh prompts, no cache)
import json, random, sys, urllib.request
port = sys.argv[1] if len(sys.argv) > 1 else "18099"
random.seed(7)
words = "the quick brown fox jumps over the lazy dog while engineers measure turbine efficiency across altitude pressure and temperature regimes".split()
for nw in [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1500,6000,12000,21000").split(",")]:
    text = " ".join(random.choice(words) + ("," if random.random() < 0.08 else "") for _ in range(nw))
    body = {"model": "q", "messages": [{"role": "user", "content": text + "\n\nSummarize the above in one sentence."}],
            "max_tokens": 128, "seed": 1, "temperature": 0, "cache_prompt": False}
    d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    t = d["timings"]
    print(f"prompt {t['prompt_n']:6d} tok: prefill {t['prompt_per_second']:7.1f} t/s ({t['prompt_ms']/1000:5.1f} s) | decode {t['predicted_per_second']:5.1f} t/s ({t['predicted_n']} tok)")
