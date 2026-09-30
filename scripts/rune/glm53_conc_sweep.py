#!/usr/bin/env python3
# Aggregate decode vs concurrency on one server: runs glm53_conc.py (short prompts) at each stream count.
#   glm53_conc_sweep.py <port> [counts=1,2,3,4,6] [max_tokens=512]
import os, subprocess, sys
port = sys.argv[1]; counts = (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4,6").split(","); mt = sys.argv[3] if len(sys.argv) > 3 else "512"
here = os.path.dirname(os.path.abspath(__file__))
for n in counts:
    out = subprocess.run([sys.executable, os.path.join(here, "glm53_conc.py"), port, n, "300", mt], capture_output=True, text=True).stdout
    agg = [l for l in out.splitlines() if l.startswith("AGG")]
    per = [float(l.split("decode")[1].split("tok")[1].split("t/s")[0]) for l in out.splitlines() if l.startswith("req")]
    acc = [l.split("drafts")[1].strip() for l in out.splitlines() if "drafts" in l]
    print(f"n={n}: per-stream {min(per):.1f}-{max(per):.1f} t/s, aggregate {sum(per):.1f} t/s" + (f", drafts {acc[0]}" if acc else ""), flush=True)
