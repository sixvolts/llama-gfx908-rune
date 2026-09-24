#!/usr/bin/env python3
# Summarize a glm53-trace.sh rocprofv3 trace: decode steps (split on idle gaps of the whole pipeline), per-GPU busy
# time per step, per-kernel totals per step, and memory copies per step.
#   glm53_trace_report.py <trace_dir> [gap_ms=3]
import collections, csv, glob, re, statistics, sys

d = sys.argv[1]
gap = float(sys.argv[2]) * 1e6 if len(sys.argv) > 2 else 3e6
kt = glob.glob(d + '/**/*kernel_trace.csv', recursive=True)[0]
mt = (glob.glob(d + '/**/*memory_copy_trace.csv', recursive=True) or [None])[0]
agents = {}
ev = []
for r in csv.DictReader(open(kt)):
    a = r['Agent_Id'].split()[-1]
    n = re.split(r'\((?:void|float|int|const|unsigned|HIP|long|char|block|ggml_cuda|bool|short|__half|half|nv_)', r['Kernel_Name'])[0]
    n = n.replace('void ', '').replace('(ggml_type)', 't')[:70]
    ev.append((int(r['Start_Timestamp']), int(r['End_Timestamp']), a, n))
ev.sort()
# steps: consecutive activity separated by global idle gaps
steps, cur, last_end = [], [], None
for e in ev:
    if last_end is not None and e[0] - last_end > gap and cur:
        steps.append(cur); cur = []
    cur.append(e); last_end = e[1] if last_end is None else max(last_end, e[1])
if cur:
    steps.append(cur)
# decode steps: the most common kernel count (prompt ubatches are much bigger); drop the first quarter (warmup)
cnt = collections.Counter(len(s) for s in steps)
common = cnt.most_common(1)[0][0]
dec = [s for s in steps if abs(len(s) - common) <= max(8, common // 50)]
dec = dec[len(dec) // 4:]
print(f"{len(steps)} activity segments, {len(dec)} decode-like steps (~{common} kernels each)")
span = [(s[-1][1] - s[0][0]) / 1e3 for s in dec]
print(f"step span median {statistics.median(span):.2f} ms")
per_agent = collections.defaultdict(list)
per_kernel = collections.defaultdict(list)
per_kernel_calls = collections.Counter()
for s in dec:
    busy = collections.defaultdict(float); kk = collections.defaultdict(float)
    for st, en, a, n in s:
        busy[a] += (en - st) / 1e3; kk[n] += (en - st) / 1e3; per_kernel_calls[n] += 1
    for a, v in busy.items(): per_agent[a].append(v)
    for n, v in kk.items(): per_kernel[n].append(v)
print("per-GPU busy per step (ms), agent order = rocprof agent id:")
tot = 0
for a in sorted(per_agent, key=lambda x: int(x)):
    m = statistics.median(per_agent[a]); tot += m
    print(f"  agent {a:>3}: {m:7.2f}")
print(f"  sum {tot:.2f} ms (pipeline stages run one after another for one sequence)")
rows = sorted(((statistics.median(v), per_kernel_calls[n] / len(dec), n) for n, v in per_kernel.items()), reverse=True)
print(f"{'ms/step':>8} {'calls':>7} {'us/call':>8}  kernel")
for t, c, n in rows[:30]:
    print(f"{t:8.2f} {c:7.1f} {1e3*t/c:8.1f}  {n}")
if mt:
    cp = collections.Counter(); cb = collections.Counter()
    t0, t1 = dec[0][0][0], dec[-1][-1][1]
    for r in csv.DictReader(open(mt)):
        st = int(r['Start_Timestamp'])
        if t0 <= st <= t1:
            k = r.get('Direction', r.get('Operation', '?'))
            cp[k] += 1
            try: cb[k] += int(r.get('Bytes', 0))
            except ValueError: pass
    for k in cp:
        print(f"copies {k}: {cp[k]/len(dec):.1f}/step, {cb[k]/len(dec)/1e6:.2f} MB/step")
