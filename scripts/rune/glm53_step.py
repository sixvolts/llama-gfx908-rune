#!/usr/bin/env python3
# Per-token decode anatomy from a glm53-trace.sh trace: stage busy/span in pipeline order, the gaps between stages
# and the host gap between tokens, top kernels. Token = a run of >= 200 kernels on the first stage's agent.
#   glm53_step.py <trace_dir> [first_agent=2] [order=2,4,5,6,7,9,10,11,8]
import collections, csv, glob, re, statistics, sys
d = sys.argv[1]
first = sys.argv[2] if len(sys.argv) > 2 else '2'
order = (sys.argv[3] if len(sys.argv) > 3 else '2,4,5,6,7,9,10,11,8').split(',')
kt = glob.glob(d + '/**/*kernel_trace.csv', recursive=True)[0]
ev = []
for r in csv.DictReader(open(kt)):
    n = re.split(r'\((?:void|float|int|const|unsigned|HIP|long|char|block|ggml_cuda|bool|short|__half|half|nv_)', r['Kernel_Name'])[0]
    ev.append((int(r['Start_Timestamp']), int(r['End_Timestamp']), r['Agent_Id'].split()[-1], n.replace('void ', '').replace('(ggml_type)', 't')[:62]))
ev.sort()
runs = []
for i, e in enumerate(ev):
    if runs and runs[-1]['a'] == e[2]:
        runs[-1]['idx'].append(i)
    else:
        runs.append({'a': e[2], 'idx': [i]})
starts = [k for k, r in enumerate(runs) if r['a'] == first and len(r['idx']) >= 200]
toks = [[ev[i] for r in runs[a:b] for i in r['idx']] for a, b in zip(starts, starts[1:])]
toks = toks[len(toks) // 4:]
n = len(toks)
step = [(t[-1][1] - t[0][0]) / 1e6 for t in toks]
nxt = [(toks[i + 1][0][0] - toks[i][0][0]) / 1e6 for i in range(n - 1)]
print(f"{n} decode tokens; token period median {statistics.median(nxt):.2f} ms ({1000/statistics.median(nxt):.1f} t/s under trace)")
busy = collections.defaultdict(list); first_s = collections.defaultdict(list); last_e = collections.defaultdict(list)
for t in toks:
    by = collections.defaultdict(list)
    for e in t: by[e[2]].append(e)
    for a in order:
        L = by.get(a)
        if L:
            busy[a].append(sum(x[1] - x[0] for x in L) / 1e6); first_s[a].append(L[0][0]); last_e[a].append(L[-1][1])
print(" stage  busy_ms  gap_after_ms")
tot_b = tot_g = 0
for i, a in enumerate(order):
    b = statistics.median(busy[a]); tot_b += b
    if i + 1 < len(order):
        nb = order[i + 1]
        g = statistics.median([(s - e) / 1e6 for s, e in zip(first_s[nb], last_e[a])])
    else:
        g = statistics.median([(toks[j + 1][0][0] - last_e[a][j]) / 1e6 for j in range(min(len(last_e[a]), n - 1))])
    tot_g += g
    print(f"  {a:>4} {b:8.2f}  {g:8.3f}{'   <- to next token' if i + 1 == len(order) else ''}")
print(f"  sum busy {tot_b:.2f} ms, sum gaps {tot_g:.2f} ms")
kk = collections.defaultdict(float); kc = collections.Counter()
for t in toks:
    for e in t: kk[e[3]] += (e[1] - e[0]) / 1e6; kc[e[3]] += 1
print(f"kernels/token {sum(kc.values())/n:.0f}, kernel time/token {sum(kk.values())/n:.2f} ms")
for name, v in sorted(kk.items(), key=lambda x: -x[1])[:22]:
    print(f"  {v/n:6.2f} ms {kc[name]/n:6.0f}x {1000*v/kc[name]:7.1f} us  {name}")
