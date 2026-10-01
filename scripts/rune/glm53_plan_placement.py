#!/usr/bin/env python3
"""Pack the GLM-5.3 (glm-dsa) trunk onto the GPUs at tensor granularity, no CPU spill, and print llama-server flags.

Layer HOMES are sequential (device order = layer order): a layer's attention, its KV cache (MLA latent 1152 B/token,
plus 256 B/token of indexer keys on the "full" indexer layers) and its small tensors live on its home. Expert tensors
(1.0 / 1.3 / 1.7 GiB each) are what fragments a 32 GB card, so with --pack dp each device chooses, by subset-sum over
the three expert tensor sizes, the count of each that fills it best, and realizes them from the nearest layers (its own
home layers first, then the following ones); a tensor placed off its home costs that layer a round trip to the device
holding it. --pack seq is plain first-fit in layer order (experts last), which only ever splits a layer across the
boundary to the next device.
Emits:
  -ts <layers per device>   (the layer's home: where its attention, KV cache and compute live)
  -ot <rules>               (exact weight placement; specific tensor rules first, then whole-layer rules)
"""
import argparse, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "gguf-py"))
from gguf import GGUFReader

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--ctx", type=int, default=8192, help="KV cells to reserve (n_ctx, all slots)")
ap.add_argument("--reserve", type=int, default=1100, help="MiB per device kept free for runtime + compute scratch")
ap.add_argument("--extra", default="", help="idx:MiB,... more reserve on given devices (e.g. 9:11000 for the drafter)")
ap.add_argument("--out-dev", type=int, default=9, help="device for output.weight / output_norm")
ap.add_argument("--ndev", type=int, default=10)
ap.add_argument("--vram", type=int, default=32752)
ap.add_argument("--pack", choices=["seq", "dp"], default="dp")
ap.add_argument("--lookahead", type=int, default=1, help="dp: how many layers past the last home a device may draw experts from "
                "(1 = fewest off-home tensors; more only helps when the reach has mixed expert sizes)")
args = ap.parse_args()

MiB = 1 << 20
r = GGUFReader(args.model)
n_layer = int(r.fields["glm-dsa.block_count"].contents()) - int(r.fields["glm-dsa.nextn_predict_layers"].contents())
kv_lora   = int(r.fields["glm-dsa.attention.kv_lora_rank"].contents())
rope_dim  = int(r.fields["glm-dsa.rope.dimension_count"].contents())
KV_BYTES  = 2*(kv_lora + rope_dim)   # f16 MLA latent + rope per token per layer
LID_BYTES = 2*128                    # f16 indexer key per token on full-indexer layers

layers, glob, full = defaultdict(list), {}, set()
for t in r.tensors:
    m = re.match(r"blk\.(\d+)\.", t.name)
    if m:
        L = int(m.group(1))
        if L < n_layer:
            layers[L].append((t.name, int(t.n_bytes)))
            if t.name.endswith("indexer.attn_k.weight"):
                full.add(L)
    else:
        glob[t.name] = int(t.n_bytes)

extra = {}
for item in filter(None, args.extra.split(",")):
    i, m = item.split(":"); extra[int(i)] = int(m)
cap = [(args.vram - args.reserve - extra.get(d, 0))*MiB for d in range(args.ndev)]
cap[args.out_dev] -= glob["output.weight"] + glob["output_norm.weight"]

def kv_bytes(L):
    return args.ctx*(KV_BYTES + (LID_BYTES if L in full else 0))

def is_exp(name):
    return "_exps." in name

def seq_pack(cap):
    """first-fit in layer order; returns (home, place, free) or None"""
    free = list(cap); dev = 0; place = {}; home = {}
    for L in range(n_layer):
        ts = sorted(layers[L], key=lambda x: (is_exp(x[0]), x[0]))
        small = sum(nb for n, nb in ts if not is_exp(n))
        while kv_bytes(L) + small > free[dev]:
            dev += 1
            if dev >= args.ndev:
                return None
        home[L] = dev
        free[dev] -= kv_bytes(L)
        for name, nb in ts:
            while nb > free[dev]:
                dev += 1
                if dev >= args.ndev:
                    return None
            free[dev] -= nb
            place[name] = dev
    return home, place, free

def dp_pack(cap):
    # homes from a sequential pass with inflated capacity (first value that succeeds), so homes track weight balance
    for infl in range(0, 8*1024, 128):
        res = seq_pack([c + infl*MiB for c in cap])
        if res:
            break
    else:
        return None
    home = res[0]
    free = list(cap); place = {}
    for L in range(n_layer):
        d = home[L]
        free[d] -= kv_bytes(L)
        for name, nb in layers[L]:
            if not is_exp(name):
                free[d] -= nb; place[name] = d
    if min(free) < 0:
        return None
    # expert tensors by size class, in layer order
    pool = [(L, name, nb) for L in range(n_layer) for name, nb in sorted(layers[L]) if is_exp(name)]
    sizes = sorted({nb for _, _, nb in pool})
    last_home = [max([L for L in range(n_layer) if home[L] == d], default=-1) for d in range(args.ndev)]
    for d in range(args.ndev):
        if d == args.ndev - 1:
            chosen = pool  # the last device takes whatever is left
        else:
            reach = [p for p in pool if p[0] <= last_home[d] + args.lookahead]
            avail = Counter(nb for _, _, nb in reach)
            # subset-sum over counts per size class: best fill <= free[d]
            best, best_counts = -1, None
            def rec(i, used, counts):
                nonlocal best, best_counts
                if i == len(sizes):
                    if used > best:
                        best, best_counts = used, dict(counts)
                    return
                s = sizes[i]
                for c in range(avail[s], -1, -1):
                    if used + c*s <= free[d]:
                        counts[s] = c
                        rec(i + 1, used + c*s, counts)
                counts[s] = 0
            rec(0, 0, {})
            if os.environ.get("PLAN_DEBUG"):
                print(f"#   dev {d}: free {free[d]/MiB:.0f} avail {{ {', '.join(f'{k/MiB:.0f}:{v}' for k,v in sorted(avail.items()))} }} best {best/MiB:.0f} -> {dict((k/MiB, v) for k, v in best_counts.items())}", file=sys.stderr)
            chosen = []
            # realize: own home layers first, then nearest following layers, then earlier leftovers
            def rank(p):
                L = p[0]
                if home[L] == d: return (0, L)
                if L > last_home[d]: return (1, L)
                return (2, -L)
            for p in sorted(reach, key=rank):
                if best_counts.get(p[2], 0) > 0:
                    chosen.append(p); best_counts[p[2]] -= 1
        for L, name, nb in chosen:
            free[d] -= nb; place[name] = d
        pool = [p for p in pool if p[1] not in place]
    if min(free) < 0 or pool:
        return None
    return home, place, free

res = (dp_pack if args.pack == "dp" else seq_pack)(cap)
if res is None:
    sys.exit(f"does not fit (pack={args.pack}, ctx {args.ctx}, reserve {args.reserve}, extra {extra})")
home, place, free = res

counts = [0]*args.ndev
for L in range(n_layer):
    counts[home[L]] += 1
rules, foreign = [], 0
for L in range(n_layer):
    for n, _ in layers[L]:
        if place[n] != home[L]:
            rules.append(f"^{re.escape(n)}$=ROCm{place[n]}"); foreign += 1
for L in range(n_layer):
    rules.append(f"^blk\\.{L}\\.=ROCm{home[L]}")
rules.append(f"^output\\.weight$=ROCm{args.out_dev}")
rules.append(f"^output_norm\\.weight$=ROCm{args.out_dev}")

split_layers = len({re.match(r"\^blk\\\.(\d+)", x).group(1) for x in rules if x.endswith(f"$=ROCm{0}"[:-1] + "0") or "weight$" in x and x.startswith("^blk")})
print(f"# pack={args.pack} ctx {args.ctx}: KV {KV_BYTES} B/tok/layer + {LID_BYTES} B/tok on {len(full)} full-indexer layers; "
      f"reserve {args.reserve} MiB/dev, extra {extra}; {foreign} expert tensors off their home", file=sys.stderr)
for d in range(args.ndev):
    ls = [L for L in range(n_layer) if home[L] == d]
    kv = sum(kv_bytes(L) for L in ls)
    w = cap[d] - free[d] - kv
    print(f"# ROCm{d}: layers {ls[0] if ls else '-':>2}..{ls[-1] if ls else '-':>2} ({counts[d]:2d}), weights {w/MiB:6.0f} MiB, "
          f"kv {kv/MiB:5.0f} MiB, spare {free[d]/MiB:5.0f} MiB", file=sys.stderr)
print("-ts", ",".join(map(str, counts)))
print("-ot", ",".join(rules))
