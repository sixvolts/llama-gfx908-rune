#!/usr/bin/env python3
# GLM-5.3-Flash on rune: offline tools over the (split) GGUF tensor index, no GPU needed.
#   glm53_plan.py typemap  <first-shard.gguf> <out.txt> [--q6k]  explicit --tensor-type-file for llama-quantize: every tensor
#                                                                  mapped to its current type (copied byte-for-byte), and with
#                                                                  --q6k the bandwidth-heavy Q8_0 projections mapped to q6_K
#   glm53_plan.py bytes    <first-shard.gguf>                    per-token decode bytes and per-layer weight sizes
#   glm53_plan.py layout   <first-shard.gguf> [n_gpus] [ctx_tokens] [n_seq]  balanced layer split (-ts) and per-GPU VRAM
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from gguf import GGUFReader  # noqa: E402

GPU_USABLE_GB = 30.5          # MI100 32 GB minus runtime/driver reserve, as measured on rune
COMPUTE_BUF_GB = 3.0          # per-GPU compute buffer headroom at -ub 512 (Flash-Next measured 2.5-3.5)
KV_BYTES_PER_TOKEN_DSA = 1400  # per DSA layer: MLA c_kv 512 f16 + indexer key 128 f16 + pooled key (approx.)
KDA_STATE_BYTES = 64 * 128 * 128 * 4 + 3 * 8192 * 4 * 4   # per KDA layer per sequence: recurrent state + conv state

# Q8_0 tensors that dominate per-token bytes; everything else keeps its type (precision-sensitive MLA/indexer/KDA gates)
Q6K_PATTERNS = [
    r"^blk\.\d+\.attn_(q|k|v|output)\.weight$",      # KDA projections (+ DSA attn_output)
    r"^blk\.\d+\.attn_q_b\.weight$",                 # DSA q up-projection
    r"^blk\.\d+\.ffn_(gate|up|down)_shexp\.weight$",  # shared experts
    r"^blk\.[0-2]\.ffn_(gate|up|down)\.weight$",      # leading dense FFNs
    r"^output\.weight$",
]
MTP_LAYER = 45


def shards(first):
    p = Path(first)
    if "-00001-of-" not in p.name:
        return [p]
    n = int(p.name.split("-of-")[1].split(".")[0])
    return [p.with_name(p.name.replace("-00001-of-", f"-{i:05d}-of-")) for i in range(1, n + 1)]


def tensors(first):
    out = []
    for sh in shards(first):
        r = GGUFReader(str(sh))
        for t in r.tensors:
            out.append((t.name, t.tensor_type.name, [int(d) for d in t.shape], int(t.n_bytes)))
    return out


def layer_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def typemap(first, out, q6k):
    lines, changed, saved = [], 0, 0
    for name, typ, shape, nb in tensors(first):
        new = typ
        if q6k and typ == "Q8_0" and layer_of(name) != MTP_LAYER and any(re.search(p, name) for p in Q6K_PATTERNS):
            new = "Q6_K"
            changed += 1
            saved += nb - nb * 6.5625 / 8.5
        lines.append(f"^{re.escape(name)}$={new.lower()}")
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} tensors, {changed} -> q6_K, file size -{saved/1e9:.2f} GB -> {out}")


def per_token_bytes(ts, n_expert=288, n_used=8):
    tot = 0.0
    parts = defaultdict(float)
    for name, typ, shape, nb in ts:
        il = layer_of(name)
        if il == MTP_LAYER or name.startswith("token_embd") or name.endswith("_norm.weight") or ".bias" in name:
            continue
        if "_exps." in name:
            b = nb * n_used / n_expert
            parts["routed experts"] += b
        elif name == "output.weight":
            b = nb
            parts["lm_head"] += b
        elif "shexp" in name or "ffn_gate_inp" in name:
            b = nb
            parts["shared experts + router"] += b
        elif il is not None and il < 3 and "ffn_" in name:
            b = nb
            parts["dense FFN"] += b
        else:
            b = nb
            parts["attention/KDA/HC/indexer"] += b
        tot += b
    return tot, parts


def cmd_bytes(first):
    ts = tensors(first)
    tot, parts = per_token_bytes(ts)
    print(f"file total {sum(t[3] for t in ts)/1e9:.1f} GB, {len(ts)} tensors")
    print(f"decode bytes/token {tot/1e9:.2f} GB  (floor at 0.9 TB/s: {tot/0.9e12*1e3:.1f} ms)")
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v/1e9:6.2f} GB")


def cmd_layout(first, n_gpu=8, ctx=4 * 131072, n_seq=4, with_mtp=1):
    ts = tensors(first)
    layer_bytes = defaultdict(int)
    glob = 0
    for name, typ, shape, nb in ts:
        il = layer_of(name)
        if il is None:
            if name != "token_embd.weight":   # the input embedding stays in host memory
                glob += nb
        else:
            layer_bytes[il] += nb
    n_layer = max(layer_bytes) + 1
    if not with_mtp:
        n_layer -= 1                      # the NextN block is only loaded for --spec-type draft-mtp
    dsa = {il for il in range(n_layer) if il % 4 == 3 or il == MTP_LAYER}
    state = {il: (KV_BYTES_PER_TOKEN_DSA * ctx if il in dsa else KDA_STATE_BYTES * n_seq) for il in range(n_layer)}
    cost = [layer_bytes[il] + state[il] for il in range(n_layer)]
    extra = [0] * n_gpu
    extra[-1] = glob                      # output layer + norm on the last device
    # exact contiguous partition minimizing the fullest GPU (DP over (layer, gpu))
    INF = float("inf")
    pre = [0]
    for c in cost:
        pre.append(pre[-1] + c)
    best = [[INF] * (n_layer + 1) for _ in range(n_gpu + 1)]
    cut = [[0] * (n_layer + 1) for _ in range(n_gpu + 1)]
    best[0][0] = 0
    for g in range(1, n_gpu + 1):
        for j in range(1, n_layer + 1):
            for i in range(j):
                if best[g - 1][i] == INF:
                    continue
                v = max(best[g - 1][i], pre[j] - pre[i] + extra[g - 1])
                if v < best[g][j]:
                    best[g][j], cut[g][j] = v, i
    bounds, j = [], n_layer
    for g in range(n_gpu, 0, -1):
        i = cut[g][j]
        bounds.append((i, j))
        j = i
    bounds.reverse()
    print(f"{n_layer} layers{'' if with_mtp else ' (MTP layer not loaded)'}, ctx {ctx} tokens total, {n_seq} seqs; "
          f"usable {GPU_USABLE_GB} GB/GPU, {COMPUTE_BUF_GB} GB compute reserve")
    print(f"weights on GPUs {sum(layer_bytes[i] for i in range(n_layer))/1e9 + glob/1e9:.1f} GB, KV+state {sum(state.values())/1e9:.1f} GB")
    ts_arg, worst = [], 1e9
    for g, (i, j) in enumerate(bounds):
        w = sum(layer_bytes[k] for k in range(i, j)) + extra[g]
        st = sum(state[k] for k in range(i, j))
        used = (w + st) / 1e9 + COMPUTE_BUF_GB
        worst = min(worst, GPU_USABLE_GB - used)
        ts_arg.append(j - i)
        print(f"  GPU {g}: layers {i:2d}-{j-1:2d} ({j-i:2d})  weights {w/1e9:5.1f}  kv/state {st/1e9:4.1f}  "
              f"+compute {COMPUTE_BUF_GB:.1f} = {used:5.1f} GB  (free {GPU_USABLE_GB - used:4.1f})")
    print(f"  -ts {','.join(str(x) for x in ts_arg)}   tightest GPU free {worst:.1f} GB -> {'FITS' if worst >= 0 else 'DOES NOT FIT'}")


if __name__ == "__main__":
    cmd, first, *rest = sys.argv[1:]
    if cmd == "typemap":
        typemap(first, rest[0], "--q6k" in rest)
    elif cmd == "bytes":
        cmd_bytes(first)
    elif cmd == "layout":
        cmd_layout(first, *(int(x) for x in rest))
    else:
        sys.exit(__doc__)
