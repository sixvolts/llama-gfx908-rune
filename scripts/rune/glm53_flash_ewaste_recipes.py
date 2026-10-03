#!/usr/bin/env python3
"""Per-tensor type files (llama-quantize --tensor-type-file, anchored regex=type, first match wins) for the
GLM-5.3-Flash e-waste variants. Tensor names are read from the unsloth UD file (identical names in mainline glm5-next
after the kv_b split). Mainline llama-quant.cpp keeps hc_*, indexer*, ssm_{f,g}_{a,b}, ssm_beta, attn_kv_a_mqa
unquantized and floors attn_q_a/attn_q_b/nextn.eh_proj at Q8_0 whatever we write here.

Layer map (46 blocks): 0-2 dense FFN; 3-44 MoE (288 routed + 1 shared); 45 = NextN/MTP head (also MoE).
KDA (linear) layers: attn_q/k/v/output + ssm_*; DSA layers (3,7,...,43 and 45): attn_q_a/q_b/kv_a_mqa/k_b/v_b/output.
Cold cut = imatrix ffn_down_exps in_sum2 (per-sample mean): blk.4-12 ~0.003-0.006, 13-28 ~0.01-0.02, 29-40 ~0.02-0.06,
41-44 0.08-0.14, blk.3 0.11 (first MoE layer, hot outlier).
"""
import glob, re, sys, collections
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..', 'gguf-py'))
from gguf import GGUFReader

UD = '/home/sixvolts/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/*.gguf'
names = []
for f in sorted(glob.glob(UD)):
    names += [t.name for t in GGUFReader(f).tensors]
assert len(names) == 1412, len(names)

MTP = 45
def blk(n):
    m = re.match(r'blk\.(\d+)\.', n); return int(m.group(1)) if m else None

def recipe(exps, nonexp, dense_ffn, embd, output, cold=(), cold_exps=None, kda=None, shexp=None):
    """exps(layer, name)->type for routed experts; nonexp: type for every other weight matrix;
    kda: override for attn_q/k/v/output in KDA layers (the per-token bandwidth lever); shexp: shared-expert override."""
    out = []
    for n in names:
        b = blk(n)
        if n.endswith(('norm.weight', 'norm.bias', '.bias', 'ssm_a', 'hc_attn_base.weight', 'hc_attn_scale.weight',
                       'hc_ffn_base.weight', 'hc_ffn_scale.weight', 'ssm_conv1d_q.weight', 'ssm_conv1d_k.weight',
                       'ssm_conv1d_v.weight', 'ffn_gate_inp.weight', 'indexer_compressor_ape.weight', 'indexer.proj.weight')):
            t = 'f32'
        elif n == 'token_embd.weight': t = embd
        elif n == 'output.weight': t = output
        elif '_exps.' in n:
            if b == MTP: t = 'q4_K'                              # MTP head: Q4_K experts in every variant
            elif b in cold: t = cold_exps(b, n)
            else: t = exps(b, n)
        elif 'nextn.eh_proj' in n: t = 'q8_0'
        elif re.search(r'\.ffn_(gate|up|down)\.weight$', n): t = dense_ffn
        elif 'shexp' in n and shexp: t = shexp
        elif kda and re.search(r'\.attn_(q|k|v|output)\.weight$', n) and b not in DSA: t = kda
        else: t = nonexp
        out.append((n, t))
    return out

DSA = {b for b in range(3, 46, 4)} | {MTP}

def write(tag, rows):
    p = f'/home/sixvolts/glm53-flash/recipes/GLM-5.3-Flash-{tag}.tensortypes'
    with open(p, 'w') as o:
        for n, t in rows: o.write(f'^{re.escape(n)}$={t}\n')
    c = collections.Counter(t for _, t in rows)
    print(f'{tag:22s} {len(rows)} tensors  ' + ' '.join(f'{k}:{v}' for k, v in sorted(c.items())))

import os; os.makedirs('/home/sixvolts/glm53-flash/recipes', exist_ok=True)

q4  = lambda b, n: 'q4_K'
q3  = lambda b, n: 'q3_K'
q2  = lambda b, n: 'q2_K'
q3m = lambda b, n: 'q4_K' if 'down' in n else 'q3_K'   # 5.2 Q3_K_M hot pattern: gate/up Q3_K, down Q4_K

# published set
write('Q4_K_XL', recipe(q4, 'q8_0', 'q8_0', 'q8_0', 'q8_0'))
write('Q3_K_XL', recipe(q3, 'q8_0', 'q8_0', 'q8_0', 'q8_0'))
# Q3_K_M cold-cut candidates (experiment A): cold tail Q2_K, everything else the 5.2 Q3_K_M pattern, non-experts Q6_K
write('Q3_K_M-c20', recipe(q3m, 'q6_K', 'q6_K', 'q6_K', 'q6_K', cold=range(4, 21), cold_exps=q2))
write('Q3_K_M-c28', recipe(q3m, 'q6_K', 'q6_K', 'q6_K', 'q6_K', cold=range(4, 29), cold_exps=q2))
# experiment B: the KDA projections (bulk of the per-token non-expert bytes) one notch lower
write('Q3_K_M-c20-kda5', recipe(q3m, 'q6_K', 'q6_K', 'q6_K', 'q6_K', cold=range(4, 21), cold_exps=q2, kda='q5_K'))
write('Q3_K_M-c28-kda5', recipe(q3m, 'q6_K', 'q6_K', 'q6_K', 'q6_K', cold=range(4, 29), cold_exps=q2, kda='q5_K'))
# Q2_K_XL: all routed experts Q2_K (no i-quants), non-experts Q5_K
write('Q2_K_XL', recipe(q2, 'q5_K', 'q5_K', 'q5_K', 'q6_K'))
