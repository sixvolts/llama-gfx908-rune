#!/usr/bin/env python3
# Lossless check for speculative sampling: sample a short, high-entropy raw-text continuation many times (one seed
# per request) and record the tokens. Run once against a nospec server and once against a spec server, then compare
# the per-position token distributions with a two-sample permutation test (chi-square statistic). Tokens at
# positions >= 1 are the ones the speculative verifier emits, so a biased verifier shows up there.
# The reference must be a spec server in compare mode (LLAMA_SPEC_RS=0), not a nospec server: the verify batch's
# logits differ slightly from single-token decode (batch numerics), and at a min-p/top-p boundary that changes the
# truncated support itself (rune GLM-5.3, 2026-09-24: two tokens at p 0.036/0.034 against a min-p cut of 0.034 were
# kept by every verify and dropped by every plain decode, p < 0.001 against nospec but p = 0.84 against compare mode).
#   glm53_rs_dist.py <port> <out.json> [n=400]      (sample; as a glm53-ab client: GLM_CLIENT_ARGS="<dir>/dist_@TAG@.json")
#   glm53_rs_dist.py compare <a.json> <b.json>
import collections, json, random, sys, urllib.request

PROMPT = "Here is a list of fifty randomly chosen English nouns, separated by commas: apple, river,"
CONDS = {"full": {"top_k": 0, "top_p": 1.0, "min_p": 0.0},        # the untruncated target distribution
         "trunc": {"top_k": 0, "top_p": 0.95, "min_p": 0.05}}     # the truncation the drafter mirrors
N_PREDICT = 4


def sample(port, out, n):
    res = {}
    for cond, extra in CONDS.items():
        rows, acc, drafted = [], 0, 0
        for seed in range(1, n + 1):
            body = {"prompt": PROMPT, "n_predict": N_PREDICT, "temperature": 1.0, "seed": seed, "cache_prompt": True,
                    "return_tokens": True, **extra}
            d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
            rows.append(d["tokens"])
            t = d.get("timings", {})
            acc += t.get("draft_n_accepted", 0); drafted += t.get("draft_n", 0)
        res[cond] = rows
        print(f"{cond}: {n} samples" + (f", drafts accepted {acc}/{drafted}" if drafted else ""), flush=True)
    json.dump(res, open(out, "w"))


def chi2(ca, cb):
    na, nb = sum(ca.values()), sum(cb.values())
    s = 0.0
    for k in set(ca) | set(cb):
        a, b = ca.get(k, 0), cb.get(k, 0)
        ea, eb = (a + b) * na / (na + nb), (a + b) * nb / (na + nb)
        s += (a - ea) ** 2 / ea + (b - eb) ** 2 / eb
    return s


def pooled(xs, keep):
    return collections.Counter(x if x in keep else "other" for x in xs)


def perm_test(xa, xb, iters=4000, rng=random.Random(0)):
    # categories seen fewer than 5 times in the pooled sample are merged so the statistic is not dominated by them
    tot = collections.Counter(xa + xb)
    keep = {k for k, c in tot.items() if c >= 5}
    obs = chi2(pooled(xa, keep), pooled(xb, keep))
    allx, na, ge = xa + xb, len(xa), 0
    for _ in range(iters):
        rng.shuffle(allx)
        ge += chi2(pooled(allx[:na], keep), pooled(allx[na:], keep)) >= obs
    return obs, (ge + 1) / (iters + 1), len(keep)


def compare(fa, fb):
    A, B = json.load(open(fa)), json.load(open(fb))
    for cond in CONDS:
        a, b = A[cond], B[cond]
        same0 = sum(x[0] == y[0] for x, y in zip(a, b)) / len(a)
        print(f"{cond}: n={len(a)}/{len(b)}, position-0 token identical per seed: {same0:.3f}")
        for pos in range(N_PREDICT):
            xa = [tuple(r[pos:pos + 1]) for r in a]
            xb = [tuple(r[pos:pos + 1]) for r in b]
            obs, p, k = perm_test(xa, xb)
            print(f"  pos {pos}: chi2 {obs:7.2f} over {k:3d} categories, permutation p = {p:.3f}")
        xa = [tuple(r[:2]) for r in a]; xb = [tuple(r[:2]) for r in b]
        obs, p, k = perm_test(xa, xb)
        print(f"  pos 0-1 joint: chi2 {obs:7.2f} over {k:3d} categories, permutation p = {p:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        sample(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 400)
