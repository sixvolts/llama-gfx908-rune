#!/usr/bin/env python3
"""Print name<TAB>type<TAB>shape for every tensor in a (sharded) GGUF on Hugging Face, reading only headers via HTTP Range."""
import struct, sys, urllib.request
TYPES = {0:"f32",1:"f16",2:"q4_0",3:"q4_1",6:"q5_0",7:"q5_1",8:"q8_0",9:"q8_1",10:"q2_K",11:"q3_K",12:"q4_K",13:"q5_K",14:"q6_K",15:"q8_K",
         16:"iq2_xxs",17:"iq2_xs",18:"iq3_xxs",19:"iq1_s",20:"iq4_nl",21:"iq3_s",22:"iq2_s",23:"iq4_xs",24:"i8",25:"i16",26:"i32",27:"i64",28:"f64",29:"iq1_m",30:"bf16",34:"tq1_0",35:"tq2_0",39:"mxfp4"}
class R:
    def __init__(s, url): s.url, s.buf, s.pos = url, b"", 0
    def need(s, n):
        while s.pos + n > len(s.buf):
            lo = len(s.buf); hi = lo + max(8 << 20, n)
            req = urllib.request.Request(s.url, headers={"Range": f"bytes={lo}-{hi-1}"})
            s.buf += urllib.request.urlopen(req).read()
    def take(s, n): s.need(n); b = s.buf[s.pos:s.pos+n]; s.pos += n; return b
    def u(s, f): return struct.unpack("<"+f, s.take(struct.calcsize(f)))[0]
    def str_(s): return s.take(s.u("Q")).decode()
    def val(s, t):
        F = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
        if t in F: return s.u(F[t])
        if t == 8: return s.str_()
        if t == 9:
            et, n = s.u("I"), s.u("Q")
            return [s.val(et) for _ in range(n)]
        raise ValueError(t)
def tensors(url):
    r = R(url); assert r.take(4) == b"GGUF"; r.u("I")
    nt, nkv = r.u("Q"), r.u("Q")
    kv = {}
    for _ in range(nkv):
        k = r.str_(); t = r.u("I"); v = r.val(t)
        if not isinstance(v, list): kv[k] = v
    out = []
    for _ in range(nt):
        name = r.str_(); nd = r.u("I"); dims = [r.u("Q") for _ in range(nd)]; t = r.u("I"); r.u("Q")
        out.append((name, TYPES.get(t, str(t)), dims))
    return kv, out
if __name__ == "__main__":
    repo, prefix, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    for i in range(1, n+1):
        url = f"https://huggingface.co/{repo}/resolve/main/{prefix}-{i:05d}-of-{n:05d}.gguf"
        kv, ts = tensors(url)
        if i == 1: print("#", {k: v for k, v in kv.items() if "file_type" in k or "quantize" in k}, file=sys.stderr)
        for name, t, dims in ts: print(f"{name}\t{t}\t{'x'.join(map(str, dims))}")
