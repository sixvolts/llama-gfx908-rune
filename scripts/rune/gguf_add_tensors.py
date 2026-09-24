#!/usr/bin/env python3
# Copy a GGUF and append tensors taken verbatim (same type, raw bytes) from another GGUF (possibly split).
# rune use: make the GLM-5.3-Flash DFlash2 draft self-contained by adding the target's token_embd.weight and
# output.weight, so the draft does not borrow them across devices when it runs on an orphan GPU.
#   gguf_add_tensors.py draft.gguf target-00001-of-00006.gguf out.gguf token_embd.weight output.weight
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from gguf import GGUFReader, GGUFWriter, GGUFValueType  # noqa: E402


def shards(first: str) -> list[str]:
    p = Path(first)
    name = p.name
    if "-00001-of-" not in name:
        return [first]
    n = int(name.split("-of-")[1].split(".")[0])
    return [str(p.with_name(name.replace("-00001-of-", f"-{i:05d}-of-"))) for i in range(1, n + 1)]


def main() -> None:
    src, tgt, out, *names = sys.argv[1:]
    want = set(names)
    extra = {}
    for sh in shards(tgt):
        r = GGUFReader(sh)
        for t in r.tensors:
            if t.name in want:
                extra[t.name] = t
        if len(extra) == len(want):
            break
    missing = want - set(extra)
    if missing:
        sys.exit(f"not found in target: {sorted(missing)}")

    r = GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(out, arch=arch)
    for key, field in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            w.add_key_value(key, field.contents(), vtype, sub_type=field.types[-1])
        else:
            w.add_key_value(key, field.contents(), vtype)
    for t in r.tensors:
        if t.name in want:
            continue
        w.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)   # reader data: byte-shaped for quantized/bf16
    for name in names:
        t = extra[name]
        w.add_tensor(name, np.asarray(t.data), raw_dtype=t.tensor_type)
        print(f"added {name} {t.tensor_type.name} {list(t.shape)}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()


if __name__ == "__main__":
    main()
