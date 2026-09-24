#!/usr/bin/env python3
# Export the NextN/MTP block of a (split) glm5next GGUF into a SELF-CONTAINED draft-only GGUF: every global tensor
# (token_embd, output_norm, output, hc_head_*, ...) plus blk.<n_layer>.*, all copied byte-for-byte, metadata copied
# minus split.*. glm5next's loader treats a file without blk.0 as a draft-only export (trunk tensors not required) and
# the MTP graph then uses the file's own embeddings and head, so the draft references no target tensor and can run on
# a device the target does not use (rune: GPU 1). Idea from llama-halo-hybrid's export_mtp.py, which instead writes a
# file that borrows the target's tensors (its loader flag does not exist here).
#   glm53_export_mtp.py <first-shard.gguf> <out.gguf>
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from gguf import GGUFReader, GGUFWriter, GGUFValueType  # noqa: E402


def shards(first):
    p = Path(first)
    if "-00001-of-" not in p.name:
        return [p]
    n = int(p.name.split("-of-")[1].split(".")[0])
    return [p.with_name(p.name.replace("-00001-of-", f"-{i:05d}-of-")) for i in range(1, n + 1)]


def main():
    first, out = sys.argv[1], sys.argv[2]
    readers = [GGUFReader(str(s)) for s in shards(first)]
    r0 = readers[0]
    arch = r0.fields["general.architecture"].contents()
    n_all = int(r0.fields[f"{arch}.block_count"].contents())
    n_nextn = int(r0.fields[f"{arch}.nextn_predict_layers"].contents())
    assert n_nextn == 1, n_nextn
    prefix = f"blk.{n_all - n_nextn}."
    w = GGUFWriter(out, arch=arch)
    for key, field in r0.fields.items():
        if key.startswith("GGUF.") or key.startswith("split.") or key == "general.architecture":
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            w.add_key_value(key, field.contents(), vtype, sub_type=field.types[-1])
        else:
            w.add_key_value(key, field.contents(), vtype)
    picked, nbytes = [], 0
    for r in readers:
        for t in r.tensors:
            if t.name.startswith(prefix) or not t.name.startswith("blk."):
                w.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)
                picked.append(t.name)
                nbytes += int(t.n_bytes)
    print(f"{arch}: {len(picked)} tensors ({nbytes/1e9:.2f} GB): {prefix}* plus globals "
          f"{sorted(n for n in picked if not n.startswith('blk.'))}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()


if __name__ == "__main__":
    main()
