#!/usr/bin/env python3
"""Re-header a mainline `glm5-next` GGUF as our fork's `glm5next` (tensor names/data identical between the two) so a
published variant can be benchmarked on the rune fork. The whole KV set is taken from a reference fork-format file
(unsloth UD-Q4_K_XL shard 1) because the fork's loader is known to accept it; only file_type/split keys are refreshed.

usage: bridge_to_fork.py IN.gguf OUT.gguf [--ref REF.gguf]
Writes a single unsplit file (needs disk for a full copy). Tensor bytes are copied verbatim.
"""
import argparse, sys
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..', 'gguf-py'))
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, Keys
from gguf.gguf_reader import ReaderField

REF = '/home/sixvolts/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf'

def kv_value(f: ReaderField):
    t = f.types[0]
    if t == GGUFValueType.ARRAY:
        et = f.types[1]
        if et == GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode('utf-8') for i in f.data], et
        return [f.parts[i][0].item() for i in f.data], et
    if t == GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode('utf-8'), t
    return f.parts[f.data[0]][0].item(), t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--ref', default=REF)
    a = ap.parse_args()
    src = GGUFReader(a.inp); ref = GGUFReader(a.ref)
    src_arch = kv_value(src.fields['general.architecture'])[0]
    ref_arch = kv_value(ref.fields['general.architecture'])[0]
    assert src_arch == 'glm5-next' and ref_arch == 'glm5next', (src_arch, ref_arch)
    w = GGUFWriter(a.out, ref_arch)
    skip = ('general.architecture', 'general.file_type', 'general.quantization_version', 'general.quantized_by',
            'general.repo_url', 'general.size_label', 'general.name')
    for k, f in ref.fields.items():
        if k.startswith(('split.', 'GGUF.')) or k in skip: continue   # GGUF.* are the reader's header pseudo-fields
        v, t = kv_value(f)
        if t == GGUFValueType.ARRAY or isinstance(v, list):
            w.add_array(k, v)
        elif isinstance(v, str): w.add_string(k, v)
        else: w.add_key_value(k, v, f.types[0])
    for k in ('general.file_type', 'general.name', 'general.size_label'):
        if k in src.fields:
            v, t = kv_value(src.fields[k])
            if isinstance(v, str): w.add_string(k, v)
            else: w.add_key_value(k, v, t)
    w.add_string('general.quantized_by', 'SixVolts')
    w.add_string('general.bridged_from', 'glm5-next (mainline) header, re-keyed for the rune fork')
    n_src = {t.name for t in src.tensors}; n_ref = {t.name for t in ref.tensors}
    print(f'tensors: src {len(n_src)}  ref-shard1 {len(n_ref)}  (ref is one shard; name check is per-prefix only)')
    for t in src.tensors:
        # reader exposes quantized tensors as uint8 byte arrays; the writer converts a byte shape back to the quant shape
        w.add_tensor_info(t.name, list(t.data.shape), t.data.dtype, t.n_bytes, raw_dtype=t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in src.tensors:
        w.write_tensor_data(t.data)
    w.close()
    print('wrote', a.out)

if __name__ == '__main__':
    main()
