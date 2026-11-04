"""Create an auditable mixed-precision diagnostic GGUF without requantization.

Keep the base model's metadata and copy selected donor tensors byte for byte.
Both inputs must have identical architecture/tokenizer metadata, tensor names and
shapes. Output is never written over an existing artifact; readback hashes verify
every tensor before publishing the result.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


def selected(name, group):
    routed = '_exps.' in name
    return routed if group == 'routed_experts' else not routed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--donor', type=Path, required=True)
    p.add_argument('--group', choices=('routed_experts', 'non_experts'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--gguf-python', type=Path,
                   default=Path('.pixi/word-games-sampling-v2/source/gguf-py'))
    a = p.parse_args()
    partial = a.output.with_suffix('.gguf.part')
    for path in (a.output, partial, a.receipt):
        if path.exists():
            raise FileExistsError(path)
    sys.path.insert(0, str(a.gguf_python.resolve()))
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    readers = [GGUFReader(a.base), GGUFReader(a.donor)]
    base, donor = readers
    if base.endianess != donor.endianess:
        raise ValueError('Tensor byte order differs between inputs')
    arch = base.fields['general.architecture'].contents()
    for key in base.fields.keys() | donor.fields.keys():
        if key.startswith(('tokenizer.', arch + '.')) or key == 'general.architecture':
            x, y = base.fields.get(key), donor.fields.get(key)
            if x is None or y is None or x.contents() != y.contents():
                raise ValueError('Metadata mismatch: ' + key)
    tensors = [{t.name: t for t in r.tensors} for r in readers]
    if tensors[0].keys() != tensors[1].keys():
        raise ValueError('Tensor-name mismatch')
    for name in tensors[0]:
        if not np.array_equal(tensors[0][name].shape, tensors[1][name].shape):
            raise ValueError('Tensor-shape mismatch: ' + name)
    chosen = [(tensors[int(selected(t.name, a.group))][t.name], selected(t.name, a.group))
              for t in base.tensors]
    required = sum(t.n_bytes for t, _ in chosen) + 1024**3
    if shutil.disk_usage(a.output.parent).free < required:
        raise RuntimeError('Insufficient disk headroom')
    writer = GGUFWriter(partial, arch=arch, endianess=base.endianess)
    for field in base.fields.values():
        if field.name.startswith('GGUF.') or field.name == 'general.architecture':
            continue
        if field.name == 'general.file_type':
            # MOSTLY_Q4_K_M would falsely describe a custom mixed model.
            continue
        if field.name.startswith(('quantize.imatrix.', 'mradermacher.')):
            continue
        typ = field.types[0]
        writer.add_key_value(field.name, field.contents(), typ,
                             field.types[-1] if typ == GGUFValueType.ARRAY else None)
    writer.add_string('structurenet.diagnostic.mix', a.group)
    writer.add_string('structurenet.diagnostic.base', a.base.name)
    writer.add_string('structurenet.diagnostic.donor', a.donor.name)
    for tensor, _ in chosen:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.n_bytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    rows = []
    for tensor, from_donor in chosen:
        writer.write_tensor_data(tensor.data, tensor_endianess=base.endianess)
        rows.append({'name': tensor.name, 'source': 'donor' if from_donor else 'base',
                     'type': tensor.tensor_type.name, 'bytes': tensor.n_bytes,
                     'sha256': hashlib.sha256(tensor.data).hexdigest()})
    writer.close()
    copied = GGUFReader(partial)
    for tensor, row in zip(copied.tensors, rows, strict=True):
        if tensor.name != row['name'] or hashlib.sha256(tensor.data).hexdigest() != row['sha256']:
            raise ValueError('Tensor copy verification failed: ' + row['name'])
    with partial.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    partial.rename(a.output)
    result = {'base': str(a.base), 'donor': str(a.donor), 'group': a.group,
              'path': str(a.output), 'size': a.output.stat().st_size, 'sha256': digest,
              'tensor_types': dict(Counter(r['type'] for r in rows)),
              'source_bytes': {source: sum(r['bytes'] for r in rows if r['source'] == source)
                               for source in ('base', 'donor')},
              'verified_tensor_copies': len(rows), 'tensors': rows,
              'classification': 'diagnostic_mixed_precision_not_a_release'}
    with a.receipt.open('x') as out:
        json.dump(result, out, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}, indent=2))


if __name__ == '__main__':
    main()
