"""Read-only GGUF metadata, F32 tensor and sampled quantized-weight comparison."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gguf-python', type=Path,
                        default=Path('.pixi/word-games-sampling-v2/source/gguf-py'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.gguf_python.resolve()))
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize
    ref, cand = GGUFReader(args.reference), GGUFReader(args.candidate)
    a, b = ({t.name: t for t in r.tensors} for r in (ref, cand))

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    def field_hash(field):
        h = hashlib.sha256()
        for part in field.parts:
            h.update(part)
        return h.hexdigest()

    fields = []
    for name in sorted(ref.fields.keys() | cand.fields.keys()):
        if name.startswith('GGUF.'):
            continue
        x, y = ref.fields.get(name), cand.fields.get(name)
        if x is None or y is None or field_hash(x) != field_hash(y):
            fields.append({'name': name, 'reference': None if x is None else str(x.contents())[:500],
                           'candidate': None if y is None else str(y.contents())[:500]})
    f32 = []
    for name in sorted(a.keys() & b.keys()):
        x, y = a[name], b[name]
        if x.tensor_type == y.tensor_type == GGMLQuantizationType.F32:
            hx, hy = digest(x.data), digest(y.data)
            f32.append({'name': name, 'exact_match': hx == hy, 'reference_sha256': hx,
                        'candidate_sha256': hy})
    samples = []
    names = ['token_embd.weight', 'output.weight', 'blk.0.attn_q.weight',
             'blk.0.attn_k.weight', 'blk.0.attn_v.weight', 'blk.0.attn_output.weight',
             'blk.0.ffn_gate.weight', 'blk.2.ffn_gate_exps.weight', 'blk.2.ffn_gate_inp.weight']
    for name in names:
        x, y = a[name], b[name]
        def sample(t):
            data = t.data.reshape(-1, t.data.shape[-1])
            indices = np.linspace(0, len(data) - 1, min(128, len(data)), dtype=int)
            return dequantize(data[indices], t.tensor_type).astype(np.float64).reshape(-1)
        u, v = sample(x), sample(y)
        samples.append({'name': name, 'sampling': 'up to 128 evenly spaced rows across all experts',
                        'reference_type': x.tensor_type.name, 'candidate_type': y.tensor_type.name,
                        'values': len(u), 'exact': bool(np.array_equal(u, v)),
                        'cosine': float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))),
                        'relative_l2': float(np.linalg.norm(u - v) / np.linalg.norm(u))})
    result = {'reference': str(args.reference), 'candidate': str(args.candidate),
              'reference_tensors': len(a), 'candidate_tensors': len(b),
              'same_tensor_names': a.keys() == b.keys(),
              'shape_mismatches': [name for name in a.keys() & b.keys()
                                   if not np.array_equal(a[name].shape, b[name].shape)],
              'reference_types': dict(Counter(t.tensor_type.name for t in a.values())),
              'candidate_types': dict(Counter(t.tensor_type.name for t in b.values())),
              'metadata_differences': fields, 'f32_count': len(f32),
              'f32_exact_matches': sum(r['exact_match'] for r in f32),
              'f32_tensors': f32, 'quantized_samples': samples,
              'limitation': 'Sampled quantized matrices cannot prove all pre-quantization weights are identical.'}
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'f32_tensors'}, indent=2))


if __name__ == '__main__':
    main()
