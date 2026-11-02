"""Inventory and selectively requantize the two remaining CPU expert banks."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import zlib

from .common import sha, write
from .native_speed import BINARY, IMAGE
from .prepare_qwen38_moe import WEIGHTS, SHA256
from .check_qwen_tokenizers import gguf_metadata

# GGML block layouts from the pinned b11960 gguf/constants.py. Reject unknown
# types rather than guessing their byte size. These cover this checkpoint.
TYPES = {0: ('f32', 1, 4), 1: ('f16', 1, 2), 8: ('q8_0', 32, 34),
         10: ('q2_k', 256, 84), 11: ('q3_k', 256, 110),
         12: ('q4_k', 256, 144), 13: ('q5_k', 256, 176),
         14: ('q6_k', 256, 210), 15: ('q8_k', 256, 292), 30: ('bf16', 1, 2)}
SELECTED = {f'blk.{i}.ffn_{part}_exps.weight' for i in (24, 25) for part in ('down', 'gate', 'up')}
PLAN = Path('docs/07 - Status Reports/word-games/2026-10-31_word-games-cpu-expert-elimination-plan.md')


def tensor_table(path):
    """Read only the GGUF directory, skipping large tokenizer metadata arrays."""
    size = Path(path).stat().st_size
    with Path(path).open('rb') as f:
        def unpack(fmt):
            n = struct.calcsize('<' + fmt)
            data = f.read(n)
            if len(data) != n:
                raise ValueError('truncated GGUF')
            return struct.unpack('<' + fmt, data)[0]

        def skip(n):
            if n < 0 or f.tell() + n > size:
                raise ValueError('truncated GGUF value')
            f.seek(n, 1)

        def string():
            n = unpack('Q')
            if n > size - f.tell():
                raise ValueError('truncated GGUF string')
            return f.read(n).decode('utf-8')

        primitive = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip_value(kind):
            if kind in primitive:
                skip(primitive[kind])
            elif kind == 8:
                skip(unpack('Q'))
            elif kind == 9:
                element, n = unpack('I'), unpack('Q')
                if element in primitive:
                    skip(n * primitive[element])
                else:
                    for _ in range(n):
                        skip_value(element)
            else:
                raise ValueError('unsupported metadata type')

        if f.read(4) != b'GGUF' or unpack('I') != 3:
            raise ValueError('GGUF v3 required')
        n_tensors, n_kv = unpack('Q'), unpack('Q')
        alignment = 32
        for _ in range(n_kv):
            key, kind = string(), unpack('I')
            if key == 'general.alignment':
                if kind != 4:
                    raise ValueError('alignment must be uint32')
                alignment = unpack('I')
            else:
                skip_value(kind)
        if alignment < 1 or alignment & (alignment - 1):
            raise ValueError('power-of-two alignment required')
        tensors = []
        for _ in range(n_tensors):
            name, ndim = string(), unpack('I')
            if not 1 <= ndim <= 4:
                raise ValueError('unsupported tensor rank')
            shape = [unpack('Q') for _ in range(ndim)]
            kind, offset = unpack('I'), unpack('Q')
            typename, block, width = TYPES[kind]
            if not all(shape) or shape[0] % block:
                raise ValueError('invalid quantized row shape')
            tensors.append({'name': name, 'shape': shape, 'type': typename,
                            'offset': offset, 'bytes': math.prod(shape) // block * width})
        data_offset = (f.tell() + alignment - 1) // alignment * alignment
        end = data_offset
        names = set()
        for t in sorted(tensors, key=lambda t: t['offset']):
            t['offset'] += data_offset
            if t['name'] in names or t['offset'] < end or t['offset'] + t['bytes'] > size:
                raise ValueError('overlapping, duplicated or truncated tensor')
            names.add(t['name'])
            end = t['offset'] + t['bytes']
    return tensors


def hashes(path, tensors):
    """Hash the complete file and every tensor payload in one streaming pass."""
    whole, result = hashlib.sha256(), {}
    with Path(path).open('rb') as f:
        def consume(n, tensor_hash=None):
            while n:
                block = f.read(min(n, 8 * 1024**2))
                if not block:
                    raise ValueError('truncated weight payload')
                whole.update(block)
                if tensor_hash is not None:
                    tensor_hash.update(block)
                n -= len(block)
        for t in sorted(tensors, key=lambda t: t['offset']):
            consume(t['offset'] - f.tell())
            h = hashlib.sha256()
            consume(t['bytes'], h)
            result[t['name']] = h.hexdigest()
        consume(Path(path).stat().st_size - f.tell())
    return whole.hexdigest(), result


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    if args.weights.exists():
        raise FileExistsError(args.weights)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    write(out / 'producer.json', {'source_sha256': sha(__file__), 'plan_sha256': sha(PLAN),
          'input': str(WEIGHTS), 'expected_input_sha256': SHA256, 'runtime_image': IMAGE,
          'quantizer_hashes': {p.name: sha(p) for p in BINARY.iterdir() if p.is_file() and not p.is_symlink()},
          'output_weights': str(args.weights), 'selected': sorted(SELECTED)})
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    original = tensor_table(WEIGHTS)
    print('Hashing original checkpoint and tensor payloads', flush=True)
    original_sha, old_hashes = hashes(WEIGHTS, original)
    if original_sha != SHA256:
        raise ValueError('original checkpoint identity mismatch')
    chosen = [t for t in original if t['name'] in SELECTED]
    if len(chosen) != 6:
        raise ValueError('six exact expert projections required')
    samples = []
    with WEIGHTS.open('rb') as f:
        for t in chosen:
            raw_bytes = compressed_bytes = 0
            for i in range(16):
                offset = t['offset'] + (t['bytes'] - 1024**2) * i // 15
                f.seek(offset)
                raw = f.read(1024**2)
                encoded = zlib.compress(raw, level=1)
                if zlib.decompress(encoded) != raw:
                    raise ValueError('compression round trip failed')
                raw_bytes += len(raw)
                compressed_bytes += len(encoded)
            samples.append({'tensor': t['name'], 'sample_bytes': raw_bytes,
                            'zlib1_bytes': compressed_bytes, 'ratio': compressed_bytes / raw_bytes})
    banks = {}
    for t in original:
        if '_exps.weight' in t['name']:
            bank = t['name'].split('.')[1]
            banks[bank] = banks.get(bank, 0) + t['bytes']
    inventory = {'tensors': original, 'bank_bytes': banks,
                 'selected_original_bytes': sum(t['bytes'] for t in chosen),
                 'selected_q2_bytes': sum(math.prod(t['shape']) // 256 * 84 for t in chosen),
                 'selected_q3_bytes': sum(math.prod(t['shape']) // 256 * 110 for t in chosen),
                 'lossless_samples': samples, 'original_payload_hashes': old_hashes}
    write(out / 'inventory.json', inventory)
    # Override every tensor explicitly, preserving original types outside the six
    # selected projections. The native quantizer copies same-type tensors.
    type_file = out / 'tensor-types.txt'
    type_file.write_text(''.join('^' + re.escape(t['name']) + '$=' +
                        ('q2_k' if t['name'] in SELECTED else t['type']) + '\n' for t in original))
    command = ['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
        '-v', f'{BINARY.resolve()}:/llama:ro', '-v', f'{WEIGHTS.resolve()}:/input.gguf:ro',
        '-v', f'{args.weights.parent.resolve()}:/weights', '-v', f'{out.resolve()}:/evidence:ro',
        '--entrypoint', '/llama/llama-quantize', IMAGE, '--allow-requantize', '--pure',
        '--tensor-type-file', '/evidence/tensor-types.txt', '--max-buffer-size', '1024',
        '/input.gguf', '/weights/' + args.weights.name, 'Q4_K_M', '12']
    write(out / 'command.json', command)
    print('Requantizing only expert banks 24 and 25', flush=True)
    with (out / 'quantize.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    final = tensor_table(args.weights)
    print('Verifying every unchanged tensor payload', flush=True)
    final_sha, final_hashes = hashes(args.weights, final)
    changed = {name for name, h in final_hashes.items() if old_hashes.get(name) != h}
    old_table, new_table = ({t['name']: t for t in table} for table in (original, final))
    if set(old_table) != set(new_table) or changed != SELECTED:
        raise ValueError('unexpected changed tensor set')
    for name, t in new_table.items():
        old = old_table[name]
        if t['shape'] != old['shape'] or t['type'] != ('q2_k' if name in SELECTED else old['type']):
            raise ValueError('unexpected tensor type/shape change')
    a, b = gguf_metadata(WEIGHTS), gguf_metadata(args.weights)
    metadata_changes = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
    if any(not k.startswith('quantize.') and k not in ('general.quantization_version',) for k in metadata_changes):
        raise ValueError('model or tokenizer metadata changed: ' + str(metadata_changes))
    write(out / 'result.json', {'completed': True, 'lossless': False, 'weights': str(args.weights),
        'weights_sha256': final_sha, 'weights_bytes': args.weights.stat().st_size,
        'original_sha256': original_sha, 'changed_tensors': sorted(changed),
        'unchanged_tensors_verified': len(original) - 6, 'metadata_changes': metadata_changes,
        'tensor_payload_hashes': final_hashes, 'inventory_sha256': sha(out / 'inventory.json'),
        'producer_sha256': sha(out / 'producer.json')})
    print('Selective Q2_K checkpoint verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    run(parser.parse_args())
