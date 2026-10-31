"""CPU-only shared-ID compatibility check of the pinned Qwen GGUF pair."""
import argparse
import json
from pathlib import Path
import random
import struct

from .build_phi_token_map import corpus, encoder, snapshot_model
from .common import digest, sha, write
from .qwen_pair import PAIRS, PLAN, HYPOTHESIS


def gguf_metadata(path):
    """Read GGUF v3 metadata only; never read or allocate tensor weights."""
    with Path(path).open('rb') as f:
        def unpack(fmt):
            size = struct.calcsize('<' + fmt)
            data = f.read(size)
            if len(data) != size:
                raise ValueError('truncated GGUF')
            return struct.unpack('<' + fmt, data)[0]
        def string():
            n = unpack('Q')
            data = f.read(n)
            if len(data) != n:
                raise ValueError('truncated GGUF string')
            return data.decode('utf-8')
        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element, n = unpack('I'), unpack('Q')
                return [value(element) for _ in range(n)]
            formats = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
            return unpack(formats[kind])
        if f.read(4) != b'GGUF' or unpack('I') != 3:
            raise ValueError('expected GGUF v3')
        unpack('Q')
        count, result = unpack('Q'), {}
        for _ in range(count):
            key = string()
            result[key] = value(unpack('I'))
        return result


def compatibility(target, draft):
    prefix = 'tokenizer.'
    a = {k:v for k,v in target.items() if k.startswith(prefix)}
    b = {k:v for k,v in draft.items() if k.startswith(prefix)}
    token_key, type_key = 'tokenizer.ggml.tokens', 'tokenizer.ggml.token_type'
    n, m = len(a[token_key]), len(b[token_key])
    common = min(n, m)
    mismatch = [i for i in range(common) if a[token_key][i] != b[token_key][i]
                or a[type_key][i] != b[type_key][i]]
    extras = a if n > m else b
    # GGUF UNKNOWN/UNUSED types are 2 and 5. Extra output padding must never
    # introduce ordinary text tokens absent from the draft.
    extra_types = extras[type_key][common:]
    variable = {token_key, type_key, 'tokenizer.ggml.scores'}
    differences = [k for k in sorted(set(a) | set(b)) if k not in variable and a.get(k) != b.get(k)]
    scores_equal = a.get('tokenizer.ggml.scores', [])[:common] == b.get('tokenizer.ggml.scores', [])[:common]
    valid = (not mismatch and not differences and scores_equal and abs(n-m) <= 128
             and all(t in (2, 5) for t in extra_types))
    return {'compatible': valid, 'target_vocab': n, 'draft_vocab': m,
            'shared_ids_checked': common, 'mismatched_shared_ids': mismatch,
            'extra_tokens': abs(n-m), 'extra_types': sorted(set(extra_types)),
            'metadata_differences': differences, 'scores_equal': scores_equal,
            'target_tokenizer_sha256': digest(a), 'draft_tokenizer_sha256': digest(b),
            'chat_template': a['tokenizer.chat_template'],
            'merges_sha256': digest(a.get('tokenizer.ggml.merges'))}


def run(out):
    from llama_cpp import Llama
    out.mkdir(parents=True, exist_ok=False)
    paths, models = {}, {}
    bindings = {str(p): sha(p) for p in [Path(__file__), PLAN, Path('pixi.lock'),
        Path(__file__).with_name('qwen_pair.py'), Path(__file__).with_name('build_phi_token_map.py'),
        Path(__file__).with_name('phi_token_map.py'), Path(__file__).with_name('common.py')]}
    for role, config in PAIRS.items():
        root = Path('/model-weights') / config['repository'].split('/')[1]
        for filename, size, expected in config['files']:
            path = root / filename
            if path.stat().st_size != size or sha(path) != expected:
                raise ValueError('checkpoint differs: ' + str(path))
        paths[role] = root / config['files'][0][0]
    result = compatibility(*(gguf_metadata(paths[r]) for r in ('target','draft')))
    write(out / 'metadata.json', result)
    if not result['compatible']:
        raise ValueError('not a native shared-ID tokenizer pair; see metadata.json')
    for role in ('target','draft'):
        models[role] = Llama(str(paths[role]), vocab_only=True, n_gpu_layers=0, n_ctx=16, verbose=False)
    snapshots = {r:snapshot_model(m, r) for r,m in models.items()}
    encode = {r:encoder(m) for r,m in models.items()}
    texts, data_hashes = corpus()
    controls = ['Grüße 世界 🦊 e\u0301', '{"number":123.004,"date":"2027-01-10"}', '\x00\t\n', ' ' * 256,
                '<|im_start|>assistant\n<|im_end|>']
    rng = random.Random(73)
    controls += [''.join(rng.choices('abc 0123456789{}[],:\n\tüé中🦊', k=rng.randrange(1,250))) for _ in range(256)]
    texts += [(f'control/{i}', x.encode()) for i,x in enumerate(controls)]
    rows = []
    for name, data in texts:
        ids = {r:f(data) for r,f in encode.items()}
        if ids['target'] != ids['draft'] or any(snapshots[r].decode(x) != data for r,x in ids.items()):
            raise ValueError('token ID / byte roundtrip failed: ' + name)
        rows.append({'id': name, 'tokens': len(ids['target']), 'identical': True})
    result.update(hypothesis_id=HYPOTHESIS, classification='shared_token_ids_verified', confirmed=False,
                  corpus_checks=rows, source_hashes=bindings, data_hashes=data_hashes,
                  inference_tested=False, mapping_required=False, weights=PAIRS)
    if any(sha(p) != h for p,h in {**bindings,**data_hashes}.items()):
        raise ValueError('producing source or corpus changed')
    write(out / 'campaign_results.json', result)
    print(json.dumps({k:v for k,v in result.items() if k in ('classification','target_vocab','draft_vocab','shared_ids_checked','extra_tokens','extra_types')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    run(p.parse_args().output)
