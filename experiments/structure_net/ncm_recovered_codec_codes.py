"""Stream fixed frozen-codec targets for every recovered-domain text frame."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from experiments.structure_net import ncm_recovered_codec_corpus as corpus
from experiments.structure_net import ncm_recovered_label_codec as runtime
from experiments.structure_net import ncm_recovered_released_codec as release

data = runtime.data
ROOT = data.ROOT/'20261027_recovered_codec_codes_v1'
STORAGE = Path('/data2/structure_net/ncm_recovered_codec_20261027_v1/codes')
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-recovered-codec-codes.md')
BATCH, LIMIT = 8, 3_500_000_000


def controls(lengths):
    n = len(lengths)
    if n < 32: raise ValueError('complete TRAIN population required')
    candidates = [*range(8), *range(n-8, n), *sorted(range(n), key=lambda i: (-lengths[i], i))[:8],
                  *sorted(range(n), key=lambda i: (lengths[i], i))[:8], *range(n)]
    return list(dict.fromkeys(candidates))[:32]


def run():
    if ROOT.exists() or STORAGE.exists(): raise ValueError('fresh code preparation required')
    receipt = data.read(corpus.ROOT/'actual_tool_receipt.json'); data.verify(receipt['artifacts'])
    state = data.read(corpus.ROOT/'result.json')
    if receipt['actual_tool_exit_code'] != 0 or state['frames'] != 99367 or state['readback_frames'] != 99367:
        raise ValueError('complete actual99367-frame corpus required')
    training, environment = runtime.codec_source.prior.runtime()
    runtime.lifecycle.reference.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060 SUPER' or torch.cuda.mem_get_info()[0] < LIMIT):
        raise ValueError('physicalGPU2 and sufficient free memory required')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    before_services = runtime.lifecycle.services()
    frames = np.memmap(corpus.ROOT/'frames.uint16.bin', dtype='<u2', mode='r', shape=(state['frames'], 128))
    hashes, lengths = [], []
    with (corpus.ROOT/'index.jsonl').open() as stream:
        for i, line in enumerate(stream):
            row = json.loads(line)
            if row['position'] != i or row['offset_bytes'] != i*256 or hashlib.sha256(frames[i].tobytes()).hexdigest() != row['frame_sha256']:
                raise ValueError('complete source frame/index binding differs')
            hashes.append(row['frame_sha256']); lengths.append(row['native_lexical_tokens'])
    if len(hashes) != len(frames): raise ValueError('complete source index required')
    pilot_positions = controls(lengths)
    files = {Path(__file__).resolve(), PLAN.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve()) or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(p)
    paths = [corpus.ROOT/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'frames.uint16.bin', 'index.jsonl')]
    paths += [Path('pixi.lock'), Path('pyproject.toml'), training.native.RELEASE/'files/lm1b/tokenizer.pt', training.native.RELEASE/'files/lm1b/small/config.yaml']
    ROOT.mkdir(); STORAGE.mkdir(parents=True); (ROOT/'producer_source').mkdir()
    for i, p in enumerate(sorted(files)): (ROOT/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema': 'nal.ncm-recovered-codec-codes.v1', 'environment': environment,
        'config': {'cases': 99367, 'batch': BATCH, 'FP32': True, 'TF32': False, 'physical_GPU': 2,
                   'codec_sha256': release.EXPECTED, 'peak_limit_bytes': LIMIT, 'pilot_positions': pilot_positions},
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)}, 'input_hashes': {str(p): data.sha(p) for p in paths}}
    data.write(ROOT/'producer.json', proof)
    torch.cuda.reset_peak_memory_stats(); codec = training.native.load_codec()
    if training.native.vendor.module_hash(codec) != release.EXPECTED: raise ValueError('unchanged released codec required')
    def encode(positions):
        ids = torch.from_numpy(np.asarray(frames[positions], dtype=np.int64).copy()).cuda()
        with torch.no_grad(): value = codec.encode(ids)
        if value.shape != (len(positions), 641) or value.min() < 0 or value.max() >= 8192:
            raise ValueError('complete finite native hierarchies required')
        if torch.cuda.max_memory_allocated() > LIMIT: raise ValueError('fixed own allocation cap exceeded')
        return value.cpu().numpy().astype('<u2')
    singleton = {i: encode([i])[0] for i in pilot_positions}
    for start in range(0, 32, BATCH):
        positions = pilot_positions[start:start+BATCH]; batched = encode(positions)
        if any(not np.array_equal(singleton[i], row) for i, row in zip(positions, batched)):
            raise ValueError('batch-eight/individual control encodings differ')
    data.write(ROOT/'pilot.json', {'positions': pilot_positions, 'all32_batched_codes_match_individual': True,
        'code_hashes': {str(i): hashlib.sha256(v.tobytes()).hexdigest() for i, v in singleton.items()}})
    output_path = STORAGE/'codes.uint16.bin'; index_path = STORAGE/'index.jsonl'; count = 0; started = time.monotonic()
    with output_path.open('xb') as output, index_path.open('x') as index:
        for start in range(0, len(frames), BATCH):
            positions = list(range(start, min(start+BATCH, len(frames)))); values = encode(positions)
            for i, value in zip(positions, values):
                if i in singleton and not np.array_equal(singleton[i], value): raise ValueError('control differs in complete sequential pass')
                raw = value.tobytes(); row = {'position': i, 'frame_sha256': hashes[i], 'offset_bytes': output.tell(),
                    'length_bytes': len(raw), 'code_sha256': hashlib.sha256(raw).hexdigest()}
                output.write(raw); index.write(json.dumps(row)+'\n'); count += 1
            if count % 1024 == 0:
                output.flush(); index.flush()
                print(json.dumps({'encoded': count, 'of': len(frames), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    checked = 0
    with output_path.open('rb') as output, index_path.open() as index:
        for line in index:
            row = json.loads(line)
            if row['position'] != checked or row['frame_sha256'] != hashes[checked] or output.tell() != row['offset_bytes'] or row['length_bytes'] != 1282:
                raise ValueError('output order/frame binding differs')
            raw = output.read(row['length_bytes'])
            if hashlib.sha256(raw).hexdigest() != row['code_sha256'] or np.frombuffer(raw, dtype='<u2').max() >= 8192:
                raise ValueError('saved code record differs')
            checked += 1
        if output.read(1) or count != len(frames) or checked != count: raise ValueError('incomplete output population')
    if training.native.vendor.module_hash(codec) != release.EXPECTED or any(p.grad is not None for p in codec.parameters()):
        raise ValueError('codec changed or received gradients')
    after_services = runtime.lifecycle.services()
    if any(after_services[u]['ExecMainPID'] != before_services[u]['ExecMainPID'] for u in before_services): raise ValueError('other worker identity changed')
    data.write(ROOT/'services.json', {'before': before_services, 'after': after_services})
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    result = {'schema': proof['schema'], 'status': 'all99367_fixed_codes_encoded_and_read_back', 'frames': count,
        'code_width': 641, 'dtype': 'little_endian_uint16', 'batch': BATCH, 'readback_frames': checked,
        'batch_individual_controls_exact': 32, 'codec_unchanged': True, 'codec_sha256': release.EXPECTED,
        'binary': str(output_path), 'index': str(index_path), 'elapsed_seconds': time.monotonic()-started,
        'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'model_training': False, 'semantic_pair_training_admitted': False, 'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(p): data.sha(p) for p in (ROOT/'producer.json', ROOT/'pilot.json', ROOT/'services.json', output_path, index_path)}}
    data.write(ROOT/'result.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args(); existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed: data.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
