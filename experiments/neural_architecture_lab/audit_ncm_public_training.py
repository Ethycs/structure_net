"""CPU replay of public NCM sample order and recorded conditioning corruption."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_public_training as study

data, pipeline, curriculum = study.data, study.pipeline, study.curriculum
LEVELS = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-result-verification-plan.md')
TEST = Path('tests/structure_net/test_audit_ncm_public_training.py')
PILOT = data.ROOT/'20261026_public_training_auditor_pilot_v1'


class Labels:
    """Read native labels without decompressing any source feature matrix."""
    def __init__(self, sources):
        self.sources = sources

    def __call__(self, example):
        if example.population == 'anchor':
            row = self.sources.anchors[example.index]
            identity, codes = row['id'], row['oracle_codes']
        else:
            public = self.sources.public
            entry = public.pairs.entries[example.index]
            row = public.codes[example.index]
            if (any(row[k] != entry[k] for k in ('id', 'source_sha256', 'target_sha256', 'native_frame_sha256'))
                    or row['codec_sha256'] != public.codec_sha256):
                raise ValueError('exact admitted public code-label binding required')
            identity, codes = row['id'], row['codes']
        if len(codes) != sum(LEVELS) or any(type(v) is not int or not 0 <= v < 8192 for v in codes):
            raise ValueError('complete integer native code hierarchy required')
        return identity, codes


def replay(records, labels, start, end, *, config=curriculum.CONFIG, complete_trace=True):
    """Check exactly (start,end]; explicit prefix mode never claims the tail."""
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= config.updates):
        raise ValueError('nonempty valid committed update interval required')
    stream, epoch = iter(records), None
    counts = {'public': 0, 'anchor': 0}
    noisy, digest, previous_scale = 0, hashlib.sha256(), None
    for completed in range(start, end):
        try:
            row = next(stream)
        except StopIteration as error:
            raise ValueError('missing committed training update') from error
        number, local = divmod(completed, config.updates_per_epoch)
        if epoch is None or epoch.number != number+1:
            epoch = curriculum.Epoch(number+1, config)
        refs = epoch.update(local)
        if (row['step'] != completed+1 or row['epoch'] != number+1 or row['examples'] != len(refs)
                or row['lr'] != curriculum.rate(completed+1, config)
                or len(row['microbatches']) != math.ceil(len(refs)/config.microbatch)
                or any(not math.isfinite(row[k]) or row[k] < 0 for k in ('weighted_CE', 'gradient_norm'))
                or not math.isfinite(row['scale']) or row['scale'] <= 0
                or previous_scale is not None and row['scale'] < previous_scale):
            raise ValueError('update boundary, geometry, schedule or finite optimizer statistics differ')
        previous_scale = row['scale']
        for offset, micro in zip(range(0, len(refs), config.microbatch), row['microbatches'], strict=True):
            selected = refs[offset:offset+config.microbatch]
            bound = [labels(e) for e in selected]
            gold = torch.tensor([r[1] for r in bound], dtype=torch.long)
            _, changes = study.engine.histories(gold, selected, LEVELS)
            expected = {'ids': [r[0] for r in bound], 'positions': [e.position for e in selected],
                'populations': [e.population for e in selected], 'noisy': [e.noisy for e in selected],
                'history_changes': changes}
            if (any(micro.get(k) != v for k, v in expected.items())
                    or not math.isfinite(micro['weighted_CE']) or micro['weighted_CE'] < 0):
                raise ValueError('case, position, noise stream or code corruption differs')
            for e in selected:
                counts[e.population] += 1
                noisy += int(e.noisy)
        if row['weighted_CE'] != sum(m['weighted_CE'] for m in row['microbatches']):
            raise ValueError('recorded effective-update loss differs from microbatch contributions')
        digest.update(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n')
    exhausted = object()
    if complete_trace and next(stream, exhausted) is not exhausted:
        raise ValueError('unaccounted training update after requested end')
    return {'first_completed_update': start, 'last_completed_update': end, 'updates': end-start,
        'presentations': counts, 'noisy_presentations': noisy, 'sample_order_replayed': True,
        'recorded_history_changes_replayed': True, 'complete_supplied_trace': complete_trace,
        'audited_records_sha256': digest.hexdigest()}


def pilot():
    if PILOT.exists():
        raise ValueError('fresh append-only auditor qualification required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only training-record replay required')
    training, environment = pipeline.broad.runtime.configure()
    passed = data.completed(pipeline.ROOT)
    if passed['status'] != 'public_mixed_effective128_pipeline_passed' or not passed['strict_resumed_update_passed']:
        raise ValueError('actual successful mixed-data lifecycle required')
    sources = pipeline.ready(training)
    paths = [PLAN, TEST, *[pipeline.ROOT/n for n in ('producer.json', 'result.json',
        'actual_tool_receipt.json', 'update_1.json', 'update_2.json', 'resumed_update_2.json')]]
    paths += [pipeline.bundle.ROOT/n for n in ('manifest.json', 'result.json', 'actual_tool_receipt.json')]
    hashes = data.start(PILOT, 'public_training_auditor_pilot', paths, environment,
        {'systems_only': True, 'updates': 2, 'native_code_positions': sum(LEVELS),
         'feature_matrices_loaded': 0, 'new_judge_requests': 0, 'final_test_opened': False})
    labels = Labels(sources)
    rows = [data.read(pipeline.ROOT/f'update_{i}.json') for i in (1, 2)]
    full = replay(rows, labels, 0, 2)
    resumed = data.read(pipeline.ROOT/'resumed_update_2.json')
    if resumed['microbatches'] != rows[1]['microbatches']:
        raise ValueError('actual restarted second update differs')
    replayed = replay([resumed], labels, 1, 2)
    data.verify(hashes)
    sources.public.pairs.close(); sources.public.codes.close()
    result = {'status': 'public_training_auditor_real_two_updates_passed', 'full': full,
        'resumed': replayed, 'public_cases_available': 162353, 'reviewed_cases_available': 1376,
        'feature_matrices_loaded': 0, 'model_training': False, 'systems_only': True,
        'new_judge_requests': 0, 'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(PILOT/'producer.json'): data.sha(PILOT/'producer.json')}}
    data.write(PILOT/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('pilot',))
    parser.parse_args()
    pilot()
