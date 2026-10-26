"""Preserve an unknown-exit interruption and audit only committed NCM updates."""
import argparse
import json
import os
from pathlib import Path

import torch

from experiments.neural_architecture_lab import finalize_ncm_public_curriculum as final

data, study, pipeline, auditor = final.data, final.study, final.pipeline, final.audit
ROOT = data.ROOT/'20261026_public_epoch1_interruption_recovery_v1'
ORIGINAL = study.root_for(1, 1)
CHECKPOINT = study.STORAGE/'epoch1_attempt1/step_000256.pt'
CHECKPOINT_SHA = '1d8708a72ab9f304edf22bbfa8b3cf3bfbb26e3692f55e7afae817426be97a97'
IDENTITY = 'c88bd6ec0cf97c1721d95ffa6409aece98954ae86681aba0513405ee3752293f'
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-interruption-recovery.md')
TEST = Path('tests/structure_net/test_ncm_public_interruption_recovery.py')


def matching_processes(proc=Path('/proc')):
    found = []
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = (directory/'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        if (b'experiments.structure_net.ncm_public_training' in args
                and b'--end-epoch' in args and b'--attempt' in args
                and args[args.index(b'--end-epoch')+1] == b'1'
                and args[args.index(b'--attempt')+1] == b'1'):
            found.append(int(directory.name))
    return found


def trace_extent(raw):
    lines = raw.splitlines(keepends=True)
    if not lines or any(not line.endswith(b'\n') for line in lines):
        raise ValueError('complete retained trace rows required')
    for step, line in enumerate(lines, 1):
        if json.loads(line)['step'] != step:
            raise ValueError('original trace order differs')
    if len(lines) != 335:
        raise ValueError('interruption observation must retain the exact335 recorded rows')
    return {'recorded_updates': len(lines), 'committed_updates': 256,
            'uncommitted_tail_updates': len(lines)-256}


def require_interruption(value):
    expected = {'status': 'interrupted_public_prefix256_verified',
        'original_root': str(ORIGINAL), 'original_tool_exit_code': None,
        'original_tool_session': 68593, 'original_process_pid': 3056752,
        'tool_handle_missing_observed': True, 'matching_processes': [],
        'original_result_present': False, 'checkpoint_sha256': CHECKPOINT_SHA,
        'training_identity': IDENTITY, 'recorded_updates': 335,
        'committed_updates': 256, 'uncommitted_tail_updates': 79,
        'task_accuracy': None, 'final_test_opened': False}
    if any(k not in value or value[k] != v for k, v in expected.items()):
        raise ValueError('exact unknown-exit interrupted ancestor required')
    replay = value['prefix_replay']
    if (replay['first_completed_update'] != 0 or replay['last_completed_update'] != 256
            or replay['complete_supplied_trace']
            or replay['presentations'] != {'public': 24576, 'anchor': 8192}):
        raise ValueError('independently verified committed prefix required')


def verify():
    if ROOT.exists() or study.root_for(1, 2).exists():
        raise ValueError('fresh interruption verification before recovery launch required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only recovery verification required')
    if matching_processes() or any((ORIGINAL/n).exists() for n in
            ('result.json', 'actual_tool_receipt.json', 'failure_actual_tool_receipt.json')):
        raise ValueError('original attempt must be stopped without a known terminal receipt')
    training, environment = pipeline.broad.runtime.configure()
    proof = data.read(ORIGINAL/'producer.json')
    training.native.vendor.verify(proof)
    if proof['fingerprint'] != IDENTITY:
        raise ValueError('unchanged original training identity required')
    final.receipt(auditor.PILOT)
    data.verify(data.read(auditor.PILOT/'producer.json')['input_hashes'])
    if data.completed(auditor.PILOT)['status'] != 'public_training_auditor_real_two_updates_passed':
        raise ValueError('qualified auditor required')
    sidecar = study.resume_record(CHECKPOINT, IDENTITY, 1)
    if sidecar['sha256'] != CHECKPOINT_SHA or sidecar['completed_updates'] != 256:
        raise ValueError('original observed recovery checkpoint required')
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    if (saved['completed_updates'] != 256 or saved['producer_fingerprint'] != IDENTITY
            or training.q.digest(saved['curriculum']) != training.q.digest(proof['config'])
            or not {'model', 'optimizer', 'scaler', 'CPU_rng', 'CUDA_rng'} <= saved.keys()
            or len(saved['model']) != 365 or len(saved['CUDA_rng']) != 1
            or saved['CPU_rng'].numel() == 0):
        raise ValueError('complete committed training state required')
    del saved
    raw = (ORIGINAL/'trace.jsonl').read_bytes()
    extent = trace_extent(raw)
    sources = pipeline.ready(training)
    initial = {str(p): data.sha(p) for p in [*final.source_files(), PLAN, TEST, Path(__file__)]}
    paths = [CHECKPOINT, CHECKPOINT.with_suffix('.json'),
        *[ORIGINAL/n for n in ('producer.json', 'segment.json', 'trace.jsonl')],
        *[auditor.PILOT/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]]
    hashes = final.bindings(paths, initial)
    ROOT.mkdir()
    data.write(ROOT/'producer.json', {'schema': 'nal.ncm-public-interruption-verifier.v1',
        'environment': environment, 'input_hashes': hashes, 'original_training_identity': IDENTITY,
        'original_tool_exit_code': None, 'systems_only': True})
    replay = auditor.replay(map(json.loads, raw.splitlines()), auditor.Labels(sources),
                           0, 256, complete_trace=False)
    sources.public.pairs.close(); sources.public.codes.close()
    if matching_processes() or (ORIGINAL/'trace.jsonl').read_bytes() != raw:
        raise ValueError('original producer or retained trace changed during verification')
    data.verify(hashes)
    value = {'status': 'interrupted_public_prefix256_verified', 'original_root': str(ORIGINAL),
        'original_tool_exit_code': None, 'original_tool_session': 68593, 'original_process_pid': 3056752,
        'tool_handle_missing_observed': True, 'matching_processes': [], 'original_result_present': False,
        'observation_tool_chunks': ['0fa44e', 'b5424f', '4a25a1'],
        'checkpoint_sha256': CHECKPOINT_SHA, 'training_identity': IDENTITY, **extent,
        'prefix_replay': replay, 'feature_matrices_loaded': 0, 'new_optimizer_updates': 0,
        'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(ROOT/'producer.json'): data.sha(ROOT/'producer.json')}}
    require_interruption(value)
    data.write(ROOT/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


def configure():
    original_receipt, original_roots = final.receipt, final.roots
    original_files, original_bindings = final.source_files, final.bindings

    def receipt(root, names=('producer.json', 'result.json'), *, failed=False):
        if not failed or root != ORIGINAL:
            return original_receipt(root, names, failed=failed)
        if names != ('producer.json', 'trace.jsonl'):
            raise ValueError('only the original interrupted training ancestor is supported')
        original_receipt(ROOT)
        value = data.completed(ROOT); require_interruption(value)
        data.verify(data.read(ROOT/'producer.json')['input_hashes'])
        if matching_processes() or (ORIGINAL/'result.json').exists():
            raise ValueError('interrupted ancestor state changed')
        return ROOT/'actual_tool_receipt.json'

    def roots(epoch):
        value = original_roots(epoch)
        value['training_audit'] = data.ROOT/f'20261026_public_curriculum_epoch{epoch}_training_audit_v2'
        return value

    def bindings(paths, initial):
        return original_bindings([*paths, *[ROOT/n for n in
            ('producer.json', 'result.json', 'actual_tool_receipt.json')]], initial)

    final.receipt, final.roots, final.bindings = receipt, roots, bindings
    final.source_files = lambda: sorted(set(original_files()) | {PLAN.resolve(), TEST.resolve()})


def run(stage, epoch, attempt):
    if stage == 'verify':
        if (epoch, attempt) != (1, 2):
            raise ValueError('fixed original interruption recovery required')
        return verify()
    configure()
    functions = {'audit': final.audit_training, 'replay': final.replay_semantics, 'report': final.report}
    functions[stage](epoch, attempt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('verify', 'audit', 'replay', 'report'))
    parser.add_argument('--epoch', type=int, choices=study.CONFIG.milestones, default=1)
    parser.add_argument('--attempt', type=int, default=2)
    args = parser.parse_args()
    run(args.stage, args.epoch, args.attempt)
