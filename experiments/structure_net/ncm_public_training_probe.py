"""Compare fixed TRAIN controls without changing the live public curriculum."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_public_training as study

data, broad = study.data, study.pipeline.broad
SCHEMA = 'nal.ncm-public-training-probe.v4'
HYPOTHESIS = 'ncm-public-epoch2-training-probe-v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-public-training-probe-plan.md')
TEST = Path('tests/structure_net/test_ncm_public_training_probe.py')
AMENDMENT = Path('docs/07 - Status Reports/2026-10-27_ncm-public-training-probe-receipt-recovery.md')
INCIDENT = data.ROOT/'20261027_public_epoch2_TRAIN_probe_launch_failure_v1'
INPUT_AMENDMENT = Path('docs/07 - Status Reports/2026-10-27_ncm-public-training-probe-input-recovery.md')
INPUT_INCIDENT = data.ROOT/'20261027_public_epoch2_TRAIN_probe_launch_failure_v2'
OUTPUT_AMENDMENT = Path('docs/07 - Status Reports/2026-10-27_ncm-public-training-probe-output-recovery.md')
OUTPUT_INCIDENT = data.ROOT/'20261027_public_epoch2_TRAIN_probe_pilot_v3'
BROAD_RESULTS = data.ROOT/'20261025_broad_generator_results_v1'
LEGACY_AUDIT = data.ROOT/'20261025_broad_generator_train_audit_v1'
BROAD_CAMPAIGN_SHA = '2c0241923d497be9d92589fe4845d3e7013f7754e0d3588cf4c2094be4b77299'
ROOTS = {s: data.ROOT/f'20261027_public_epoch2_TRAIN_probe_{s}_v4'
         for s in ('pilot', 'run', 'audit', 'results')}
PARENTS = {'BROAD': broad.RUNS['train'], 'epoch2': study.root_for(2, 1)}
LEVELS = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)
UNIT = 'structure-net-ncm-public-epoch4-attempt1.service'
INVOCATION = '4a547d3283d743998a3e2843b17a4992'


@dataclass(frozen=True)
class Config:
    controls: int = 16
    physical_GPU: int = 0
    seed: int = 42
    precision: str = 'FP32'
    TF32: bool = False
    peak_limit_bytes: int = 4_900_000_000
    cache_bytes: int = 256 * 1024**2
    archived_CE_rtol: float = 1e-5
    archived_CE_atol: float = 1e-6
    model_updates: int = 0
    new_judge_requests: int = 0
    final_test_opened: bool = False
    speed_tested: bool = False


CONFIG = Config()


def select(rows, selection):
    ids = selection['diagnostic_ids']
    lookup = {r['id']: r for r in rows}
    if (len(lookup) != len(rows) or len(ids) != 16 or len(set(ids)) != 16
            or any(i not in lookup for i in ids)
            or selection['longest_source_id'] not in ids
            or any(r['split'] != 'TRAIN' for r in rows)):
        raise ValueError('unchanged sixteen unique TRAIN controls required')
    chosen = [lookup[i] for i in ids]
    for row in chosen:
        integers(row['oracle_codes'], 641, 8192)
        integers(row['native_word_ids'], 128, 30528)
    return chosen


def integers(values, size, vocabulary):
    if (len(values) != size or any(type(v) is not int or not 0 <= v < vocabulary for v in values)):
        raise ValueError('complete in-range integer frame required')


@contextmanager
def capture_teacher(head):
    original, captured = head.teacher_logits, []

    def forward(*args, **kwargs):
        logits = original(*args, **kwargs)
        if logits.ndim != 3 or logits.shape[:2] != (1, 641):
            raise ValueError('one complete teacher-forced hierarchy required')
        captured.append(logits.detach().argmax(-1).cpu().tolist()[0])
        return logits

    head.teacher_logits = forward
    try:
        yield captured
    finally:
        head.teacher_logits = original


def compare_archived(actual, expected):
    if set(actual) - {'teacher_codes'} != set(expected):
        raise ValueError('archived diagnostic fields changed')
    if any(actual[k] != v for k, v in expected.items() if k != 'levels'):
        raise ValueError('archived codes, words, counts or text differ')
    if len(actual['levels']) != 12 or len(expected['levels']) != 12:
        raise ValueError('all twelve levels required')
    for a, b in zip(actual['levels'], expected['levels']):
        if (any(a[k] != v for k, v in b.items() if k != 'CE')
                or not math.isfinite(a['CE']) or not math.isfinite(b['CE'])
                or not math.isclose(a['CE'], b['CE'], rel_tol=CONFIG.archived_CE_rtol,
                                    abs_tol=CONFIG.archived_CE_atol)):
            raise ValueError('archived teacher diagnostics differ')


def score_rows(outputs, cases, shuffled=False):
    if len(outputs) != len(cases) or not cases:
        raise ValueError('complete nonempty paired diagnostics required')
    totals = {'teacher_correct_codes': 0, 'rollout_correct_codes': 0,
              'code_positions': len(cases)*641, 'active_word_matches': 0, 'active_words': 0}
    levels = [{'positions': 0, 'teacher_correct': 0, 'rollout_correct': 0} for _ in LEVELS]
    ce = 0.
    for i, (out, row) in enumerate(zip(outputs, cases)):
        source = cases[(i+1) % len(cases)] if shuffled else row
        if (row['split'] != 'TRAIN' or out['id'] != row['id']
                or out['source_id_used'] != source['id'] or out['task_accuracy'] is not None):
            raise ValueError('TRAIN pairing or source intervention differs')
        for key in ('codes', 'teacher_codes'):
            integers(out[key], 641, 8192)
        integers(out['raw_word_ids'], 128, 30528)
        integers(row['oracle_codes'], 641, 8192)
        integers(row['native_word_ids'], 128, 30528)
        if len(out['levels']) != 12:
            raise ValueError('all levels required')
        offset = 0
        for k, length in enumerate(LEVELS):
            target = row['oracle_codes'][offset:offset+length]
            teacher = sum(a == b for a, b in zip(out['teacher_codes'][offset:offset+length], target))
            rollout = sum(a == b for a, b in zip(out['codes'][offset:offset+length], target))
            metric = out['levels'][k]
            if (metric['positions'] != length or metric['teacher_correct'] != teacher
                    or metric['rollout_correct'] != rollout
                    or not math.isfinite(metric['CE']) or metric['CE'] < 0):
                raise ValueError('raw code counts or finite loss differ')
            for key, value in (('positions', length), ('teacher_correct', teacher), ('rollout_correct', rollout)):
                levels[k][key] += value
            totals['teacher_correct_codes'] += teacher
            totals['rollout_correct_codes'] += rollout
            ce += metric['CE']; offset += length
        active = sum(v != 0 for v in row['native_word_ids'])
        matches = sum(a == b and b != 0 for a, b in zip(out['raw_word_ids'], row['native_word_ids']))
        if active == 0 or out['active_words'] != active or out['active_word_matches'] != matches:
            raise ValueError('raw native-word counts differ')
        totals['active_words'] += active; totals['active_word_matches'] += matches
    totals.update(teacher_code_agreement=totals['teacher_correct_codes']/totals['code_positions'],
                  rollout_code_agreement=totals['rollout_correct_codes']/totals['code_positions'],
                  native_word_agreement=totals['active_word_matches']/totals['active_words'],
                  mean_teacher_CE=ce/(len(cases)*12), levels=levels)
    totals['teacher_rollout_gap'] = totals['teacher_code_agreement']-totals['rollout_code_agreement']
    return totals


def legacy_broad_receipt(root):
    if root != PARENTS['BROAD']:
        raise ValueError('legacy receipt support is restricted to the recorded BROAD parent')
    campaign = BROAD_RESULTS/'campaign_results.json'
    if data.sha(campaign) != BROAD_CAMPAIGN_SHA:
        raise ValueError('published BROAD campaign changed')
    bound = data.read(campaign)['artifacts']
    audit_paths = [LEGACY_AUDIT/n for n in ('campaign_results.json', 'actual_tool_receipt.json')]
    if any(bound.get(str(p)) != data.sha(p) for p in audit_paths):
        raise ValueError('published campaign does not bind the exact completed training audit')
    audited = data.read(audit_paths[0])
    audit_receipt = data.read(audit_paths[1])
    value = data.read(root/'actual_tool_receipt.json')
    if (audited['status'] != 'broad_generator_training_independently_replayed'
            or audited['updates'] != 4096 or audited['cases'] != 1376
            or type(value['actual_tool_exit_code']) is not int or value['actual_tool_exit_code'] != 0
            or type(audit_receipt['actual_tool_exit_code']) is not int or audit_receipt['actual_tool_exit_code'] != 0
            or any(audited['artifacts'].get(str(root/n)) != data.sha(root/n) for n in
                   ('producer.json', 'result.json', 'model.pt', 'outputs.json', 'actual_tool_receipt.json'))):
        raise ValueError('complete successful independently audited BROAD evidence required')
    data.verify(audited['artifacts'])


def receipt(root, names=('producer.json', 'result.json')):
    if root == PARENTS['BROAD'] and names == ('producer.json', 'result.json'):
        return legacy_broad_receipt(root)
    value = data.read(root/'actual_tool_receipt.json')
    if type(value['actual_tool_exit_code']) is not int or value['actual_tool_exit_code'] != 0 or any(
            value.get('artifacts', {}).get(str(root/n)) != data.sha(root/n) for n in names):
        raise ValueError('actual successful artifact-bound receipt required')
    data.verify(value['artifacts'])


def verify_training_inputs(paths, bindings):
    if any(bindings.get(str(p)) != data.sha(p) for p in paths):
        raise ValueError('diagnostic inputs differ from the original BROAD training producer')


def prepared_controls():
    paths = [broad.RUNS['prepare']/n for n in
             ('cases.json', 'selection.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [broad.catalog.DEST/n for n in ('cases.json', 'source_index.json', 'feature_manifest.json')]
    verify_training_inputs(paths, data.read(PARENTS['BROAD']/'producer.json')['input_hashes'])
    if data.sha(broad.catalog.DEST/'cases.json') != broad.CASES_SHA:
        raise ValueError('original catalog hash differs')
    rows = data.read(broad.RUNS['prepare']/'cases.json')
    if len(rows) != 1376:
        raise ValueError('all original prepared TRAIN cases required')
    selection = data.read(broad.RUNS['prepare']/'selection.json')
    return select(rows, selection), selection


def live_training():
    raw = subprocess.run(['systemctl', '--user', 'show', UNIT,
        '--property=ActiveState,SubState,MainPID,InvocationID,NRestarts'],
        capture_output=True, text=True, check=True).stdout
    state = dict(line.split('=', 1) for line in raw.splitlines())
    if (state['ActiveState'] != 'active' or state['SubState'] != 'running'
            or state['MainPID'] != '294897' or state['InvocationID'] != INVOCATION
            or state['NRestarts'] != '0'):
        raise ValueError('the unchanged epoch4 training service must remain live')
    return state


def freeze(root, stage, training, environment, cases):
    files = {Path(__file__), PLAN, AMENDMENT, INPUT_AMENDMENT, OUTPUT_AMENDMENT,
             TEST, Path('pyproject.toml'), Path('pixi.lock')}
    files.update(Path('experiments/structure_net').glob('ncm*.py'))
    files.update(training.native.vendor.VENDOR.rglob('*.py'))
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):
                files.add(p)
    paths = [broad.RUNS['prepare']/n for n in ('cases.json', 'selection.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [broad.catalog.DEST/n for n in ('source_index.json', 'feature_manifest.json')]
    paths += [PARENTS['BROAD']/n for n in ('outputs.json', 'shuffled_source_outputs.json')]
    paths += [p/n for p in PARENTS.values() for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'model.pt')]
    paths += [BROAD_RESULTS/'campaign_results.json', *[LEGACY_AUDIT/n for n in
        ('campaign_results.json', 'actual_tool_receipt.json')], *[INCIDENT/n for n in
        ('failure.json', 'actual_tool_receipt.json')], *[INPUT_INCIDENT/n for n in
        ('failure.json', 'actual_tool_receipt.json')], *[OUTPUT_INCIDENT/n for n in
        ('producer.json', 'failure.json', 'actual_tool_receipt.json')]]
    if stage == 'run':
        paths += [ROOTS['pilot']/n for n in ('producer.json', 'result.json', 'outputs.json', 'actual_tool_receipt.json')]
    root.mkdir(); data.write(root/'cases.json', cases)
    (root/'producer_source').mkdir()
    for i, path in enumerate(sorted(files)):
        (root/'producer_source'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    proof = {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS, 'stage': stage,
        'config': asdict(CONFIG), 'environment': environment, 'task_accuracy': None,
        'systems_lifecycle_only_not_quality_evidence': stage == 'pilot',
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
        'input_hashes': {str(p): data.sha(p) for p in [*paths, root/'cases.json']}}
    proof['fingerprint'] = training.q.digest(proof)
    data.write(root/'producer.json', proof)
    return proof


def run(stage):
    root = ROOTS[stage]
    if root.exists():
        raise ValueError('fresh append-only diagnostic root required')
    training, environment = broad.runtime.configure()
    study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0'
            or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3060'):
        raise ValueError('read-only diagnostic uses the idle physical GPU0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(CONFIG.seed); torch.cuda.manual_seed_all(CONFIG.seed)
    before_service = live_training()
    for parent in PARENTS.values():
        receipt(parent); data.completed(parent)
    if stage == 'run':
        receipt(ROOTS['pilot']); pilot = data.completed(ROOTS['pilot'])
        if pilot['status'] != 'TRAIN_probe_pilot_passed':
            raise ValueError('passing real inference/concurrency pilot required')
        training.native.vendor.verify(data.read(ROOTS['pilot']/'producer.json'))
    controls, selection = prepared_controls()
    cases = [controls[0], next(r for r in controls if r['id'] == selection['longest_source_id'])] if stage == 'pilot' else controls
    proof = freeze(root, stage, training, environment, cases)
    data.write(root/'training_service_before.json', before_service)
    index, manifest = [data.read(broad.catalog.DEST/n) for n in ('source_index.json', 'feature_manifest.json')]
    loader = training.preparation.streaming.FeatureCache(index, manifest, max_bytes=CONFIG.cache_bytes)
    tokenizer = training.native.tokenizer()
    model = training.load_initial()
    archives = {False: {r['id']: r for r in data.read(PARENTS['BROAD']/'outputs.json')},
                True: {r['id']: r for r in data.read(PARENTS['BROAD']/'shuffled_source_outputs.json')}}
    outputs, scores = {}, {}
    for arm, parent in PARENTS.items():
        state = data.read(parent/'result.json')
        if data.sha(parent/'model.pt') != state['checkpoint_sha256']:
            raise ValueError('selected checkpoint bytes changed')
        saved = torch.load(parent/'model.pt', map_location='cpu', weights_only=True, mmap=True)
        model.load_state_dict(saved['model'], strict=True); del saved
        expected = state['final_model_sha256']
        if training.native.vendor.module_hash(model) != expected:
            raise ValueError('strict diagnostic checkpoint load differs')
        outputs[arm], scores[arm] = {}, {}
        for shuffled in ((False,) if stage == 'pilot' else (False, True)):
            name = 'shuffled' if shuffled else 'matched'
            with capture_teacher(model.concepts) as captured:
                rows = training.prior.diagnostics(model, cases, loader, tokenizer, shuffle=shuffled)
            if len(captured) != len(rows):
                raise ValueError('missing captured teacher predictions')
            rows = [{**row, 'teacher_codes': codes} for row, codes in zip(rows, captured)]
            scores[arm][name] = score_rows(rows, cases, shuffled)
            if arm == 'BROAD':
                for row in rows:
                    compare_archived(row, archives[shuffled][row['id']])
            if stage == 'pilot' and arm == 'epoch2':
                with capture_teacher(model.concepts) as repeated_codes:
                    repeated = training.prior.diagnostics(model, cases, loader, tokenizer)
                if [{**r, 'teacher_codes': c} for r, c in zip(repeated, repeated_codes)] != rows:
                    raise ValueError('epoch2 diagnostic does not repeat exactly')
            if stage == 'run' and not shuffled and arm == 'epoch2':
                prior = data.read(ROOTS['pilot']/'outputs.json')[arm][name]
                lookup = {r['id']: r for r in rows}
                if any(lookup[r['id']] != r for r in prior):
                    raise ValueError('full diagnostic differs from pilot')
            if (training.native.vendor.module_hash(model) != expected
                    or any(p.grad is not None for p in model.parameters())
                    or torch.cuda.max_memory_allocated() > CONFIG.peak_limit_bytes):
                raise ValueError('model mutation, gradient or memory limit violation')
            outputs[arm][name] = rows
            data.write(root/f'outputs_{arm}_{name}.json', rows)
            print(json.dumps({'arm': arm, 'condition': name, 'cases': len(rows)}), flush=True)
    data.write(root/'outputs.json', outputs)
    data.write(root/'training_service_after.json', live_training())
    training.native.vendor.verify(proof)
    result = {'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS,
        'status': 'TRAIN_probe_'+stage+'_passed', 'cases': len(cases), 'scores': scores,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'model_updates': 0,
        'new_judge_requests': 0, 'task_accuracy': None, 'final_test_opened': False,
        'speed_tested': False, 'artifacts': {str(root/n): data.sha(root/n) for n in
            ['producer.json', 'cases.json', 'outputs.json', 'training_service_before.json', 'training_service_after.json',
             *[f'outputs_{arm}_{name}.json' for arm, conditions in outputs.items() for name in conditions]]}}
    data.write(root/'result.json', result); model.close()
    print(json.dumps({'status': result['status'], 'cases': len(cases), 'task_accuracy': None}), flush=True)


def audit():
    destination, source = ROOTS['audit'], ROOTS['run']
    if destination.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only audit required')
    receipt(source); state = data.completed(source)
    proof = data.read(source/'producer.json')
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes']); data.verify(state['artifacts'])
    cases, outputs = [data.read(source/n) for n in ('cases.json', 'outputs.json')]
    selected = select(data.read(broad.RUNS['prepare']/'cases.json'),
                      data.read(broad.RUNS['prepare']/'selection.json'))
    if cases != selected or set(outputs) != set(PARENTS):
        raise ValueError('both complete fixed TRAIN arms required')
    scores = {arm: {condition: score_rows(rows, cases, condition == 'shuffled')
                   for condition, rows in conditions.items()} for arm, conditions in outputs.items()}
    if (scores != state['scores'] or any(set(v) != {'matched', 'shuffled'} for v in scores.values())):
        raise ValueError('raw replay differs from producer aggregates')
    delta = scores['epoch2']['matched']['rollout_correct_codes']-scores['BROAD']['matched']['rollout_correct_codes']
    classification = 'TRAIN_rollout_'+('improved' if delta > 0 else 'decreased' if delta < 0 else 'unchanged')
    destination.mkdir()
    value = {'schema': SCHEMA, 'status': 'TRAIN_probe_raw_counts_replayed', 'classification': classification,
        'scores': scores, 'matched_rollout_correct_change': delta, 'task_accuracy': None,
        'model_updates': 0, 'new_judge_requests': 0, 'final_test_opened': False,
        'artifacts': {str(source/n): data.sha(source/n) for n in
            ('producer.json', 'result.json', 'cases.json', 'outputs.json', 'actual_tool_receipt.json')}}
    data.write(destination/'result.json', value)
    print(json.dumps({'classification': classification, 'matched_rollout_correct_change': delta}), flush=True)


def report():
    source, root = ROOTS['audit'], ROOTS['results']
    document = Path('docs/08 - Analysis/2026-10-27_ncm-public-training-probe.md')
    if root.exists() or document.exists():
        raise ValueError('fresh report required')
    receipt(source, ('result.json',)); value = data.completed(source); data.verify(value['artifacts'])
    proof = data.read(ROOTS['run']/'producer.json')
    data.verify(proof['source_hashes']); data.verify(proof['input_hashes'])
    lines = ['# Public epoch2: fixed TRAIN-control diagnostic', '',
        'Both semantic DEV milestones remain 0/128. This sixteen-case TRAIN diagnostic is not task accuracy.', '',
        '| Model | Source | Teacher code agreement | Rollout code agreement | Native-word agreement |',
        '|---|---|---:|---:|---:|']
    for arm, conditions in value['scores'].items():
        for condition, v in conditions.items():
            lines.append(f"| {arm} | {condition} | {v['teacher_code_agreement']:.2%} | {v['rollout_code_agreement']:.2%} | {v['native_word_agreement']:.2%} |")
    lines += ['', f"Classification: `{value['classification']}`. Matched rollout correct-code change: {value['matched_rollout_correct_change']}.",
        'Native-word agreement includes non-padding special tokens. These controls were used in training; there is no held-out generalization claim.',
        'All weights stayed fixed during this diagnostic. The epoch4 curriculum continued unchanged on the RTX2060. No semantic judge was called and no speed was measured.']
    root.mkdir(); document.write_text('\n'.join(lines)+'\n')
    data.write(root/'campaign_results.json', {**value, 'hypothesis_id': HYPOTHESIS,
        'artifacts': {str(p): data.sha(p) for p in (source/'result.json', source/'actual_tool_receipt.json', document)}})
    print(json.dumps({'report': str(document), 'classification': value['classification']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('pilot', 'run', 'audit', 'report'))
    args = parser.parse_args()
    root = ROOTS[args.stage] if args.stage in ('pilot', 'run', 'audit') else ROOTS['results']
    existed = root.exists()
    try:
        {'pilot': lambda: run('pilot'), 'run': lambda: run('run'), 'audit': audit, 'report': report}[args.stage]()
    except Exception as error:
        if not existed and root.exists():
            data.write(root/'failure.json', {'exception': type(error).__name__, 'message': str(error), 'task_accuracy': None})
        raise
