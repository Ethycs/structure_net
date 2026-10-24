"""Expand the qualified evidence-bound teacher to all10000 reserved TRAIN sources."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.structure_net import ncm_evidence_teacher_pilot_v3 as recipe
from experiments.structure_net import ncm_evidence_teacher_qualification as qualification
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

original, q, ROOT, native = recipe.original, recipe.q, recipe.ROOT, recipe.original.native
data = original.data
HYPOTHESIS = 'ncm-evidence-teacher10000-TRAIN-labels-v1'
MODULE = 'experiments.structure_net.ncm_evidence_teacher_bulk'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-evidence-teacher-bulk-plan.md')
TEST = Path('tests/structure_net/test_ncm_evidence_teacher_bulk.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_evidence_teacher_bulk.py')
WORKFLOW = ROOT / '20261024_evidence_teacher10000_workflow_v1'
PREP = ROOT / '20261024_evidence_teacher10000_prepare_v1'
PILOT = ROOT / '20261024_evidence_teacher10000_systems_pilot_v1'
GENERATE = ROOT / '20261024_evidence_teacher10000_candidates_v1'
QUALIFIED_SHA = 'a09911fa275b89100a7c645f25d6cf6b26d6274ab0d7624f27db4ecd7e125b8d'
LABEL_SHA = 'a38d269a7c3a0da9dcb56367172a0ae30468d74ecee1035a120807d09dd32247'
WORKERS = 8


def require_teacher():
    path = qualification.RESULTS / 'campaign_results.json'
    result = q.read(path)
    receipt = q.read(qualification.RESULTS / (qualification.HYPOTHESIS + '_registry_readback.json'))
    if (file_hash(path) != QUALIFIED_SHA or not result['teacher_labels_qualified']
            or result['joint'] != {'cases': 64, 'joint_passed': 63, 'all_valid': True, 'qualified': True}
            or receipt['record']['campaign_sha256'] != QUALIFIED_SHA or receipt['exact']['campaign_sha256'] != QUALIFIED_SHA
            or q.read(qualification.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0
            or q.read(qualification.RESULTS / 'backend_exact_readback.json')['status'] != 'independent_backend_exact_readback_passed'
            or file_hash(recipe.GENERATE / 'labels.json') != LABEL_SHA
            or q.read(qualification.RAW_AUDIT / 'actual_tool_terminal_receipt.json')['exit_code'] != 0):
        raise ValueError('actual qualified63of64 first teacher candidates and backend identity required')
    for path, expected in q.read(qualification.RAW_AUDIT / 'result.json')['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('original first teacher evidence changed:' + path)


def pilot_ids(rows, retained):
    options = [r for r in rows if r['id'] not in retained]
    longest = min(options, key=lambda r: (-r['source_cl100k_tokens'], r['id']))
    key = lambda r: (text_hash(HYPOTHESIS + ':pilot:' + r['id']), r['id'])
    probes = sorted((r for r in options if r['bin'] == 0), key=key)[:3]
    for band in range(1, 5):
        probes.append(min((r for r in options if r['bin'] == band and r['id'] != longest['id']), key=key))
    probes.append(longest)
    if len(probes) != 8 or len({r['id'] for r in probes}) != 8:
        raise ValueError('eight distinct fresh metadata-only source probes required')
    return [r['id'] for r in probes]


def combine(rows, retained, generated):
    if set(retained) & set(generated) or set(retained) | set(generated) != {r['id'] for r in rows}:
        raise ValueError('every source must have exactly one retained or new first candidate')
    return [retained[r['id']] if r['id'] in retained else generated[r['id']] for r in rows]


def producing(root, stage, inputs=()):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(recipe.__file__), recipe.PLAN, recipe.AUDITOR,
        Path(recipe.recovery.__file__), Path(original.__file__), Path(qualification.__file__), qualification.PLAN]
    common = [qualification.RESULTS / 'campaign_results.json', qualification.RESULTS / (qualification.HYPOTHESIS + '_registry_readback.json'),
        qualification.RESULTS / 'backend_exact_readback.json', qualification.WORKFLOW / 'tool_terminal_receipt.json',
        recipe.GENERATE / 'labels.json', qualification.RAW_AUDIT / 'result.json', qualification.RAW_AUDIT / 'actual_tool_terminal_receipt.json']
    p = native.vendor.freeze(root, stage, [*inputs, *common, *files], {'hypothesis_id': HYPOTHESIS,
        'TRAIN_source_population': 10000, 'qualified_recipe': QUALIFIED_SHA, 'workers': WORKERS,
        'transport_concurrency_only_change': '4to8, fresh8case systems pilot before full generation',
        'models': original.MODELS, 'systems': original.SYSTEMS, 'draft_requested_max_tokens': 768,
        'review_requested_max_tokens': 512, 'actual_gateway_usage_cap_not_asserted': True,
        'prompt_token_limit': 6300, 'maximum_attempts': 3, 'no_semantic_or_capacity_retries': True,
        'all64_original_candidates_reused_including_failed_salience_case': True,
        'semantic_filtering': False, 'canonical_cache_changed': False, 'student_trained': False,
        'individual_label_semantic_quality_not_assumed': True, 'task_accuracy': None,
        'all_final_sources_closed': True, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'teacher_bulk_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def prepare():
    require_teacher()
    if file_hash(data.DEST / 'cases.json') != original.old.SOURCE_SHA:
        raise ValueError('unchanged complete10000 reserved TRAIN sources required')
    data.reservation.verify(q.read(data.DEST / 'producer.json'))
    rows = q.read(data.DEST / 'cases.json')
    retained = {r['id']: r for r in q.read(recipe.GENERATE / 'labels.json')}
    if (len(rows) != 10000 or len({r['source_component'] for r in rows}) != 10000
            or len(retained) != 64 or not set(retained) <= {r['id'] for r in rows}):
        raise ValueError('complete10000 distinct TRAIN sources with all64 retained firsts required')
    proof = qualification.reference.verify_public(rows)
    p = producing(PREP, 'all10000_reserved_TRAIN_source_bindings', [data.DEST / n for n in ('cases.json', 'producer.json', 'result.json')]
        + [qualification.reference.PUBLIC_FILE])
    write_json(PREP / 'cases.json', rows)
    write_json(PREP / 'retained64.json', list(retained.values()))
    write_json(PREP / 'pilot_ids.json', pilot_ids(rows, retained))
    write_json(PREP / 'public_proof.json', {'public_file_sha256': qualification.reference.PUBLIC_SHA256, 'exact_sources': proof})
    native.vendor.verify(p)
    write_json(PREP / 'result.json', {'status': 'all10000_public_reserved_TRAIN_sources_prepared', 'cases': 10000, 'retained': 64, 'task_accuracy': None})


def candidate_jobs(rows, root, encoding, tokenizer):
    """At most8pending jobs; save every completed first before replacing it."""
    iterator = iter(rows)
    generated, failures = {}, []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = {}

        def submit():
            row = next(iterator, None)
            if row is not None:
                pending[pool.submit(original.candidate, row, root, encoding, tokenizer)] = row

        for _ in range(WORKERS):
            submit()
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                row = pending.pop(future)
                try:
                    value = future.result()
                    generated[row['id']] = value
                    write_json(root / 'cases' / (row['id'] + '.json'), value)
                except Exception as error:
                    failures.append({'id': row['id'], 'error_type': type(error).__name__, 'error': str(error)})
                    write_json(root / 'local_failures.json', failures)
            if not failures:
                for _ in ready:
                    submit()
            completed = len(generated)
            write_json(root / 'progress.json', {'new_completed': completed, 'new_population': len(rows), 'local_failures': failures,
                'pending_jobs': len(pending), 'canonical_cache_changed': False, 'task_accuracy': None})
            if completed and (completed % 16 == 0 or not pending):
                print(json.dumps({'teacher_new_completed': completed, 'new_population': len(rows)}), flush=True)
    if failures:
        raise RuntimeError('bulk local failure preserved; pending jobs drained and no further sources submitted')
    return generated


def generate(pilot=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('gateway teacher needs no local GPU')
    recipe.configure()
    native.vendor.corpus.quality.dependencies()
    native.vendor.terminal(PREP)
    all_rows = q.read(PREP / 'cases.json')
    retained = {r['id']: r for r in q.read(PREP / 'retained64.json')}
    inputs = [PREP / n for n in ('cases.json', 'retained64.json', 'pilot_ids.json', 'public_proof.json', 'worker_terminal_receipt.json')]
    ids = q.read(PREP / 'pilot_ids.json')
    if not pilot:
        native.vendor.terminal(PILOT)
        if q.read(PILOT / 'result.json')['status'] != 'eight_case_teacher_concurrency_systems_passed':
            raise ValueError('actual complete8case transport/capacity pilot required')
        pilot_values = {r['id']: r for r in q.read(PILOT / 'labels.json')}
        if set(pilot_values) & set(retained):
            raise ValueError('pilot must not regenerate any qualified64 first')
        retained.update(pilot_values)
        inputs += [PILOT / n for n in ('labels.json', 'result.json', 'worker_terminal_receipt.json')]
    rows = [r for r in all_rows if r['id'] in ids] if pilot else [r for r in all_rows if r['id'] not in retained]
    root = PILOT if pilot else GENERATE
    p = producing(root, 'eight_case_transport_concurrency_pilot' if pilot else 'all10000_first_teacher_candidates', inputs)
    (root / 'cases').mkdir()
    started = time.monotonic()
    generated = candidate_jobs(rows, root, data.storage_encoding(), native.tokenizer())
    labels = combine(rows, {}, generated) if pilot else combine(all_rows, retained, generated)
    write_json(root / 'labels.json', labels)
    valid, capacity_fit = sum(r['teacher_valid'] for r in labels), sum(r['capacity_fit'] for r in labels)
    status = ('eight_case_teacher_concurrency_systems_passed' if valid == capacity_fit == 8 else 'eight_case_teacher_concurrency_systems_failed') if pilot else 'all10000_first_teacher_candidates_completed_student_quality_pending'
    native.vendor.verify(p)
    result = {'status': status, 'cases': len(labels), 'valid': valid, 'capacity_fit': capacity_fit,
        'retained_firsts': 0 if pilot else len(retained), 'new_candidates': len(generated), 'workers': WORKERS,
        'elapsed_seconds': time.monotonic() - started, 'labels_sha256': file_hash(root / 'labels.json'),
        'teacher_recipe_qualified': True, 'individual_label_semantic_quality_not_assumed': True,
        'canonical_cache_changed': False, 'student_trained': False, 'task_accuracy': None, 'speed_tested': False}
    write_json(root / 'result.json', result)
    print(json.dumps(result), flush=True)
    if pilot and (valid != 8 or capacity_fit != 8):
        raise ValueError('bulk transport systems failure preserved; no full generation admission')


def workflow():
    require_teacher()
    if any(p.exists() for p in (WORKFLOW, PREP, PILOT, GENERATE)):
        raise ValueError('fresh append-only bulk teacher roots required')
    p = producing(WORKFLOW, 'supervise_qualified_teacher10000_TRAIN_expansion')
    receipts = []
    for stage, root in (('prepare', PREP), ('pilot', PILOT), ('generate', GENERATE)):
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', MODULE, stage]
        child = subprocess.Popen(command)
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'exit_code': code, 'command': command, 'child_pid': child.pid, 'observer_pid': os.getpid(), 'observer': 'subprocess.wait'}
        write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('bulk teacher stage failed:' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'bulk_teacher_raw_workflow_completed_independent_audit_pending', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'generate', 'workflow'))
    args = parser.parse_args()
    {'prepare': prepare, 'pilot': lambda: generate(True), 'generate': generate, 'workflow': workflow}[args.stage]()
