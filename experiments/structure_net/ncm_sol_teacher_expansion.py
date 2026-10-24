"""Collect only the remaining8819 first teacher candidates, with bounded submission."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import ncm_sol_evidence_teacher_recovery as teacher
from experiments.structure_net import ncm_sol_teacher_semantics as qualification
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT, data = teacher.native, teacher.q, teacher.ROOT, teacher.data
original = teacher.base.original
MODULE = 'experiments.structure_net.ncm_sol_teacher_expansion'
HYPOTHESIS = 'ncm-sol-teacher-remaining8819-first-candidates-v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sol-teacher-expansion-plan.md')
TEST = Path('tests/structure_net/test_ncm_sol_teacher_expansion.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_sol_teacher_expansion.py')
RESERVATION = ROOT / '20261024_remaining8819_teacher_reservation_v1'
RESERVATION_SHA = '7e47a1615aa826db18fcadeaece25638561d44d78e1872742e41517f0b971028'
QUALIFICATION_SHA = '2d6ee5f2dbca777bfb0774569f2bd7e69ba426245dd0247f80a04d62f80d1b4b'
WORKFLOW, PREP, PILOT, GENERATE = (ROOT / ('20261024_sol_teacher8819_' + n + '_v1')
    for n in ('workflow', 'prepare', 'pilot', 'candidates'))
WORKERS = 8


def configure():
    teacher.configure()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('gateway teacher remains CPU-only')


def sources():
    reservation = q.read(RESERVATION / 'reservation.json')
    if (file_hash(RESERVATION / 'reservation.json') != RESERVATION_SHA
            or q.read(RESERVATION / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('actual unchanged8819-source reservation required')
    for entry in reservation['source_inputs'].values():
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('original reservation input changed')
    rows = q.read(data.DEST / 'cases.json')
    old = q.read(teacher.base.INVENTORY / 'first_candidates.json')
    fresh = q.read(teacher.GENERATE / 'labels.json')
    blocked = {r['id'] for r in (*old, *fresh)}
    result = [r for r in rows if r['id'] not in blocked]
    identities = [{k: r[k] for k in ('id', 'source_id', 'group_id', 'source_component', 'source_sha256', 'bin')}
        for r in result]
    if (len(old) != 1117 or len(fresh) != 64 or len(blocked) != 1181 or len(result) != 8819
            or identities != reservation['reserved']
            or len({r['source_component'] for r in result}) != 8819
            or {r['source_component'] for r in result} & {r['source_component'] for r in (*old, *fresh)}):
        raise ValueError('unchanged original order minus1117+64 firsts required')
    return result


def require_teacher():
    path = qualification.RESULTS / 'campaign_results.json'
    state = q.read(path)
    receipt = q.read(qualification.RESULTS / (qualification.HYPOTHESIS + '_registry_readback.json'))
    if (file_hash(path) != QUALIFICATION_SHA
            or state['classification'] != 'fixed64_sol_teacher_labels_semantically_qualified'
            or state['joint'] != {'cases': 64, 'joint_passed': 62, 'all_valid': True, 'qualified': True}
            or receipt['record']['campaign_sha256'] != QUALIFICATION_SHA
            or receipt['exact']['campaign_sha256'] != QUALIFICATION_SHA
            or q.read(qualification.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or q.read(qualification.RESULTS / 'backend_actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or q.read(qualification.RESULTS / 'backend_exact_readback.json')['status'] != 'independent_backend_exact_readback_passed'):
        raise ValueError('actual jointly-qualified62of64 unchanged teacher recipe required')
    qualification.require_raw()
    previous = q.read(teacher.GENERATE / 'producer.json')['config']
    if previous['models'] != teacher.MODELS or previous['systems'] != original.SYSTEMS:
        raise ValueError('same qualified model identities and prompts required')


def pilot_ids(rows):
    longest = min(rows, key=lambda r: (-r['source_cl100k_tokens'], r['id']))
    chosen = [rows[0]['id'], longest['id']]
    chosen += [r['id'] for r in rows if r['id'] not in chosen][:6]
    if len(chosen) != len(set(chosen)) or len(chosen) != 8:
        raise ValueError('eight distinct first/longest original-order pilot sources required')
    return chosen


def producing(root, stage, inputs=()):
    modules = [teacher, teacher.base, teacher.base.recipe, teacher.base.recipe.recovery, original,
        qualification, data]
    files = [Path(__file__), PLAN, TEST, AUDITOR, *[Path(m.__file__) for m in modules]]
    common = [RESERVATION / n for n in ('reservation.json', 'prepare.py', 'actual_tool_receipt.json')]
    common += [qualification.RESULTS / 'campaign_results.json', qualification.RESULTS / (qualification.HYPOTHESIS + '_registry_readback.json'),
        qualification.RESULTS / 'backend_exact_readback.json', qualification.RESULTS / 'backend_actual_tool_receipt.json',
        qualification.WORKFLOW / 'actual_tool_receipt.json', teacher.GENERATE / 'labels.json',
        teacher.AUDIT / 'result.json', teacher.AUDIT / 'actual_tool_receipt.json',
        teacher.base.INVENTORY / 'first_candidates.json', data.DEST / 'cases.json']
    p = native.vendor.freeze(root, stage, list(dict.fromkeys([*inputs, *common, *files])), {
        'hypothesis_id': HYPOTHESIS, 'new_source_population': 8819, 'old_firsts_preserved': 1181,
        'old_invalid_firsts_not_retried': 63, 'models': teacher.MODELS, 'systems': original.SYSTEMS,
        'workers': WORKERS, 'maximum_pending': WORKERS, 'max_attempts_per_stage': 3,
        'no_semantic_or_capacity_retries': True, 'stop_new_submissions_on_exhausted_or_local_failure': True,
        'drain_inflight_and_preserve_partial_firsts': True, 'individual_semantic_quality_not_assumed': True,
        'draft_requested_max_tokens': 768, 'review_requested_max_tokens': 512,
        'prompt_token_limit': 6300, 'student_training_admitted': False,
        'student_task_accuracy': None, 'canonical_semantic_cache_changed': False,
        'component64_excluded_from_future_gradients': True, 'final_test_opened': False, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'expansion_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def prepare():
    configure()
    require_teacher()
    rows = sources()
    reference = qualification.reference
    proof = reference.verify_public(rows)
    p = producing(PREP, 'unchanged_original8819_firsts_only', [reference.PUBLIC_FILE])
    write_json(PREP / 'cases.json', rows)
    write_json(PREP / 'pilot_ids.json', pilot_ids(rows))
    write_json(PREP / 'public_proof.json', {'public_file_sha256': reference.PUBLIC_SHA256, 'exact_sources': proof})
    native.vendor.verify(p)
    write_json(PREP / 'result.json', {'status': 'exact8819_unattempted_sources_prepared', 'cases': 8819,
        'student_training_admitted': False, 'canonical_semantic_cache_changed': False})


def candidate_jobs(rows, root, encoding, tokenizer, candidate=None):
    """No more than8 submissions in flight; invalid firsts stop replacement jobs."""
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('unique first-candidate IDs required before submission')
    candidate = original.candidate if candidate is None else candidate
    iterator, generated, failures, stopped = iter(rows), {}, [], False
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = {}

        def submit():
            row = next(iterator, None)
            if row is not None:
                pending[pool.submit(candidate, row, root, encoding, tokenizer)] = row

        for _ in range(WORKERS):
            submit()
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                row = pending.pop(future)
                try:
                    value = future.result()
                    if value['id'] != row['id']:
                        raise ValueError('candidate source identity changed')
                    write_json(root / 'cases' / (row['id'] + '.json'), value)
                    generated[row['id']] = value
                    if not value['teacher_valid']:
                        stopped = True
                        failures.append({'id': row['id'], 'kind': 'exhausted_first_pipeline'})
                except Exception as error:
                    stopped = True
                    failures.append({'id': row['id'], 'kind': 'local_failure',
                        'error_type': type(error).__name__, 'error': str(error)})
            if not stopped:
                for _ in ready:
                    submit()
            write_json(root / 'progress.json', {'completed_firsts': len(generated), 'population': len(rows),
                'pending_jobs': len(pending), 'submissions_stopped': stopped, 'failures': failures})
            if len(generated) % 16 == 0 or not pending:
                print(json.dumps({'teacher_firsts_completed': len(generated), 'population': len(rows),
                    'submissions_stopped': stopped}), flush=True)
    ordered = [generated[r['id']] for r in rows if r['id'] in generated]
    write_json(root / 'partial_firsts.json', ordered)
    write_json(root / 'local_or_exhausted_failures.json', failures)
    if stopped:
        raise RuntimeError('first failure preserved; inflight jobs drained; no further source submitted')
    return generated


def generate(pilot=False):
    configure()
    require_teacher()
    native.vendor.terminal(PREP)
    rows = q.read(PREP / 'cases.json')
    ids = q.read(PREP / 'pilot_ids.json')
    inputs = [PREP / n for n in ('cases.json', 'pilot_ids.json', 'public_proof.json', 'worker_terminal_receipt.json')]
    prior = {}
    if not pilot:
        native.vendor.terminal(PILOT)
        if q.read(PILOT / 'result.json')['status'] != 'eight_first_candidate_systems_passed':
            raise ValueError('complete eight-case first-pipeline pilot required')
        prior = {r['id']: r for r in q.read(PILOT / 'labels.json')}
        if set(prior) != set(ids):
            raise ValueError('every pilot first must be retained unchanged')
        inputs += [PILOT / n for n in ('labels.json', 'result.json', 'worker_terminal_receipt.json')]
    work = [r for r in rows if r['id'] in ids] if pilot else [r for r in rows if r['id'] not in prior]
    root = PILOT if pilot else GENERATE
    p = producing(root, 'eight_case_bounded_submission_pilot' if pilot else 'all8819_preserved_first_candidates', inputs)
    (root / 'cases').mkdir()
    generated = candidate_jobs(work, root, data.storage_encoding(), native.tokenizer())
    merged = {**prior, **generated}
    labels = [merged[r['id']] for r in (work if pilot else rows)]
    write_json(root / 'labels.json', labels)
    valid, capacity = sum(r['teacher_valid'] for r in labels), sum(r['capacity_fit'] for r in labels)
    native.vendor.verify(p)
    status = ('eight_first_candidate_systems_passed' if valid == capacity == 8 else 'eight_first_candidate_systems_failed') if pilot else 'all8819_first_candidates_collected_independent_audit_pending'
    result = {'status': status, 'cases': len(labels), 'valid': valid, 'capacity_fit': capacity,
        'labels_sha256': file_hash(root / 'labels.json'), 'retained_pilot_firsts': len(prior),
        'new_candidates': len(generated), 'student_training_admitted': False,
        'canonical_semantic_cache_changed': False, 'student_task_accuracy': None}
    write_json(root / 'result.json', result)
    print(json.dumps(result), flush=True)
    if pilot and (valid != 8 or capacity != 8):
        raise ValueError('pilot capacity failure preserved without rerolls')


def workflow():
    configure()
    require_teacher()
    if any(p.exists() for p in (WORKFLOW, PREP, PILOT, GENERATE)):
        raise ValueError('fresh append-only8819 roots required')
    p = producing(WORKFLOW, 'supervise_remaining8819_firsts')
    receipts = []
    for stage, root in (('prepare', PREP), ('pilot', PILOT), ('generate', GENERATE)):
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', MODULE, stage]
        child = subprocess.Popen(command)
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'exit_code': code, 'command': command, 'child_pid': child.pid,
            'observer_pid': os.getpid(), 'observer': 'subprocess.wait'}
        write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('remaining8819 stage failed: ' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'remaining8819_raw_collection_completed_audit_pending', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'generate', 'workflow'))
    args = parser.parse_args()
    {'prepare': prepare, 'pilot': lambda: generate(True), 'generate': generate, 'workflow': workflow}[args.stage]()
