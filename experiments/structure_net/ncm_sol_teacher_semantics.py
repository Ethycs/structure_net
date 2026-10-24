"""Measure all fresh Sol-reviewed labels without confusing them with student outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import ncm_sol_evidence_teacher_recovery as teacher
from experiments.structure_net import ncm_evidence_teacher_qualification as inherited
from experiments.structure_net.ncm_memory_data import file_hash, write_json

lineage, native, q, ROOT = teacher.lineage, teacher.native, teacher.q, teacher.ROOT
oracle, reference = inherited.oracle, inherited.reference
HYPOTHESIS = 'ncm-sol-evidence64-teacher-semantic-qualification-v1'
MODULE = 'experiments.structure_net.ncm_sol_teacher_semantics'
AUDITOR_MODULE = 'experiments.neural_architecture_lab.finalize_ncm_sol_teacher_semantics'
AUDITOR = Path('experiments/neural_architecture_lab/finalize_ncm_sol_teacher_semantics.py')
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sol-teacher64-semantic-plan.md')
TEST = Path('tests/structure_net/test_ncm_sol_teacher_semantics.py')
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-sol-teacher64-semantic-qualification.md')
WORKFLOW = ROOT / '20261024_sol_teacher64_semantic_workflow_v1'
PREP = ROOT / '20261024_sol_teacher64_semantic_prepare_v1'
PILOT = ROOT / '20261024_sol_teacher64_semantic_pilot_v1'
GRADE = ROOT / '20261024_sol_teacher64_semantic_grade_v1'
RESULTS = ROOT / '20261024_sol_teacher64_semantic_results_v1'


def configure():
    teacher.configure()


def joint_gate(secondary, primary):
    if (len(secondary) != 64 or len(primary) != 64
            or len({r['id'] for r in secondary}) != 64 or len({r['id'] for r in primary}) != 64
            or {r['id'] for r in secondary} != {r['id'] for r in primary}):
        raise ValueError('both judges must score same complete64 cases')
    other = {r['id']: r for r in primary}
    valid = all(r['valid'] and other[r['id']]['valid'] for r in secondary)
    count = sum(r['passed'] is True and other[r['id']]['passed'] is True for r in secondary)
    return {'cases': 64, 'joint_passed': count, 'all_valid': valid, 'qualified': valid and count >= 61}


def require_raw():
    if q.read(teacher.AUDIT / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual fresh64 raw replay completion required')
    audit = q.read(teacher.AUDIT / 'result.json')
    if (audit['status'] != 'fresh64_sol_teacher_firsts_replayed_quality_pending'
            or audit['cases'] != 64 or audit['teacher_valid'] != 64 or audit['capacity_fit'] != 64):
        raise ValueError('all64 first candidates must be schema-valid and capacity-fit; no rerolls')
    for path, expected in audit['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('raw first teacher artifact changed: ' + path)
    for root in (teacher.WORKFLOW, teacher.PREP, teacher.PILOT, teacher.GENERATE):
        native.vendor.verify(q.read(root / 'producer.json'))
    return inherited.ordered_cases(q.read(teacher.GENERATE / 'labels.json'))


def producing(root, stage, inputs=()):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(teacher.__file__), teacher.PLAN, Path(lineage.__file__),
        Path(inherited.__file__), Path(oracle.__file__), Path(reference.first.__file__),
        Path(reference.profile.__file__), reference.profile.PLAN, lineage.qualification.PLAN]
    common = [lineage.CACHE, lineage.CAMPAIGN, teacher.GENERATE / 'labels.json',
        teacher.AUDIT / 'result.json', teacher.AUDIT / 'actual_tool_receipt.json', *lineage.judge_inputs()]
    p = native.vendor.freeze(root, stage, [*inputs, *common, *files], {'hypothesis_id': HYPOTHESIS,
        'teacher_label_quality_only': True, 'student_task_accuracy': None, 'student_trained': False,
        'models': teacher.MODELS, 'judges': q.JUDGES, 'rubric': q.RUBRIC,
        'canonical_prior_firsts': 3222, 'all64_cases': True, 'workers': 4, 'max_attempts': 3,
        'joint_minimum': 61, 'all_valid_required': True, 'no_semantic_retries': True,
        'final_test_opened': False, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'sol_semantic_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def prepare():
    configure()
    lineage.base_cache()
    cases = require_raw()
    proof = reference.verify_public(cases)
    packets = {'labels': [q.packet(r['id'], r['source'], [{'candidate_id': 'summary', 'text': r['texts']['labels']}]) for r in cases]}
    p = producing(PREP, 'fixed64_exact_public_TRAIN_label_packets', [reference.PUBLIC_FILE])
    write_json(PREP / 'cases.json', cases)
    write_json(PREP / 'packets.json', packets)
    write_json(PREP / 'public_proof.json', {'public_file_sha256': reference.PUBLIC_SHA256, 'exact_sources': proof})
    native.vendor.verify(p)
    write_json(PREP / 'result.json', {'status': 'fixed64_sol_teacher_packets_prepared', 'cases': 64,
        'cases_sha256': file_hash(PREP / 'cases.json'), 'packets_sha256': file_hash(PREP / 'packets.json'), 'student_task_accuracy': None})


def grade(pilot=False):
    configure()
    native.vendor.terminal(PREP)
    cases, packets = q.read(PREP / 'cases.json'), q.read(PREP / 'packets.json')
    cache = lineage.base_cache()
    inputs = [PREP / n for n in ('cases.json', 'packets.json', 'public_proof.json', 'worker_terminal_receipt.json')]
    if not pilot:
        native.vendor.terminal(PILOT)
        prior = q.read(PILOT / 'judgment_cache.json')
        if (any(prior.get(k) != v for k, v in cache.items())
                or q.read(PILOT / 'result.json')['status'] != 'four_sol_teacher_judgments_valid_only'):
            raise ValueError('actual four-case valid pilot and all prior firsts required')
        cache = prior
        inputs += [PILOT / n for n in ('judgment_cache.json', 'result.json', 'worker_terminal_receipt.json')]
    inherited_cache = dict(cache)
    root = PILOT if pilot else GRADE
    p = producing(root, 'judge_path_validity_only' if pilot else 'fixed64_teacher_label_semantic_quality', inputs)
    write_json(root / 'judgment_cache.json', cache)
    metrics = {}

    def judge(role):
        oracle.grade_arm(root, 'labels', role, cases[:4] if pilot else cases, packets, cache)
        scores = q.read(root / f'{role}_labels_case_scores.json')
        metrics[role] = {**inherited.metric(scores), 'judge': q.JUDGES[role]}
        write_json(root / f'{role}_labels_result.json', metrics[role])
        write_json(root / 'judgment_cache.json', cache)
        return scores

    secondary = judge('secondary')
    admitted = not pilot and oracle.qualifies(metrics['secondary'])
    joint = joint_gate(secondary, judge('primary')) if admitted else None
    if any(cache.get(k) != v for k, v in inherited_cache.items()):
        raise ValueError('a preceding first judgment changed')
    lineage.audit_cache(cache)
    valid_pilot = all(r['valid'] for r in secondary)
    qualified = admitted and joint['qualified']
    status = ('four_sol_teacher_judgments_valid_only' if valid_pilot else 'sol_teacher_judge_pilot_invalid') if pilot else (
        'fixed64_sol_teacher_labels_semantically_qualified' if qualified else 'fixed64_sol_teacher_label_quality_failed')
    native.vendor.verify(p)
    result = {'status': status, 'metrics': metrics, 'joint': joint, 'primary_admitted': admitted,
        'teacher_labels_qualified': bool(qualified), 'cases': 4 if pilot else 64,
        'canonical_first_judgments': len(cache), 'inherited_first_judgments': len(inherited_cache),
        'student_task_accuracy': None, 'student_trained': False, 'final_test_opened': False, 'speed_tested': False}
    write_json(root / 'result.json', result)
    print(json.dumps(result), flush=True)
    if pilot and not valid_pilot:
        raise ValueError('teacher judge validity pilot failed; firsts preserved')


def workflow():
    configure()
    lineage.base_cache()
    require_raw()
    if any(p.exists() for p in (WORKFLOW, PREP, PILOT, GRADE, RESULTS, REPORT)):
        raise ValueError('fresh Sol semantic qualification roots required')
    p = producing(WORKFLOW, 'supervise_fixed64_sol_teacher_semantic_qualification')
    receipts = []
    stages = ((MODULE, 'prepare', PREP), (MODULE, 'pilot', PILOT), (MODULE, 'grade', GRADE),
        (AUDITOR_MODULE, 'finalize', RESULTS), (AUDITOR_MODULE, 'register', None))
    for module, stage, root in stages:
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', module, stage]
        child = subprocess.Popen(command)
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'command': command, 'child_pid': child.pid,
            'observer_pid': os.getpid(), 'observer': 'subprocess.wait', 'exit_code': code}
        write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root is not None and root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('Sol teacher semantic stage failed: ' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'sol_teacher_semantic_qualification_workflow_completed', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'grade', 'workflow'))
    args = parser.parse_args()
    {'prepare': prepare, 'pilot': lambda: grade(True), 'grade': grade, 'workflow': workflow}[args.stage]()
