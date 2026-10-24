"""Isolate tokenizer/rendering semantics on all64 new fixed teacher summaries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_current_judgment_lineage as lineage
from experiments.structure_net import ncm_sol_teacher_semantics as teacher
from experiments.structure_net import ncm_lm1b_native_development as development
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = teacher.native, teacher.q, teacher.ROOT
reference, oracle = teacher.reference, teacher.oracle
HYPOTHESIS = 'ncm-sol64-native-text-roundtrip-semantic-v1'
MODULE = 'experiments.structure_net.ncm_sol64_bert_roundtrip'
AUDITOR_MODULE = 'experiments.neural_architecture_lab.finalize_ncm_sol64_bert_roundtrip'
AUDITOR = Path('experiments/neural_architecture_lab/finalize_ncm_sol64_bert_roundtrip.py')
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sol64-bert-roundtrip-plan.md')
TEST = Path('tests/structure_net/test_ncm_sol64_bert_roundtrip.py')
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-sol64-bert-roundtrip.md')
WORKFLOW, PREP, PILOT, GRADE, RESULTS = (ROOT / ('20261024_sol64_bert_roundtrip_' + n + '_v1')
    for n in ('workflow', 'prepare', 'pilot', 'grade', 'results'))


def configure():
    lineage.configure()
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('tokenizer and semantic component study must remain CPU-only')


def render(row, tokenizer):
    ids = tokenizer.encode(row['target_text'], add_special_tokens=False)
    if (not ids or len(ids) > 126 or tokenizer.unk_token_id in ids
            or row['native_word_ids'] != [101, *ids, 101] + [0] * (126 - len(ids))):
        raise ValueError('complete unchanged first target and native CLS frame required')
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def split_guard(rows):
    original = q.read(native.CASES)['train']
    dev = q.read(development.SOURCE / 'cases.json')
    other = [*original, *[r for cohort in development.COHORTS for r in dev[cohort]]]
    proof = {}
    if len(rows) != 64:
        raise ValueError('all64 first sources required')
    for field in ('id', 'source_id', 'group_id', 'source_sha256', 'source_component'):
        a, b = {r[field] for r in rows}, {r[field] for r in other if field in r}
        if len(a) != 64 or a & b:
            raise ValueError('new64 must be disjoint from original training and DEV: ' + field)
        proof[field] = {'new64_unique': len(a), 'earlier_unique': len(b), 'overlap': 0}
    return proof


def expected_cases():
    rows = teacher.teacher.q.read(teacher.teacher.GENERATE / 'labels.json')
    ordered = teacher.require_raw()
    by_id = {r['id']: r for r in rows}
    tokenizer = native.tokenizer()
    return [{**r, 'texts': {'labels': render(by_id[r['id']], tokenizer)}} for r in ordered]


def producing(root, stage, inputs=()):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(lineage.__file__), Path(lineage.historical.__file__),
        Path(teacher.__file__), Path(teacher.teacher.__file__), Path(oracle.__file__),
        Path(reference.first.__file__), Path(reference.profile.__file__), reference.profile.PLAN,
        Path(native.__file__)]
    tokenizer_files = [p for p in sorted((native.INVENTORY / 'bert-tokenizer').iterdir()) if p.is_file()]
    common = [*lineage.inputs(), teacher.teacher.GENERATE / 'labels.json', teacher.PREP / 'cases.json',
        teacher.PREP / 'packets.json', teacher.GRADE / 'secondary_labels_case_scores.json',
        teacher.GRADE / 'primary_labels_case_scores.json', native.CASES, development.SOURCE / 'cases.json',
        *tokenizer_files]
    p = native.vendor.freeze(root, stage, list(dict.fromkeys([*inputs, *common, *files])), {
        'hypothesis_id': HYPOTHESIS, 'CPU_only': True, 'model_inference': False, 'student_trained': False,
        'component_semantics_only': True, 'representation': 'BERT native lexical tokenization/decode',
        'cases': 64, 'canonical_prior_firsts': lineage.COUNT, 'judges': q.JUDGES, 'rubric': q.RUBRIC,
        'joint_minimum': 61, 'all_valid_required': True, 'max_attempts': 3, 'workers': 4,
        'no_semantic_retries': True, 'source_prediction': False, 'oracle_target_text': True,
        'student_task_accuracy': None, 'final_test_opened': False, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'roundtrip_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def prepare():
    configure()
    lineage.base_cache()
    cases = expected_cases()
    proof = split_guard(cases)
    public = reference.verify_public(cases)
    packets = {'labels': [q.packet(r['id'], r['source'], [{'candidate_id': 'summary', 'text': r['texts']['labels']}]) for r in cases]}
    p = producing(PREP, 'all64_native_text_rendering_with_exact_sources', [reference.PUBLIC_FILE])
    write_json(PREP / 'cases.json', cases)
    write_json(PREP / 'packets.json', packets)
    write_json(PREP / 'split_proof.json', proof)
    write_json(PREP / 'public_proof.json', {'public_file_sha256': reference.PUBLIC_SHA256, 'exact_sources': public})
    native.vendor.verify(p)
    write_json(PREP / 'result.json', {'status': 'all64_BERT_roundtrip_packets_prepared', 'cases': 64,
        'cases_sha256': file_hash(PREP / 'cases.json'), 'packets_sha256': file_hash(PREP / 'packets.json'),
        'student_task_accuracy': None})


def metric(scores):
    value = oracle.metric(scores)
    value['representation_semantic_pass_rate'] = value.pop('oracle_semantic_pass_rate')
    value.update(student_task_accuracy=None, source_prediction=False, oracle_target_text=True)
    return value


def grade(pilot=False):
    configure()
    lineage.require_judges()
    native.vendor.terminal(PREP)
    cases, packets = q.read(PREP / 'cases.json'), q.read(PREP / 'packets.json')
    cache = lineage.base_cache()
    inputs = [PREP / n for n in ('cases.json', 'packets.json', 'split_proof.json', 'worker_terminal_receipt.json')]
    if not pilot:
        native.vendor.terminal(PILOT)
        prior = q.read(PILOT / 'judgment_cache.json')
        if (any(prior.get(k) != v for k, v in cache.items())
                or q.read(PILOT / 'result.json')['status'] != 'four_BERT_roundtrip_judgments_valid_only'):
            raise ValueError('actual valid pilot and all preceding firsts required')
        cache = prior
        inputs += [PILOT / n for n in ('result.json', 'judgment_cache.json', 'worker_terminal_receipt.json')]
    preceding = dict(cache)
    root = PILOT if pilot else GRADE
    p = producing(root, 'four_case_validity_only' if pilot else 'complete64_BERT_roundtrip_semantics', inputs)
    write_json(root / 'judgment_cache.json', cache)
    metrics = {}

    def judge(role):
        oracle.grade_arm(root, 'labels', role, cases[:4] if pilot else cases, packets, cache)
        scores = q.read(root / f'{role}_labels_case_scores.json')
        metrics[role] = {**metric(scores), 'judge': q.JUDGES[role]}
        write_json(root / f'{role}_labels_result.json', metrics[role])
        write_json(root / 'judgment_cache.json', cache)
        return scores

    secondary = judge('secondary')
    admitted = not pilot and oracle.qualifies(metrics['secondary'])
    joint = teacher.joint_gate(secondary, judge('primary')) if admitted else None
    qualified = admitted and joint['qualified']
    if any(cache.get(k) != v for k, v in preceding.items()):
        raise ValueError('a preceding first judgment changed')
    lineage.audit_cache(cache)
    valid = all(r['valid'] for r in secondary)
    status = ('four_BERT_roundtrip_judgments_valid_only' if valid else 'BERT_roundtrip_judge_pilot_invalid') if pilot else (
        'fixed64_BERT_roundtrip_semantically_qualified' if qualified else 'fixed64_BERT_roundtrip_quality_failed')
    native.vendor.verify(p)
    result = {'status': status, 'metrics': metrics, 'joint': joint, 'primary_admitted': admitted,
        'representation_qualified': bool(qualified), 'canonical_first_judgments': len(cache),
        'inherited_first_judgments': len(preceding), 'cases': 4 if pilot else 64,
        'student_task_accuracy': None, 'student_trained': False, 'final_test_opened': False, 'speed_tested': False}
    write_json(root / 'result.json', result)
    print(json.dumps(result), flush=True)
    if pilot and not valid:
        raise ValueError('validity-only pilot failed; firsts preserved')


def workflow():
    configure()
    lineage.require_judges()
    lineage.base_cache()
    expected_cases()
    if any(root.exists() for root in (WORKFLOW, PREP, PILOT, GRADE, RESULTS)) or REPORT.exists():
        raise ValueError('fresh BERT component workflow required')
    p = producing(WORKFLOW, 'supervise64_native_text_roundtrip_semantics')
    receipts = []
    for module, stage, root in ((MODULE, 'prepare', PREP), (MODULE, 'pilot', PILOT), (MODULE, 'grade', GRADE),
            (AUDITOR_MODULE, 'finalize', RESULTS), (AUDITOR_MODULE, 'register', None)):
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
            raise RuntimeError('BERT roundtrip stage failed: ' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'BERT_roundtrip_semantic_workflow_completed', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'grade', 'workflow'))
    args = parser.parse_args()
    {'prepare': prepare, 'pilot': lambda: grade(True), 'grade': grade, 'workflow': workflow}[args.stage]()
