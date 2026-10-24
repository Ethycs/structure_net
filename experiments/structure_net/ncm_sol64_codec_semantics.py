"""Whole-summary semantic controls for all fixed released/domain reconstructions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import ncm_sol64_codec_component as numeric
from experiments.structure_net import ncm_after_bert_judgment_lineage as lineage
from experiments.structure_net import ncm_sol64_bert_roundtrip_recovery as rendering
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = numeric.native, numeric.q, numeric.ROOT
reference, oracle = numeric.text_control.reference, numeric.text_control.oracle
HYPOTHESIS = 'ncm-sol64-frozen-codec-semantic-v1'
MODULE = 'experiments.structure_net.ncm_sol64_codec_semantics'
AUDITOR_MODULE = 'experiments.neural_architecture_lab.finalize_ncm_sol64_codec_semantics'
AUDITOR = Path('experiments/neural_architecture_lab/finalize_ncm_sol64_codec_semantics.py')
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sol64-codec-semantics-plan.md')
TEST = Path('tests/structure_net/test_ncm_sol64_codec_semantics.py')
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-sol64-codec-semantics.md')
WORKFLOW, PREP, PILOT, GRADE, RESULTS = (ROOT / ('20261024_sol64_codec_semantics_' + n + '_v1')
    for n in ('workflow', 'prepare', 'pilot', 'grade', 'results'))
NUMERIC_AUDIT = ROOT / '20261024_sol64_codec_component_audit_v1'
ARMS = ('released', 'domain')


def configure():
    numeric.configure()
    lineage.configure()
    if numeric.torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('semantic control must remain CPU-only')


def require_numeric():
    if (q.read(numeric.WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or q.read(NUMERIC_AUDIT / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('actual full numeric inference and independent CPU audit required')
    audit = q.read(NUMERIC_AUDIT / 'result.json')
    if audit['status'] != 'all64_fixed_codec_output_bindings_and_rendering_replayed_semantics_pending' or audit['cases'] != 64:
        raise ValueError('complete64 audited output bindings required')
    for path, expected in audit['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('audited fixed codec evidence changed: ' + path)
    for root in (numeric.PREP, numeric.PILOT, numeric.OUTPUTS):
        native.vendor.terminal(root)
    rows, outputs = numeric.cases(), q.read(numeric.OUTPUTS / 'outputs.json')
    if [r['id'] for r in rows] != [r['id'] for r in outputs]:
        raise ValueError('all64 ordered fixed first reconstructions required')
    return [{**{k: v for k, v in row.items() if k not in ('texts', 'native_word_ids')},
        'texts': {arm: output['conditions'][arm]['text'] for arm in ARMS}} for row, output in zip(rows, outputs)]


def producing(root, stage, inputs=()):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(numeric.__file__), numeric.PLAN,
        Path(lineage.__file__), Path(lineage.preceding.__file__),
        Path(oracle.__file__), Path(reference.first.__file__), Path(reference.profile.__file__), reference.profile.PLAN]
    common = [*lineage.inputs(), NUMERIC_AUDIT / 'result.json', NUMERIC_AUDIT / 'actual_tool_receipt.json',
        numeric.WORKFLOW / 'actual_tool_receipt.json', numeric.OUTPUTS / 'outputs.json',
        numeric.OUTPUTS / 'result.json', numeric.OUTPUTS / 'worker_terminal_receipt.json',
        rendering.teacher.PREP / 'cases.json', rendering.teacher.PREP / 'packets.json',
        rendering.PREP / 'cases.json', rendering.PREP / 'packets.json']
    p = native.vendor.freeze(root, stage, list(dict.fromkeys([*inputs, *common, *files])), {
        'hypothesis_id': HYPOTHESIS, 'cases': 64, 'arms': ARMS, 'oracle_target_codes': True,
        'source_prediction': False, 'student_task_accuracy': None, 'student_trained': False,
        'canonical_prior_firsts': lineage.COUNT, 'rubric': q.RUBRIC, 'judges': q.JUDGES,
        'same_case_joint_minimum': 61, 'max_attempts': 3, 'workers': 4,
        'no_semantic_retries': True, 'final_test_opened': False, 'speed_tested': False})
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'codec_semantics_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def prepare():
    configure()
    lineage.base_cache()
    cases = require_numeric()
    packets = {arm: [q.packet(r['id'], r['source'], [{'candidate_id': 'summary', 'text': r['texts'][arm]}]) for r in cases] for arm in ARMS}
    proof = reference.verify_public(cases)
    p = producing(PREP, 'exact64_source_and_oracle_codec_packets', [reference.PUBLIC_FILE])
    write_json(PREP / 'cases.json', cases)
    write_json(PREP / 'packets.json', packets)
    write_json(PREP / 'public_proof.json', {'public_file_sha256': reference.PUBLIC_SHA256, 'exact_sources': proof})
    native.vendor.verify(p)
    write_json(PREP / 'result.json', {'status': 'exact64_codec_semantic_packets_prepared', 'cases': 64,
        'cases_sha256': file_hash(PREP / 'cases.json'), 'packets_sha256': file_hash(PREP / 'packets.json'), 'student_task_accuracy': None})


def metric(scores):
    result = oracle.metric(scores)
    result['representation_semantic_pass_rate'] = result.pop('oracle_semantic_pass_rate')
    result.update(student_task_accuracy=None, source_prediction=False, oracle_target_codes=True)
    return result


def grade(pilot=False):
    configure()
    lineage.require_judges()
    native.vendor.terminal(PREP)
    cases, packets = q.read(PREP / 'cases.json'), q.read(PREP / 'packets.json')
    cache = lineage.base_cache()
    inputs = [PREP / n for n in ('cases.json', 'packets.json', 'worker_terminal_receipt.json')]
    if not pilot:
        native.vendor.terminal(PILOT)
        previous = q.read(PILOT / 'judgment_cache.json')
        if any(previous.get(k) != v for k, v in cache.items()) or q.read(PILOT / 'result.json')['status'] != 'all_codec_pilot_judgments_valid_only':
            raise ValueError('actual valid eight-case pilot with prior firsts required')
        cache = previous
        inputs += [PILOT / n for n in ('result.json', 'judgment_cache.json', 'worker_terminal_receipt.json')]
    inherited = dict(cache)
    root = PILOT if pilot else GRADE
    p = producing(root, 'four_per_arm_validity_only' if pilot else 'two_codec_whole_summary_semantics', inputs)
    write_json(root / 'judgment_cache.json', cache)
    metrics, joints, admitted = {}, {}, {}
    valid = True
    for arm in ARMS:
        def judge(role):
            oracle.grade_arm(root, arm, role, cases[:4] if pilot else cases, packets, cache)
            scores = q.read(root / f'{role}_{arm}_case_scores.json')
            value = {**metric(scores), 'judge': q.JUDGES[role]}
            write_json(root / f'{role}_{arm}_result.json', value)
            write_json(root / 'judgment_cache.json', cache)
            return scores, value
        secondary, value = judge('secondary')
        metrics[arm] = {'secondary': value}
        valid = valid and all(r['valid'] for r in secondary)
        admitted[arm] = not pilot and oracle.qualifies(value)
        joints[arm] = None
        if admitted[arm]:
            primary, metrics[arm]['primary'] = judge('primary')
            joints[arm] = rendering.teacher.joint_gate(secondary, primary)
    if any(cache.get(k) != v for k, v in inherited.items()):
        raise ValueError('a preceding first judgment changed')
    lineage.audit_cache(cache)
    qualified = [arm for arm in ARMS if joints[arm] is not None and joints[arm]['qualified']]
    status = ('all_codec_pilot_judgments_valid_only' if valid else 'codec_pilot_invalid') if pilot else (
        'fixed64_codec_component_semantically_qualified' if qualified else 'fixed64_codec_component_quality_failed')
    native.vendor.verify(p)
    result = {'status': status, 'metrics': metrics, 'joint': joints, 'primary_admitted': admitted,
        'qualified_codecs': qualified, 'canonical_first_judgments': len(cache),
        'inherited_first_judgments': len(inherited), 'student_task_accuracy': None, 'student_trained': False,
        'final_test_opened': False, 'speed_tested': False}
    write_json(root / 'result.json', result)
    print(json.dumps(result), flush=True)
    if pilot and not valid:
        raise ValueError('codec judgment validity pilot failed; all firsts retained')


def workflow():
    configure()
    lineage.require_judges()
    lineage.base_cache()
    require_numeric()
    if any(root.exists() for root in (WORKFLOW, PREP, PILOT, GRADE, RESULTS)) or REPORT.exists():
        raise ValueError('fresh codec semantic workflow required')
    p = producing(WORKFLOW, 'supervise_fixed64_codec_semantic_study')
    receipts = []
    for module, stage, root in ((MODULE, 'prepare', PREP), (MODULE, 'pilot', PILOT), (MODULE, 'grade', GRADE),
            (AUDITOR_MODULE, 'finalize', RESULTS), (AUDITOR_MODULE, 'register', None)):
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', module, stage]
        child = subprocess.Popen(command)
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'exit_code': code, 'command': command, 'child_pid': child.pid,
            'observer_pid': os.getpid(), 'observer': 'subprocess.wait'}
        write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root is not None and root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('codec semantic stage failed: ' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'codec_semantic_workflow_completed', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'grade', 'workflow'))
    args = parser.parse_args()
    {'prepare': prepare, 'pilot': lambda: grade(True), 'grade': grade, 'workflow': workflow}[args.stage]()
