"""Recover the eight completed first responses after the v1 writer-directory failure."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_sol_evidence_teacher as base
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT, data, lineage = base.native, base.q, base.ROOT, base.data, base.lineage
MODELS = base.MODELS
MODULE = 'experiments.structure_net.ncm_sol_evidence_teacher_recovery'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sol-teacher-writer-recovery.md')
TEST = Path('tests/structure_net/test_ncm_sol_evidence_teacher_recovery.py')
FAILED_WORKFLOW, FAILED_PREP, FAILED_PILOT = base.WORKFLOW, base.PREP, base.PILOT
ORIGINAL_PLAN = base.PLAN
WORKFLOW = ROOT / '20261024_sol_teacher64_workflow_v2'
PREP = ROOT / '20261024_sol_teacher64_prepare_v2'
PILOT = ROOT / '20261024_sol_teacher64_systems_pilot_v2'
GENERATE = ROOT / '20261024_sol_teacher64_candidates_v2'
AUDIT = ROOT / '20261024_sol_teacher64_raw_audit_v2'
_configure, _producing = base.configure, base.producing


def configure():
    _configure()
    for name in ('MODULE', 'PLAN', 'TEST', 'WORKFLOW', 'PREP', 'PILOT', 'GENERATE', 'AUDIT'):
        setattr(base, name, globals()[name])
    base.configure, base.producing = configure, producing


def producing(root, stage, inputs=()):
    if (q.read(FAILED_WORKFLOW / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 1
            or q.read(FAILED_WORKFLOW / 'pilot_receipt.json')['exit_code'] != 1):
        raise ValueError('actual failed v1 writer must remain terminal')
    files = [Path(__file__), PLAN, TEST, ORIGINAL_PLAN]
    raw = sorted((FAILED_PILOT / 'requests').rglob('attempt_*.json'))
    p = _producing(root, stage, [*inputs, *files, *raw, FAILED_WORKFLOW / 'actual_tool_receipt.json',
        FAILED_WORKFLOW / 'pilot_receipt.json', FAILED_PREP / 'cases.json', FAILED_PREP / 'pilot_ids.json'])
    p['config'].update(writer_recovery='create case directory; reconstruct eight original first candidates from raw responses',
        original_pilot_candidates_reused=True, candidate_text_edited=False, previous_failure_preserved=True)
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'sol_writer_recovery_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    if root in (PILOT, GENERATE):
        (root / 'cases').mkdir()
    return p


def descriptor(row, stage):
    paths = sorted((FAILED_PILOT / 'requests' / row['id'] / stage).glob('attempt_*.json'))
    if (not 1 <= len(paths) <= 3
            or [p.name for p in paths] != [f'attempt_{i}.json' for i in range(len(paths))]):
        raise ValueError('complete original ordered stage attempts required; no new allowance')
    value = None
    for i, path in enumerate(paths):
        raw = q.read(path)
        if raw['valid']:
            if i != len(paths) - 1:
                raise ValueError('later response after valid first')
            value = base.original.parse(raw['response']['choices'][0]['message']['content'], row)
    if value is None and len(paths) != 3:
        raise ValueError('unfinished original stage; no automatic retry')
    return {'status': 'valid' if value is not None else 'invalid', 'value': value, 'attempts': [str(p) for p in paths]}


def reconstruct(row, encoding, tokenizer):
    draft = descriptor(row, 'draft')
    first_draft = base.replay(row, 'draft', draft, encoding)
    review = descriptor(row, 'review') if first_draft is not None else None
    first_review = base.replay(row, 'review', review, encoding, first_draft) if review is not None else None
    text = first_review['summary'] if first_review is not None else ''
    lexical = tokenizer.encode(text, add_special_tokens=False)
    stored = len(encoding.encode(text, disallowed_special=()))
    fits = bool(text.strip()) and 0 < stored <= 128 and len(lexical) <= 126 and not set(lexical) & {0, 101}
    return {**row, 'draft': draft, 'review': review, 'target_text': text, 'teacher_valid': first_review is not None,
        'stored_tokens': stored, 'native_lexical_tokens': len(lexical), 'capacity_fit': fits,
        'native_word_ids': [101, *lexical, 101] + [0] * (126 - len(lexical)) if fits else None,
        'semantic_qualified': False}


def save_candidate(root, value):
    write_json(root / 'cases' / (value['id'] + '.json'), value)


def pilot():
    configure()
    native.vendor.terminal(PREP)
    rows = q.read(PREP / 'cases.json')
    ids = q.read(PREP / 'pilot_ids.json')
    if rows != q.read(FAILED_PREP / 'cases.json') or ids != q.read(FAILED_PREP / 'pilot_ids.json'):
        raise ValueError('unchanged64selection and eight original pilot cases required')
    p = producing(PILOT, 'recover_eight_completed_concurrent_firsts_without_new_requests',
        [PREP / 'cases.json', PREP / 'pilot_ids.json', PREP / 'worker_terminal_receipt.json'])
    encoding, tokenizer = data.storage_encoding(), native.tokenizer()
    labels = [reconstruct(row, encoding, tokenizer) for row in rows if row['id'] in ids]
    if len(labels) != 8:
        raise ValueError('all eight original pipelines required')
    for value in labels:
        save_candidate(PILOT, value)
    write_json(PILOT / 'labels.json', labels)
    valid, capacity = sum(r['teacher_valid'] for r in labels), sum(r['capacity_fit'] for r in labels)
    native.vendor.verify(p)
    result = {'status': 'eight_case_sol_teacher_systems_passed' if valid == capacity == 8 else 'eight_case_sol_teacher_systems_failed',
        'cases': 8, 'valid': valid, 'capacity_fit': capacity, 'labels_sha256': file_hash(PILOT / 'labels.json'),
        'new_requests': 0, 'original_firsts_recovered': 8, 'student_training_admitted': False,
        'semantic_qualification_pending': True, 'canonical_semantic_cache_changed': False, 'task_accuracy': None}
    write_json(PILOT / 'result.json', result)
    print(json.dumps(result), flush=True)
    if valid != 8 or capacity != 8:
        raise ValueError('preserved original pilot fails admission; no replacement labels')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'pilot', 'generate', 'workflow', 'audit'))
    args = parser.parse_args()
    configure()
    {'prepare': base.prepare, 'pilot': pilot, 'generate': base.generate, 'workflow': base.workflow, 'audit': base.audit}[args.stage]()
