"""Check original speaker metadata behind the fixed saved-label diagnostic."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from experiments.structure_net import ncm_recovered_pair_context as original

BASE = original.BASE
ROOT = BASE/'20261028_recovered_label_role_diagnostic_v1'
CASES = BASE/'20261027_recovered_label_review_prepare_v1/cases.json'
GRADE = BASE/'20261027_recovered_label_review_grade_v1'
# Human review of the existing reasons, not a replacement judge or new pass rule.
ROLE_ONLY = {
    '7d9efa8b2f529b0524c4f61ce9cd3dd378049b4cf3b63f4a63f2c2a95f8c5463',
    '066b9c36814b6fefb5592ceaca2e81959c947803abf00de287840b95568d9dcc',
    '6f53082bc482ef3dcf1676496aebbce3d82e48842a2a07e21d03d0dd5decf30a',
    '42bf529618bf11a600fae5c3b280a5a4f41d16dc433a73699d8cb5633a459933',
    '9b91a8aed3e16b6ee6efaf0ad715552c1e35696e24a513689c9d2aaba1e6d74a',
    '9aadf666ff5daec5baebdfb6c9b3113e74bf55f5e3b0fbd2203592c80a3e0661',
    'a181f0b190e35dd43b862df537722458472a2b6a4c8e61fefa8a54fa1648c5f4',
}


def run():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only metadata diagnostic required')
    cases = original.read(CASES); lookup = {r['pair_id']: r for r in cases}
    scores = {role: {r['id']: r for r in original.read(GRADE/f'{role}_saved_scores.json')}
              for role in ('primary', 'secondary')}
    if len(cases) != 128 or len(lookup) != 128: raise ValueError('fixed full128 sample required')
    if any(set(rows) != {r['id'] for r in cases} for rows in scores.values()):
        raise ValueError('same128 existing qualified decisions required')
    inputs = [CASES, GRADE/'result.json', GRADE/'actual_tool_receipt.json',
              *[GRADE/f'{role}_saved_scores.json' for role in scores]]
    receipt = original.read(GRADE/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 0: raise ValueError('completed saved-label review required')
    for p, expected in receipt['artifacts'].items():
        if original.sha(p) != expected: raise ValueError('saved-label evidence changed')
    proof = {'schema': 'nal.ncm-recovered-role-diagnostic.v1',
        'source_hashes': {str(p): original.sha(p) for p in (Path(__file__), Path(original.__file__))},
        'input_hashes': {str(p): original.sha(p) for p in inputs},
        'database_binding': 'exact selected SQL rows and SHA256 of canonical JSON, not an asserted new full-database hash',
        'posthoc_role_only_cases': sorted(ROLE_ONLY), 'semantic_regrading': False}
    ROOT.mkdir(); original.write(ROOT/'producer.json', proof)
    found = {p: [] for p in lookup}; logical = []
    for path in original.DATABASES:
        db = original.connect(path)
        query = ('SELECT r.pair_id,l.id,l.kind,l.target,l.pointers,l.metadata '
                 'FROM label_results r JOIN labels l ON l.id=r.id WHERE r.pair_id IN ('+
                 ','.join('?' for _ in lookup)+') ORDER BY r.pair_id,l.id')
        rows = db.execute(query, tuple(lookup)).fetchall(); db.close()
        logical.append({'database': str(path), 'rows': rows})
        for pair, label, kind, target, raw_pointers, raw_meta in rows:
            case = lookup[pair]; pointers = json.loads(raw_pointers); meta = json.loads(raw_meta)
            if original.text_sha(target) != case['target_sha256']: raise ValueError('target binding differs')
            if (kind != 'raw' or len(pointers) != 1
                    or pointers[0].get('span_text_sha256') != case['source_sha256']): continue
            found[pair].append({'origin': str(path), 'label_id': label,
                'source_pointer': pointers[0], 'summarizer_identity': meta.get('summarizer_identity')})
    original.write(ROOT/'selected_original_label_rows.json', logical)
    output = []
    for row in cases:
        decisions = [scores[role][row['id']] for role in scores]
        if any(s['text'] != row['target_text'] or not s['valid'] for s in decisions):
            raise ValueError('same valid saved-target judgments required')
        roles = sorted({v['source_pointer'].get('role') for v in found[row['pair_id']]
                        if v['source_pointer'].get('role') is not None})
        shortlisted = row['pair_id'] in ROLE_ONLY
        if shortlisted and (row['target_text'] in row['source'] or all(s['passed'] for s in decisions)):
            raise ValueError('posthoc role-only subset differs')
        output.append({'id': row['id'], 'pair_id': row['pair_id'], 'source_sha256': row['source_sha256'],
            'target_sha256': row['target_sha256'], 'exact_substring_target': row['target_text'] in row['source'],
            'original_contexts': found[row['pair_id']], 'observed_roles': roles,
            'joint_pass_before_metadata': all(s['passed'] for s in decisions),
            'posthoc_role_only_rejection': shortlisted,
            'original_metadata_unambiguously_assistant': roles == ['assistant'],
            'existing_reasons': {role: [e['reason'] for e in scores[role][row['id']]['evaluations']] for role in scores}})
    if sum(r['posthoc_role_only_rejection'] for r in output) != 7: raise ValueError('all seven selected cases required')
    original.write(ROOT/'cases.json', output)
    for collection in (proof['source_hashes'], proof['input_hashes']):
        for p, expected in collection.items():
            if original.sha(p) != expected: raise ValueError('diagnostic input changed')
    value = {'schema': proof['schema'], 'status': 'fixed128_original_role_metadata_checked',
        'cases': 128, 'cases_with_direct_original_context': sum(bool(r['original_contexts']) for r in output),
        'posthoc_role_only_rejections': 7,
        'role_only_rejections_with_unambiguous_original_assistant_role': sum(
            r['posthoc_role_only_rejection'] and r['original_metadata_unambiguously_assistant'] for r in output),
        'old_semantic_decisions_unchanged': True, 'new_requests': 0,
        'training_admitted': False, 'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(p): original.sha(p) for p in ROOT.iterdir() if p.is_file()}}
    original.write(ROOT/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed:
            original.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
