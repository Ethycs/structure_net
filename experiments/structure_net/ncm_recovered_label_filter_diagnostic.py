"""Describe saved label scores after catalog filters; never admit held labels."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

BASE = Path('data/experiments/ncm_paper_regime')
ROOT = BASE/'20261028_recovered_label_filter_diagnostic_v1'
SAMPLE = BASE/'20261027_recovered_label_sample_v1'
CATALOG = BASE/'20261028_recovered_paraphrase_catalog_v1'
PREPARED = BASE/'20261027_recovered_label_review_prepare_v1'
GRADED = BASE/'20261027_recovered_label_review_grade_v1'
RESERVATION = {'reserved_diagnostic_component', 'reserved_diagnostic_native_frame'}


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def summarize(rows):
    result = {'cases': len(rows), 'both_valid': sum(r['both_valid'] for r in rows),
              'joint_passed': sum(r['joint_passed'] for r in rows)}
    result['joint_pass_rate'] = (result['joint_passed']/len(rows)
        if rows and result['both_valid'] == len(rows) else None)
    result['per_judge'] = {}
    for role in ('primary', 'secondary'):
        scores = [r['scores'][role] for r in rows]
        result['per_judge'][role] = {'valid': sum(s['valid'] for s in scores),
            'passed': sum(s['passed'] for s in scores),
            'components': {k: sum(s[k] for s in scores) for k in ('faithful', 'salient', 'readable')}}
    result['nonreservation_exclusions'] = dict(Counter(reason for r in rows for reason in r['nonreservation_exclusions']))
    return result


def run():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only descriptive replay required')
    paths = [Path(__file__), SAMPLE/'cases.json', PREPARED/'cases.json',
             CATALOG/'excluded.jsonl', GRADED/'primary_saved_scores.json', GRADED/'secondary_saved_scores.json']
    for parent in (SAMPLE, PREPARED, CATALOG, GRADED):
        receipt = read(parent/'actual_tool_receipt.json')
        if (receipt['actual_tool_exit_code'] != 0
                or receipt['artifacts'].get(str(parent/'result.json')) != sha(parent/'result.json')):
            raise ValueError('actual successful parent receipt required')
        bound = {**read(parent/'result.json').get('artifacts', {}), **receipt['artifacts']}
        for path in paths:
            if path.parent == parent and bound.get(str(path)) != sha(path):
                raise ValueError('consumed saved artifact differs: '+str(path))
        paths += [parent/'result.json', parent/'actual_tool_receipt.json']
    hashes = {str(p): sha(p) for p in paths}
    sample = read(SAMPLE/'cases.json'); prepared = read(PREPARED/'cases.json')
    if len(sample) != 128 or len(prepared) != 128 or len({r['id'] for r in sample}) != 128:
        raise ValueError('complete fixed 128-label review required')
    identities = {r['id'] for r in sample}; exclusions = {}
    with (CATALOG/'excluded.jsonl').open() as stream:
        for row in map(json.loads, stream):
            if row['id'] in identities:
                if row['id'] in exclusions: raise ValueError('duplicate catalog exclusion')
                exclusions[row['id']] = row['reasons']
    if set(exclusions) != identities: raise ValueError('all diagnostic labels must remain excluded')
    scores = {role: read(GRADED/(role+'_saved_scores.json')) for role in ('primary', 'secondary')}
    if any(len(v) != 128 for v in scores.values()): raise ValueError('complete paired scores required')
    records = []
    for index, (row, case) in enumerate(zip(sample, prepared, strict=True)):
        if case['pair_id'] != row['id'] or case['texts']['saved'] != row['target_text']:
            raise ValueError('saved review case/target pairing differs')
        reasons = exclusions[row['id']]
        if not RESERVATION <= set(reasons): raise ValueError('whole held component and frame must remain excluded')
        compact = {}
        for role, values in scores.items():
            score = values[index]
            if (score['id'] != case['id'] or score['text'] != row['target_text']
                    or not score['valid'] or len(score['evaluations']) != 1):
                raise ValueError('same valid first-judgment summary required')
            verdict = score['evaluations'][0]
            if any(type(verdict[k]) is not bool for k in ('faithful', 'salient', 'readable')):
                raise ValueError('complete semantic component scores required')
            expected = bool(score['text'].strip()) and score['tokens'] <= 128 and all(
                verdict[k] for k in ('faithful', 'salient', 'readable'))
            if score['passed'] != expected: raise ValueError('saved complete-summary score differs')
            compact[role] = {k: score[k] for k in ('valid', 'passed')}
            compact[role].update({k: verdict[k] for k in ('faithful', 'salient', 'readable')})
        nonreservation = [r for r in reasons if r not in RESERVATION]
        records.append({'id': row['id'], 'stratum': row['stratum'], 'scores': compact,
            'both_valid': all(v['valid'] for v in compact.values()),
            'joint_passed': all(v['passed'] for v in compact.values()),
            'catalog_exclusions': reasons, 'nonreservation_exclusions': nonreservation,
            'would_pass_nonreservation_filters': not nonreservation, 'training_admitted': False})
    overall = summarize(records)
    old = read(GRADED/'result.json')['metrics']['overall']
    if (overall['cases'], overall['both_valid'], overall['joint_passed']) != (old['cases'], old['both_valid'], old['joint_passed']):
        raise ValueError('original complete review must replay unchanged')
    groups = {'would_pass_nonreservation_filters': [r for r in records if r['would_pass_nonreservation_filters']],
              'other_filter_exclusions': [r for r in records if not r['would_pass_nonreservation_filters']]}
    result = {'schema': 'ncm-recovered-label-filter-diagnostic-v1', 'status': 'complete_descriptive_saved_label_replay',
        'overall': overall, 'groups': {name: summarize(rows) for name, rows in groups.items()},
        'model_task_accuracy': None, 'estimates_current_training_pool_accuracy': False,
        'posthoc_subgroup_analysis': True, 'held_cases_admitted_to_training': 0,
        'new_model_calls': 0, 'new_judge_calls': 0, 'final_test_opened': False, 'artifacts': hashes}
    if any(sha(p) != h for p,h in hashes.items()): raise ValueError('saved inputs changed')
    ROOT.mkdir()
    for name, value in (('cases.json', records), ('result.json', result)):
        with (ROOT/name).open('x') as f: json.dump(value,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__': run()
