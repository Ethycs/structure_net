"""Freeze source-diverse saved-label diagnostics without making model calls."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_recovered_pair_training_pool as pool

ROOT = pool.BASE/'20261027_recovered_label_sample_v1'
AUDIT = pool.BASE/'20261027_recovered_pair_training_pool_audit_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-recovered-label-sample.md')
SEED = 'recovered-label-review-20261027-v1:'


def select(metadata):
    chosen, components = [], set()
    for name, lower, upper in (('long', 512, 2048), ('short', 192, 511)):
        ordered = sorted((r for r in metadata if lower <= r['source_cl100k_tokens'] <= upper),
            key=lambda r: (hashlib.sha256((SEED+r['id']).encode()).hexdigest(), r['id']))
        count = 0
        for row in ordered:
            if row['source_component'] in components:
                continue
            chosen.append({**row, 'stratum': name})
            components.add(row['source_component']); count += 1
            if count == 64:
                break
        if count != 64:
            raise ValueError('64 distinct source components required in each stratum')
    return chosen


def run():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only sample preparation required')
    result, audited = pool.read(pool.ROOT/'result.json'), pool.read(AUDIT/'result.json')
    for parent in (pool.ROOT, AUDIT):
        receipt = pool.read(parent/'actual_tool_receipt.json')
        if receipt['actual_tool_exit_code'] != 0:
            raise ValueError('actual successful parent required')
        for path, sha in receipt['artifacts'].items():
            if pool.file_hash(path) != sha:
                raise ValueError('parent artifact differs')
    if (result['status'] != 'source_guarded_native_fit_review_pool_complete'
            or audited['status'] != 'all_export_rows_and_outcomes_verified'
            or audited['pairs'] != 104503 or audited['export_sha256'] != result['export']['sha256']):
        raise ValueError('complete audited recovered review pool required')
    export = Path(result['export']['path'])
    if pool.file_hash(export) != result['export']['sha256']:
        raise ValueError('audited export changed')
    paths = [Path(__file__), PLAN, Path(pool.__file__), export, Path('pixi.lock'), Path('pyproject.toml')]
    paths += [p/n for p in (pool.ROOT, AUDIT) for n in ('result.json', 'actual_tool_receipt.json')]
    proof = {'schema': 'nal.ncm-recovered-label-sample.v1', 'stage': 'fixed_source_diverse_saved_label_sample',
        'selection_seed': SEED, 'per_stratum': 64, 'one_pair_per_component': True,
        'inputs': {str(p): pool.file_hash(p) for p in paths}, 'new_model_calls': 0}
    ROOT.mkdir(); pool.write(ROOT/'producer.json', proof)
    metadata = []
    with export.open() as source:
        for line in source:
            row = json.loads(line)
            metadata.append({k: row[k] for k in ('id', 'source_sha256', 'target_sha256', 'source_component', 'source_cl100k_tokens')})
    selected = select(metadata); by_id = {r['id']: r for r in selected}; cases = {}
    if len(metadata) != 104503 or select(list(reversed(metadata))) != selected:
        raise ValueError('selection population or order independence failed')
    with export.open() as source:
        for line in source:
            row = json.loads(line)
            if row['id'] in by_id:
                cases[row['id']] = {**row, 'stratum': by_id[row['id']]['stratum'], 'review_case_id': 'recovered-label-'+row['id']}
    ordered = [cases[r['id']] for r in selected]
    if len(ordered) != 128 or len({r['source_component'] for r in ordered}) != 128:
        raise ValueError('all selected independent components required')
    pool.write(ROOT/'selection.json', selected); pool.write(ROOT/'cases.json', ordered)
    for path, sha in proof['inputs'].items():
        if pool.file_hash(path) != sha:
            raise ValueError('input changed during selection')
    value = {'schema': proof['schema'], 'status': 'fixed_saved_label_review_sample_prepared', 'cases': 128,
        'strata': dict(Counter(r['stratum'] for r in ordered)), 'distinct_source_components': 128,
        'training_admitted': False, 'semantic_quality_reviewed': False, 'new_model_calls': 0,
        'model_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(p): pool.file_hash(p) for p in ROOT.iterdir() if p.is_file()}}
    pool.write(ROOT/'result.json', value); print(json.dumps(value), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args()
    existed = ROOT.exists()
    try:
        run()
    except Exception as error:
        if ROOT.exists() and not existed:
            pool.write(ROOT/'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise
