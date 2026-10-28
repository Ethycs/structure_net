"""Describe complete training/DEV input coverage without scoring model outputs."""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

BASE = Path('data/experiments/ncm_paper_regime')
ROOT = BASE/'20261028_recovered_input_coverage_v1'
CATALOG = BASE/'20261028_recovered_paraphrase_catalog_v1'
META = BASE/'20261028_recovered_paraphrase_metadata_full_v1'
BINDINGS = BASE/'20261027_isolated_task_supervision_preflight_v1/bindings.json'
PREPARE = BASE/'20261028_recovered_paraphrase_training_prepare_v1'
SOURCE = BASE/'20261024_pinned_date_development_sources_v1/cases.json'
MANIFEST = BASE/'20261024_pinned_date_development_features_v1/manifest.json'
SCHEDULE = BASE/'20261028_recovered_source_balancing_v1/planned_examples.jsonl'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as f: yield from map(json.loads, f)


def profile(rows):
    lengths = sorted(r['positions'] for r in rows)
    if not lengths or min(lengths) < 1: raise ValueError('complete nonempty input lengths required')
    return {'cases': len(rows), 'unique_source_texts': len({r['source_sha256'] for r in rows}),
        'source_components': len({r['source_component'] for r in rows}),
        'source_positions': {'min': lengths[0], 'median_nearest_rank': lengths[math.ceil(.5*len(lengths))-1],
            'p95_nearest_rank': lengths[math.ceil(.95*len(lengths))-1], 'max': lengths[-1], 'sum': sum(lengths)},
        'source_span_histogram': dict(sorted(Counter(r['spans'] for r in rows).items())),
        'multi_span_cases': sum(r['spans'] > 1 for r in rows),
        'position_bins': dict(Counter(next((f'<= {n}' for n in (512,1024,2048,4096,8192)
                                           if r['positions'] <= n), '> 8192') for r in rows))}


def compact(row, positions):
    return {'id': row['id'], 'source_sha256': row['source_sha256'],
            'source_component': row['source_component'], 'spans': len(row['spans']), 'positions': positions}


def run():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only descriptive inventory required')
    receipt = read(PREPARE/'actual_tool_receipt.json')
    if (receipt['actual_tool_exit_code'] != 0
            or receipt['artifacts'].get(str(PREPARE/'result.json')) != sha(PREPARE/'result.json')
            or not read(PREPARE/'result.json')['data_admitted']):
        raise ValueError('complete recovered data preparation must have finished')
    bound = read(BINDINGS)
    anchor_root, anchor_targets = Path(bound['catalog']['prepare']), Path(bound['targets']['full'])
    paths = [Path(__file__), BINDINGS, PREPARE/'result.json', PREPARE/'actual_tool_receipt.json',
             CATALOG/'sources.jsonl', CATALOG/'pairs.jsonl', META/'index.jsonl', SOURCE, MANIFEST, SCHEDULE,
             anchor_root/'source_index.json', anchor_targets/'cases.json']
    hashes = {str(p): sha(p) for p in paths}
    metadata = {r['id']: r for r in lines(META/'index.jsonl')}
    recovered = [compact(r, metadata[r['id']]['source_positions']) for r in lines(CATALOG/'sources.jsonl')]
    if len(recovered) != 40890 or len(metadata) != 40890: raise ValueError('complete recovered context population required')
    recovered_ids = {r['id']: r for r in recovered}
    pairs = {r['id']: {'source_record_id': r['source_record_id'], 'lexical_tokens': r['native_lexical_tokens']}
             for r in lines(CATALOG/'pairs.jsonl')}
    anchor_index = {r['id']: r for r in read(anchor_root/'source_index.json')}
    anchors, anchor_lexical = [], {}
    for r in read(anchor_targets/'cases.json'):
        anchors.append(compact(r, anchor_index[r['id']]['source_positions']))
        anchor_lexical[r['id']] = r['native_lexical_tokens']
    if len(pairs) != 43894 or len(anchors) != 2384: raise ValueError('complete pair and anchor populations required')
    anchor_ids = {r['id']: r for r in anchors}
    source, manifest = read(SOURCE), read(MANIFEST)
    cohorts = {c: [compact(r, manifest[r['id']]['source_tokens']) for r in rows] for c, rows in source.items()}
    if set(cohorts) != {'short_development','long_diagnostic'} or any(len(v) != 64 for v in cohorts.values()):
        raise ValueError('only original two DEV cohorts may enter this inventory')
    train_hashes = {r['source_sha256'] for r in recovered+anchors}
    overlaps = {c: len(train_hashes & {r['source_sha256'] for r in rows}) for c, rows in cohorts.items()}
    if any(overlaps.values()): raise ValueError('exact TRAIN/DEV source overlap')
    exposure = Counter(); populations = Counter()
    for e in lines(SCHEDULE):
        if e['population'] == 'recovered':
            pair = pairs[e['id']]; row = recovered_ids[pair['source_record_id']]; lexical = pair['lexical_tokens']
        elif e['population'] == 'anchor': row = anchor_ids[e['id']]; lexical = anchor_lexical[e['id']]
        else: raise ValueError('unknown planned population')
        if row['source_sha256'] != e['source_sha256']: raise ValueError('planned source identity differs')
        populations[e['population']] += 1
        exposure['presentations'] += 1; exposure['noisy_presentations'] += e['noisy']
        exposure['source_feature_positions'] += row['positions']
        exposure['native_target_lexical_tokens'] += lexical
        exposure['multi_span_presentations'] += row['spans'] > 1
        exposure['source_positions_above4096_presentations'] += row['positions'] > 4096
    if exposure['presentations'] != 205248 or exposure['noisy_presentations'] != 102624:
        raise ValueError('complete unchanged four-epoch schedule required')
    exposure['native_word_positions_including_framing_and_padding'] = exposure['presentations']*128
    exposure['code_positions_including_hierarchy_and_padding'] = exposure['presentations']*641
    profiles = {name: profile(rows) for name, rows in {'recovered_contexts': recovered,
                 'reviewed_anchors': anchors, **cohorts}.items()}
    maximum = profiles['recovered_contexts']['source_positions']['max']
    value = {'schema': 'ncm-recovered-input-coverage-v1', 'status': 'complete_descriptive_inventory',
        'profiles': profiles, 'train_union_unique_source_texts': len(train_hashes),
        'anchor_recovered_source_overlap': len({r['source_sha256'] for r in anchors} & {r['source_sha256'] for r in recovered}),
        'DEV_exact_source_hash_overlaps': overlaps,
        'DEV_above_recovered_source_position_max': {c: sum(r['positions'] > maximum for r in rows) for c,rows in cohorts.items()},
        'planned_exposure': dict(exposure), 'planned_population_counts': dict(populations),
        'source_position_units': 'GPT2-aligned source feature positions; not stored cl100k tokens',
        'exposure_is_planned_not_completed_training': True, 'semantic_accuracy': None,
        'training_updates_added': 0, 'new_model_calls': 0, 'new_judge_calls': 0,
        'final_test_opened': False, 'replaces_existing_split_audits': False, 'artifacts': hashes}
    if any(sha(p) != h for p,h in hashes.items()): raise ValueError('inventory inputs changed during reading')
    ROOT.mkdir()
    with (ROOT/'result.json').open('x') as f: json.dump(value,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k != 'artifacts'}),flush=True)


if __name__ == '__main__': run()
