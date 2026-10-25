"""Inventory original shared TRAIN summaries under the current native128 limits."""
from collections import Counter
import gc
import json
import os
from pathlib import Path
import resource

import torch

from experiments.structure_net import ncm_additional_training_reservation as reservation
from experiments.structure_net import ncm_lm1b_codec as native
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

ROOT = native.ROOT
HYPOTHESIS = 'ncm-shared-summary-native128-inventory-v1'
DATA = Path('data/experiments/ncm_memory/20261022_data_v1')
DEST = ROOT / '20261025_shared_summary_native128_inventory_v1'
COMPONENT = ROOT / '20261024_sol_teacher64_prepare_v2/cases.json'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-shared-corpus-inventory-plan.md')
TEST = Path('tests/structure_net/test_ncm_shared_summary_inventory.py')


def protected_metadata():
    index, components, paths = reservation.parent_metadata()
    # This file contains original source-only component metadata, not judge labels.
    teacher = reservation.read(COMPONENT)
    if len(teacher) != 64 or len({r['source_component'] for r in teacher}) != 64:
        raise ValueError('all64 held-out teacher/codec component sources required')
    components |= {r['source_component'] for r in teacher}
    protected = {key: set() for key in ('families', 'groups', 'hashes', 'turns')}
    rows = 0
    for row in index:
        if row['source_component'] in components:
            rows += 1
            protected['families'].update(row['families'])
            protected['groups'].update(row['groups'])
            protected['hashes'].add(row['source_sha256'])
            protected['turns'].update(row['informative_turns'])
    for row in teacher:
        protected['families'].update(reservation.graph.guard.source_family(v)
            for v in [row['source_id'], *[s['source_id'] for s in row['spans']]])
        protected['groups'].add(row['group_id'])
        protected['hashes'].add(row['source_sha256'])
        protected['turns'].update(s['turn_text_sha256'] for s in row['spans'])
    del index
    gc.collect()
    return protected, [*paths, COMPONENT], {'reserved_components': len(components),
        'component_rows': rows, **{k: len(v) for k, v in protected.items()}}


def reasons(row, ids, protected, specials):
    result = []
    if row['input_tokens'] < 192:
        result.append('application_bypass_below192')
    if row['input_tokens'] > 2048:
        result.append('input_above2048_application_tokens')
    if not row['target'].strip() or not 0 < row['target_tokens'] <= 128:
        result.append('target_empty_or_above128_application_tokens')
    if not 0 < len(ids) <= 126:
        result.append('target_not_one_complete_native128_frame')
    if set(ids) & specials:
        result.append('native_target_special_or_UNK')
    families = {reservation.graph.guard.source_family(v) for v in
        [row['source_id'], *[s['source_id'] for s in row['spans']]]}
    if families & protected['families']:
        result.append('protected_source_family')
    if row['group_id'] in protected['groups']:
        result.append('protected_group')
    if row['source_sha256'] in protected['hashes']:
        result.append('protected_exact_source')
    if {s['turn_text_sha256'] for s in row['spans']} & protected['turns']:
        result.append('protected_informative_turn')
    return result


def main():
    if DEST.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('fresh CPU-only original TRAIN inventory required')
    native.vendor.corpus.quality.dependencies()
    from experiments.structure_net.ncm_additional_training_sources import storage_encoding
    manifest = reservation.read(DATA / 'manifest.json')
    expected = manifest['files']['train.jsonl']
    if (expected['rows'] != 204967 or file_hash(DATA / 'train.jsonl') != expected['sha256']
            or manifest['target_origin'] != 'unchanged saved atomic summaries'):
        raise ValueError('unaltered complete original shared TRAIN corpus required')
    protected, paths, protected_counts = protected_metadata()
    tokenizer, encoding = native.tokenizer(), storage_encoding()
    files = [Path(__file__), PLAN, TEST, Path(reservation.__file__),
        Path(reservation.graph.guard.__file__), Path(native.__file__)]
    inputs = [*paths, DATA / 'manifest.json', DATA / 'train.jsonl',
        *[p for p in sorted((native.INVENTORY / 'bert-tokenizer').iterdir()) if p.is_file()]]
    producer = native.vendor.freeze(DEST, 'original_shared_TRAIN_summary_inventory_not_quality', [*files, *inputs], {
        'hypothesis_id': HYPOTHESIS, 'original_train_rows': 204967,
        'original_old_native_rows': 45136, 'input_range_cl100k': [192, 2048],
        'target_limit_cl100k': 128, 'native_lexical_limit': 126,
        'target_changes': False, 'metadata_only_split_exclusions': True,
        'full_text_near_duplicate_audit_completed': False,
        'old_test_file_opened_or_hashed': False, 'final_source_files_opened_or_hashed': False,
        'model_training': False, 'task_accuracy': None, 'speed_tested': False})
    for i, path in enumerate(files):
        (DEST / 'producer_source' / f'shared_inventory_{i}_{path.name}').write_bytes(path.read_bytes())
    exclusions, input_histogram, target_histogram, native_histogram = (Counter() for _ in range(4))
    eligible, unique_ids, groups, source_ids = 0, set(), set(), set()
    numeric_fit = protected_excluded = both_fit = 0
    with (DATA / 'train.jsonl').open() as source, (DEST / 'eligible_train_metadata.jsonl').open('x') as output:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            if (row['id'] in unique_ids or row['split'] != 'train'
                    or text_hash(row['source']) != row['source_sha256']
                    or text_hash(row['target']) != row['target_sha256']
                    or len(encoding.encode(row['source'], disallowed_special=())) != row['input_tokens']
                    or len(encoding.encode(row['target'], disallowed_special=())) != row['target_tokens']):
                raise ValueError('original source/summary/hash/token/split identity differs')
            unique_ids.add(row['id'])
            ids = tokenizer.encode(row['target'], add_special_tokens=False)
            why = reasons(row, ids, protected, set(tokenizer.all_special_ids))
            exclusions.update(why)
            range_reasons = [x for x in why if not x.startswith('protected_')]
            protection_reasons = [x for x in why if x.startswith('protected_')]
            numeric_fit += not range_reasons
            protected_excluded += bool(protection_reasons)
            both_fit += not range_reasons and bool(protection_reasons)
            input_histogram[str(row['input_tokens'] // 128 * 128)] += 1
            target_histogram[str(row['target_tokens'] // 16 * 16)] += 1
            native_histogram[str(len(ids) // 16 * 16)] += 1
            if not why:
                eligible += 1
                groups.add(row['group_id'])
                source_ids.add(row['source_id'])
                value = {k: row[k] for k in ('id', 'source_id', 'group_id', 'source_sha256',
                    'target_sha256', 'input_tokens', 'target_tokens')}
                value.update(original_line_number=number, native_target_tokens=len(ids))
                output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
            if number % 20000 == 0:
                print(json.dumps({'original_TRAIN_scanned': number, 'eligible_after_metadata_guards': eligible}), flush=True)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8_000_000_000:
                raise ValueError('inventory exceeds8GB host RSS')
    if number != expected['rows'] or eligible != numeric_fit - both_fit:
        raise ValueError('complete TRAIN coverage or independent exclusion accounting differs')
    native.vendor.verify(producer)
    write_json(DEST / 'protected_metadata_counts.json', protected_counts)
    result = {'status': 'complete_shared_TRAIN_native128_inventory_semantics_and_training_admission_pending',
        'original_train_pairs': number, 'old_native_train_pairs': 45136,
        'current_numeric_fit_before_metadata_guards': numeric_fit,
        'eligible_after_metadata_guards': eligible, 'eligible_source_sessions': len(source_ids),
        'eligible_groups': len(groups), 'all_protected_records': protected_excluded,
        'otherwise_numeric_fit_but_protected': both_fit, 'exclusion_counts_can_overlap': dict(exclusions),
        'histograms': {'input_cl100k_bin128': dict(input_histogram),
            'target_cl100k_bin16': dict(target_histogram), 'target_BERT_bin16': dict(native_histogram)},
        'protected_metadata': protected_counts,
        'eligible_metadata_sha256': file_hash(DEST / 'eligible_train_metadata.jsonl'),
        'original_train_sha256': expected['sha256'], 'targets_unchanged': True,
        'full_text_near_duplicate_audit_completed': False, 'training_admitted': False,
        'new_judge_requests': 0, 'task_accuracy': None, 'old_test_file_opened_or_hashed': False,
        'final_source_files_opened_or_hashed': False, 'speed_tested': False,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    write_json(DEST / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'histograms'}), flush=True)


if __name__ == '__main__':
    main()
