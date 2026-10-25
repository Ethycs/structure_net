"""Replay stored historical TRAIN targets joined to frozen source features."""
from collections import Counter
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_shared_summary_inventory as inventory
from experiments.structure_net import ncm_shared_summary_source_guard as guard
from experiments.structure_net import ncm_additional_source_features as features
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

native, ROOT = inventory.native, inventory.ROOT
JOIN = ROOT / '20261025_core_target_cached_source_join_v1'
DEST = ROOT / '20261025_core_cached_target_admission_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-core-cached-target-admission-plan.md')
TEST = Path('tests/structure_net/test_ncm_core_cached_target_admission.py')


def replay_binding(row, pair, cached, source, lexical, protected, specials):
    keys = ('id', 'source_id', 'group_id', 'source_sha256', 'target_sha256', 'input_tokens', 'target_tokens')
    if (row['split'] != 'train' or any(row[k] != pair[k] for k in keys)
            or text_hash(row['source']) != pair['source_sha256']
            or text_hash(row['target']) != pair['target_sha256']
            or len(lexical) != pair['native_target_tokens']
            or inventory.reasons(row, lexical, protected, specials)):
        raise ValueError('original TRAIN source/target/native/protected binding differs')
    if (source['id'] != cached['id'] or source['source'] != row['source']
            or source['source_sha256'] != row['source_sha256']
            or source['source_component'] != cached['source_component']
            or source['alignment_sha256'] != cached['alignment_sha256']
            or source['native_prompt_sha256'] != cached['native_prompt_sha256']
            or source['source_truncated']):
        raise ValueError('cached complete source/component/prompt/alignment differs')


def main():
    if DEST.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only historical TRAIN replay required')
    native.vendor.corpus.quality.dependencies()
    state = inventory.reservation.read(JOIN / 'result.json')
    matches_path = JOIN / 'metadata_matches.json'
    if (state['unique_source_matches'] != 4949 or state['training_admitted']
            or file_hash(matches_path) != state['metadata_matches_sha256']):
        raise ValueError('unchanged metadata-only exact-source join required')
    for binding in state['input_bindings'].values():
        if file_hash(binding['path']) != binding['sha256']:
            raise ValueError('completed source or feature metadata changed')
    if inventory.reservation.read(inventory.DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual original complete TRAIN inventory required')
    original = inventory.reservation.read(inventory.DEST / 'result.json')
    if file_hash(inventory.DATA / 'train.jsonl') != original['original_train_sha256']:
        raise ValueError('unaltered complete original TRAIN data required')
    protected, protection_paths, _ = inventory.protected_metadata()
    selected = inventory.reservation.read(inventory.reservation.PARENT / 'selection_metadata.json')
    reserved = {r['source_component'] for values in selected.values() for r in values}
    reserved |= {r['source_component'] for r in inventory.reservation.read(inventory.COMPONENT)}
    if len(reserved) != 512:
        raise ValueError('all512 evaluation/component reservations required')
    development = guard.development
    if file_hash(development.SOURCE / 'cases.json') != development.SOURCE_SHA:
        raise ValueError('unchanged open DEV source cases required')
    dev = development.validate_sources(inventory.reservation.read(development.SOURCE / 'cases.json'))
    heldout = [r for k in development.COHORTS for r in dev[k]]
    heldout += inventory.reservation.read(inventory.COMPONENT)
    text_index = guard.TextIndex([r['source'] for r in heldout])
    matches = inventory.reservation.read(matches_path)
    by_line = {r['original_pair']['original_line_number']: r for r in matches}
    if len(by_line) != 4949:
        raise ValueError('unique original physical TRAIN rows required')
    paths = [Path(__file__), PLAN, TEST, Path(inventory.__file__), Path(guard.__file__),
        Path(features.__file__), matches_path, JOIN / 'result.json',
        inventory.DATA / 'train.jsonl', inventory.DATA / 'manifest.json',
        inventory.DEST / 'producer.json', inventory.DEST / 'result.json', inventory.DEST / 'actual_tool_receipt.json',
        development.SOURCE / 'cases.json', *protection_paths]
    paths += [Path(v['path']) for v in state['input_bindings'].values()]
    paths += [Path(r['cached_source_metadata']['path']) for r in matches]
    paths += [Path(r['cached_feature']['header_path']) for r in matches]
    producer = native.vendor.freeze(DEST, 'replay4949_historical_TRAIN_stored_targets_and_cached_sources', paths,
        {'cases_before_replay': 4949, 'historical_core_LME_TRAIN': True,
         'broader_non_LME_partition_unchanged': True, 'reserved_components_excluded': 512,
         'targets_changed': False, 'feature_payload_validation': 'existing bounded loader verifies at each use',
         'feature_reextraction': False, 'student_training_admitted': False, 'labels_semantically_qualified': False,
         'new_requests': 0, 'task_accuracy': None, 'final_test_opened': False})
    from experiments.structure_net.ncm_additional_training_sources import storage_encoding
    encoding, tokenizer = storage_encoding(), native.tokenizer()
    result, conflicts, subset_index, manifest = [], [], [], {}
    exclusions = Counter()
    replayed = 0
    with (inventory.DATA / 'train.jsonl').open() as handle:
        for line_number, line in enumerate(handle, 1):
            matched = by_line.get(line_number)
            if matched is None:
                continue
            row = json.loads(line)
            pair, cached, feature = (matched[k] for k in ('original_pair', 'cached_source_metadata', 'cached_feature'))
            if file_hash(cached['path']) != cached['sha256'] or file_hash(feature['header_path']) != feature['header_sha256']:
                raise ValueError('cached source metadata or feature header changed')
            source = inventory.reservation.read(cached['path'])
            lexical = tokenizer.encode(row['target'], add_special_tokens=False)
            replay_binding(row, pair, cached, source, lexical, protected, set(tokenizer.all_special_ids))
            if (len(encoding.encode(row['source'], disallowed_special=())) != row['input_tokens']
                    or len(encoding.encode(row['target'], disallowed_special=())) != row['target_tokens']):
                raise ValueError('original token counts differ')
            header = inventory.reservation.read(feature['header_path'])
            if (header['binding'] != features.binding(source)
                    or feature['source_tokens'] != cached['source_positions']
                    or cached['source_positions'] != len(source['source_ids'])):
                raise ValueError('full frozen feature identity/shape differs')
            replayed += 1
            overlap = text_index.matches(source['source'])
            why = []
            if source['source_component'] in reserved: why.append('reserved_source_component')
            if overlap: why.append('open_heldout_text_overlap')
            if why:
                exclusions.update(why)
                conflicts.append({'original_line': line_number, 'id': source['id'], 'reasons': why,
                    'open_heldout_ids': [heldout[j]['id'] for j, _ in overlap]})
                continue
            keep = {k: source[k] for k in ('id', 'source', 'source_id', 'group_id', 'source_component',
                'source_sha256', 'spans', 'source_cl100k_tokens', 'alignment_sha256', 'native_prompt_sha256')}
            keep.update(target_text=row['target'], target_sha256=row['target_sha256'],
                stored_tokens=row['target_tokens'], native_lexical_tokens=len(lexical),
                native_word_ids=[101, *lexical, 101] + [0] * (126-len(lexical)),
                target_origin='existing_core_stored_summary_unchanged_unreviewed',
                stored_target_provenance={k: row[k] for k in ('id', 'source_id', 'group_id', 'store', 'section_id', 'spans', 'summarizer_identity')},
                original_physical_line=line_number, semantic_qualified=False)
            result.append(keep)
            subset_index.append(cached)
            manifest[source['id']] = feature
            if replayed % 512 == 0:
                print(json.dumps({'raw_cached_target_bindings_replayed': replayed, 'eligible': len(result)}), flush=True)
    if replayed != 4949 or len({r['source_component'] for r in result}) != len(result):
        raise ValueError('all exact matches and distinct admitted source components required')
    write_json(DEST / 'cases.json', result)
    write_json(DEST / 'source_index.json', subset_index)
    write_json(DEST / 'feature_manifest.json', manifest)
    write_json(DEST / 'source_conflicts.json', conflicts)
    native.vendor.verify(producer)
    state = {'status': 'historical_stored_TRAIN_targets_and_frozen_source_bindings_replayed',
        'replayed': replayed, 'eligible_pairs': len(result), 'distinct_source_components': len(result),
        'exclusions': dict(exclusions), 'reserved_components_excluded': 512, 'open_heldout_sources': 192,
        'targets_unchanged': True, 'labels_individually_semantically_qualified': False,
        'student_training_admitted': False, 'new_requests': 0, 'task_accuracy': None,
        'existing_payloads_verified_at_use': True, 'feature_reextraction': False, 'final_test_opened': False,
        'artifacts': {str(p): file_hash(p) for p in [DEST / n for n in
            ('cases.json', 'source_index.json', 'feature_manifest.json', 'source_conflicts.json', 'producer.json')]}}
    write_json(DEST / 'result.json', state)
    print(json.dumps({k: v for k, v in state.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
