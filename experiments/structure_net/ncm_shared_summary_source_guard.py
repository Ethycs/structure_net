"""Replay shared TRAIN candidates and exclude text overlap with open cohorts."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import resource

import torch

from experiments.structure_net import ncm_shared_summary_inventory as inventory
from experiments.structure_net import ncm_lm1b_native_development as development
from experiments.structure_net import ncm_source_split_integrity as integrity

native, ROOT = inventory.native, inventory.ROOT
HYPOTHESIS = 'ncm-shared-summary-open-source-guard-v1'
DEST = ROOT / '20261025_shared_summary_open_source_guard_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-shared-summary-source-guard-plan.md')
TEST = Path('tests/structure_net/test_ncm_shared_summary_source_guard.py')


class TextIndex:
    def __init__(self, texts):
        self.postings, self.exact, self.sizes = defaultdict(list), defaultdict(list), []
        for i, text in enumerate(texts):
            words = integrity.normalized_words(text)
            grams = {words[j:j + 5] for j in range(max(0, len(words) - 4))}
            self.sizes.append(len(grams))
            if words:
                self.exact[words].append(i)
            for gram in grams:
                self.postings[gram].append(i)

    def matches(self, text):
        words = integrity.normalized_words(text)
        grams = {words[j:j + 5] for j in range(max(0, len(words) - 4))}
        counts = Counter(j for gram in grams for j in self.postings.get(gram, ()))
        found = {}
        for j, common in counts.items():
            minimum = min(len(grams), self.sizes[j])
            if common >= 32 and common * 5 >= minimum * 4:
                found[j] = {'normalized_equal': False, 'common_5grams': common, 'smaller_5grams': minimum}
        for j in self.exact.get(words, ()):
            found[j] = {'normalized_equal': True, 'common_5grams': len(grams), 'smaller_5grams': len(grams)}
        return sorted(found.items())


def main():
    if (DEST.exists() or torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or inventory.reservation.read(inventory.DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('fresh CPU-only audit after actual completed inventory required')
    native.vendor.corpus.quality.dependencies()
    producer = inventory.reservation.read(inventory.DEST / 'producer.json')
    native.vendor.verify(producer)
    original = inventory.reservation.read(inventory.DEST / 'result.json')
    candidates = inventory.DEST / 'eligible_train_metadata.jsonl'
    if (original['eligible_after_metadata_guards'] != 73836
            or inventory.file_hash(candidates) != original['eligible_metadata_sha256']):
        raise ValueError('complete unchanged candidate population required')
    selected = {}
    with candidates.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['original_line_number'] in selected:
                raise ValueError('unique original physical lines required')
            selected[row['original_line_number']] = row
    if len(selected) != 73836:
        raise ValueError('all original candidates required')
    protected, guard_paths, _ = inventory.protected_metadata()
    dev, _ = development.source_inputs()
    heldout = [r for name in development.COHORTS for r in dev[name]]
    heldout += inventory.reservation.read(inventory.COMPONENT)
    if len(heldout) != 192 or len({r['id'] for r in heldout}) != 192:
        raise ValueError('all128 DEV plus64 component sources required')
    index = TextIndex([r['source'] for r in heldout])
    files = [Path(__file__), PLAN, TEST, Path(inventory.__file__), Path(integrity.__file__)]
    inputs = [*guard_paths, candidates, inventory.DATA / 'train.jsonl',
        inventory.DEST / 'producer.json', inventory.DEST / 'result.json', inventory.DEST / 'actual_tool_receipt.json',
        development.SOURCE / 'cases.json']
    p = native.vendor.freeze(DEST, 'all_candidate_TRAIN_bindings_and_open_text_overlap', [*files, *inputs], {
        'hypothesis_id': HYPOTHESIS, 'candidate_pairs': 73836, 'open_heldout_sources': 192,
        'source_text_rule': 'normalized equality or>=32common_distinct5grams with>=80percent_shorter_containment',
        'deduplication': 'exact source hash; first original TRAIN pair retained',
        'target_changes': False, 'final_source_files_opened_or_hashed': False,
        'sealed_final_full_text_overlap_audit_completed': False, 'model_training': False,
        'new_judge_requests': 0, 'task_accuracy': None, 'speed_tested': False})
    for i, path in enumerate(files):
        (DEST / 'producer_source' / f'shared_guard_{i}_{path.name}').write_bytes(path.read_bytes())
    tokenizer = native.tokenizer()
    reviewed = conflicts = duplicates = retained = 0
    seen_hashes, groups, sessions, targets = set(), set(), set(), set()
    paths = (DEST / 'training_metadata.jsonl', DEST / 'text_conflicts.jsonl')
    with (inventory.DATA / 'train.jsonl').open() as source, paths[0].open('x') as output, paths[1].open('x') as errors:
        for number, line in enumerate(source, 1):
            meta = selected.get(number)
            if meta is None:
                continue
            row = json.loads(line)
            ids = tokenizer.encode(row['target'], add_special_tokens=False)
            if (row['split'] != 'train' or any(row[k] != v for k, v in meta.items()
                    if k not in ('original_line_number', 'native_target_tokens'))
                    or inventory.text_hash(row['source']) != meta['source_sha256']
                    or inventory.text_hash(row['target']) != meta['target_sha256']
                    or len(ids) != meta['native_target_tokens']
                    or inventory.reasons(row, ids, protected, set(tokenizer.all_special_ids))):
                raise ValueError('an accepted TRAIN binding, native target or protected metadata guard differs')
            reviewed += 1
            matches = index.matches(row['source'])
            if matches:
                conflicts += 1
                errors.write(json.dumps({'train_id': row['id'], 'conflicts': [
                    {'heldout_id': heldout[j]['id'], **proof} for j, proof in matches]}) + '\n')
            elif row['source_sha256'] in seen_hashes:
                duplicates += 1
            else:
                retained += 1
                seen_hashes.add(row['source_sha256'])
                groups.add(row['group_id'])
                sessions.add(row['source_id'])
                targets.add(row['target_sha256'])
                output.write(json.dumps(meta, ensure_ascii=False, allow_nan=False) + '\n')
            if reviewed % 10000 == 0:
                print(json.dumps({'candidate_bindings_replayed': reviewed, 'text_conflicts': conflicts,
                    'exact_source_duplicates': duplicates, 'retained': retained}), flush=True)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8_000_000_000:
                raise ValueError('source-guard replay exceeds8GB host RSS')
    if reviewed != 73836 or reviewed != conflicts + duplicates + retained:
        raise ValueError('all candidates and disjoint exclusion accounting required')
    native.vendor.verify(p)
    result = {'status': 'all_shared_TRAIN_candidates_replayed_and_open_source_text_guarded',
        'hypothesis_id': HYPOTHESIS, 'reviewed': reviewed, 'text_overlap_exclusions': conflicts,
        'exact_source_duplicate_exclusions': duplicates, 'retained_unique_source_pairs': retained,
        'source_sessions': len(sessions), 'original_groups': len(groups), 'unique_target_texts': len(targets),
        'open_heldout_sources': 192, 'targets_unchanged': True,
        'training_metadata_sha256': inventory.file_hash(paths[0]), 'conflicts_sha256': inventory.file_hash(paths[1]),
        'sealed_final_full_text_overlap_audit_completed': False, 'final_source_files_opened_or_hashed': False,
        'training_hyperparameters_fixed': False, 'model_training': False, 'new_judge_requests': 0,
        'task_accuracy': None, 'speed_tested': False,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    inventory.write_json(DEST / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
