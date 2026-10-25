"""Reserve engineering candidates with informative-turn overlap checks."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess

from experiments.structure_net import ncm_shared_summary_inventory as old
from experiments.structure_net import ncm_shared_summary_source_guard as text_guard
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

native, ROOT = old.native, old.ROOT
DATA = Path('data/experiments/ncm_memory/20261025_broad_handoff_v1')
SHARD = DATA / 'handoff'
DEST = ROOT / '20261025_engineering_summary_inventory_v2'
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-engineering-shard-v2-plan.md')
TEST = Path('tests/structure_net/test_ncm_engineering_summary_inventory_v2.py')
SEED = 'ncm-engineering-components-20261025-v1'


def compressed_rows(path):
    with subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            yield json.loads(line)
        if process.wait() != 0:
            raise ValueError('complete compressed artifact required')


def validate(row, databases):
    p = row['provenance']
    if (row['split'] != 'unassigned' or row['id'] != p['section_receipt_sha256']
            or p['target_sha256'] != text_hash(row['target']['summary'])
            or p['label_origin'] != 'existing_stored_summary'):
        raise ValueError('original stored target/receipt binding differs')
    for key, filename, digest in (
        ('source_bundle', 'memory.db', p['source_database_sha256']),
        ('summary_bundle', p['summary_file'], p['summary_database_sha256'])):
        identity = (p[key], p['store'] + '/' + filename)
        if databases.get(identity) != digest:
            raise ValueError('source or summary DB not bound to frozen artifact index')
    for span in row['evidence']:
        if (not 0 <= span['start_char'] < span['end_char']
                or span['end_char'] - span['start_char'] != len(span['text'])
                or text_hash(span['text']) != span['span_text_sha256']):
            raise ValueError('exact character span/hash binding differs')
    if row['task'] == 'raw_summary_candidate':
        if (row['section_kind'] != 'atomic' or len(row['evidence']) != 1
                or row['input'].get('text') != row['evidence'][0]['text']
                or row['input'].get('turns') != row['evidence']):
            raise ValueError('complete single-fragment raw input differs')


def components(rows):
    """Join session aliases and informative shared text/turns before partitioning."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    seen = {}
    for i, row in enumerate(rows):
        aliases = {old.reservation.graph.guard.source_family(v) for v in
            [row['source_group_id'], *[s['source_id'] for s in row['evidence']]]}
        keys = [('family', v) for v in aliases]
        if row['input_tokens'] >= 192:
            keys += [('source', row['source_sha256'])]
            keys += [('turn', s['turn_text_sha256']) for s in row['evidence']]
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    # Exact normalized/near matches supplement IDs and byte hashes.
    active = [i for i, r in enumerate(rows) if r['input_tokens'] >= 192]
    index = text_guard.TextIndex([rows[i]['input']['text'] for i in active])
    for i in active:
        for j, _ in index.matches(rows[i]['input']['text']):
            union(i, active[j])
    return [find(i) for i in range(len(rows))]


def split_components(names):
    ordered = sorted(set(names), key=lambda v: (text_hash(SEED + ':' + v), v))
    if len(ordered) < 2:
        raise ValueError('at least two independent source components required')
    heldout = set(ordered[:max(1, min(len(ordered)-1, round(.2*len(ordered))))])
    return {name: 'test' if name in heldout else 'train' for name in ordered}


def informative_turn_hashes(candidates, public_turns):
    # Generic public acknowledgements must not block whole source families.
    known_generic = {key for key, value in public_turns.items() if value['tokens'] < 32}
    return set(candidates) - known_generic


def main():
    if DEST.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only engineering inventory required')
    native.vendor.corpus.quality.dependencies()
    manifest = old.reservation.read(SHARD / 'SHA256.json')
    for name in ('engineering-candidates.jsonl.zst', 'engineering-shard-verification.json',
            'source-families.json', 'read_summary_pairs.py', 'summary-artifacts.jsonl.zst'):
        if file_hash(SHARD / name) != manifest[name]['sha256'] or (SHARD / name).stat().st_size != manifest[name]['bytes']:
            raise ValueError('received handoff payload SHA256 or size differs')
    receipt = old.reservation.read(DATA / 'restore_receipt.json')
    if receipt['lakefs_commit'] != 'ded03c25a12e56d620710d2bff334ba5beed923584783c460084eb5922a8270a':
        raise ValueError('pinned immutable publication required')
    protected, metadata_paths, _ = old.protected_metadata()
    whole_index = old.reservation.read(old.reservation.PARENT / 'component_index.json')
    public_path = old.reservation.PARENT / 'public_inventory.json'
    public = old.reservation.read(public_path)
    core_families = {old.reservation.graph.guard.source_family(v['source_id']) for v in public['sessions'].values()}
    core_families |= {f for r in whole_index for f in r['families']}
    core_hashes = {r['source_sha256'] for r in whole_index}
    core_turns = set(public['turns']) | {t for r in whole_index for t in r['informative_turns']}
    excluded_families = core_families | protected['families']
    excluded_hashes = core_hashes | protected['hashes']
    excluded_turns = informative_turn_hashes(core_turns | protected['turns'], public['turns'])
    development = text_guard.development
    if (file_hash(development.SOURCE / 'cases.json') != development.SOURCE_SHA
            or old.reservation.read(development.SOURCE / 'worker_terminal_receipt.json')['exit_code'] != 0):
        raise ValueError('pinned successful open source cases required')
    dev = development.validate_sources(old.reservation.read(development.SOURCE / 'cases.json'))
    heldout = [r for k in text_guard.development.COHORTS for r in dev[k]]
    heldout += old.reservation.read(old.COMPONENT)
    open_index = text_guard.TextIndex([r['source'] for r in heldout])
    files = [Path(__file__), PLAN, TEST, Path(old.__file__), Path(text_guard.__file__),
        Path(old.reservation.graph.guard.__file__)]
    first = ROOT / '20261025_engineering_summary_inventory_v1'
    if old.reservation.read(first / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('completed first overexclusion inventory must be preserved')
    inputs = [*metadata_paths, public_path, DATA / 'restore_receipt.json',
        first / 'result.json', first / 'actual_tool_receipt.json',
        development.SOURCE / 'worker_terminal_receipt.json',
        *[SHARD / n for n in ('SHA256.json', 'engineering-candidates.jsonl.zst',
            'engineering-shard-verification.json', 'source-families.json', 'read_summary_pairs.py', 'summary-artifacts.jsonl.zst')],
        text_guard.development.SOURCE / 'cases.json']
    DEST.mkdir()
    (DEST / 'producer_source').mkdir()
    for i, path in enumerate(files):
        (DEST / 'producer_source' / f'{i}_{path.name}').write_bytes(path.read_bytes())
    producer = {'schema': 'nal.ncm-engineering-inventory.v1', 'source_hashes': {str(p): file_hash(p) for p in files},
        'input_hashes': {str(p): file_hash(p) for p in inputs},
        'source_scope': 'existing stored engineering candidates; whole-core and protected metadata exclusions',
        'config': {'seed': SEED, 'input_cl100k': [192, 2048], 'target_cl100k_max': 128,
            'native_lexical_max': 126, 'test_component_fraction': .2,
            'deduplication': 'first physical raw record for each exact source, before token filtering',
            'grouping': 'source families, informative shared turns/inputs, normalized equality or32/80word5gram containment',
            'informative_public_turn_minimum_cl100k_tokens': 32,
            'targets_changed': False, 'new_requests': 0, 'model_training': False,
            'training_admitted': False, 'sealed_final_sources_opened_or_hashed': False}}
    producer['fingerprint'] = native.q.digest(producer)
    write_json(DEST / 'producer.json', producer)
    databases = {(r['bundle'], r['path']): r['sha256'] for r in compressed_rows(SHARD / 'summary-artifacts.jsonl.zst')
        if r['category'] in ('source_database', 'summary_database')}
    from experiments.structure_net.ncm_additional_training_sources import storage_encoding
    encoding, tokenizer = storage_encoding(), native.tokenizer()
    raw, tasks = [], Counter()
    for number, row in enumerate(compressed_rows(SHARD / 'engineering-candidates.jsonl.zst'), 1):
        validate(row, databases)
        tasks[row['task']] += 1
        if row['task'] == 'raw_summary_candidate':
            source = row['input']['text']
            raw.append({**row, 'physical_line': number, 'source_sha256': text_hash(source),
                'input_tokens': len(encoding.encode(source, disallowed_special=()))})
    if number != 4542 or len(raw) != 3315:
        raise ValueError('complete received engineering shard required')
    roots = components(raw)
    group_rows = defaultdict(list)
    for i, root in enumerate(roots):
        group_rows[root].append(i)
    group_names = {k: min(old.reservation.graph.guard.source_family(raw[i]['source_group_id']) for i in v)
        for k, v in group_rows.items()}
    blocked, conflicts = set(), []
    for i, row in enumerate(raw):
        families = {old.reservation.graph.guard.source_family(v) for v in
            [row['source_group_id'], *[s['source_id'] for s in row['evidence']]]}
        why = []
        if families & excluded_families: why.append('core_or_protected_family')
        if row['source_sha256'] in excluded_hashes: why.append('core_or_protected_source_hash')
        if {s['turn_text_sha256'] for s in row['evidence']} & excluded_turns: why.append('core_or_protected_turn')
        matches = open_index.matches(row['input']['text']) if row['input_tokens'] >= 192 else []
        if matches: why.append('open_heldout_text_overlap')
        if why:
            blocked.add(roots[i]);conflicts.append({'physical_line': row['physical_line'], 'reasons': why,
                'open_heldout_ids': [heldout[j]['id'] for j, _ in matches]})
    split = split_components([v for k, v in group_names.items() if k not in blocked])
    seen, selected, exclusions = set(), {'train': [], 'test': []}, Counter()
    for i, row in enumerate(raw):
        if row['source_sha256'] in seen:
            exclusions['exact_source_duplicate'] += 1;continue
        seen.add(row['source_sha256'])
        if roots[i] in blocked:
            exclusions['blocked_source_component'] += 1;continue
        text = row['target']['summary']
        lexical = tokenizer.encode(text, add_special_tokens=False)
        stored = len(encoding.encode(text, disallowed_special=()))
        why = []
        if row['input_tokens'] < 192: why.append('application_bypass')
        if row['input_tokens'] > 2048: why.append('source_above2048')
        if not text.strip() or not 0 < stored <= 128: why.append('target_empty_or_above128')
        if not 0 < len(lexical) <= 126: why.append('native_frame_overflow_or_empty')
        if set(lexical) & set(tokenizer.all_special_ids): why.append('native_special_or_UNK')
        if why:
            exclusions.update(why);continue
        group = group_names[roots[i]]
        selected[split[group]].append({'id': text_hash('engineering:' + row['source_sha256']),
            'source': row['input']['text'], 'source_sha256': row['source_sha256'],
            'target_text': text, 'target_sha256': row['provenance']['target_sha256'],
            'source_id': row['source_group_id'], 'source_component': group, 'group_id': group,
            'physical_line': row['physical_line'], 'input_tokens': row['input_tokens'],
            'stored_tokens': stored, 'native_lexical_tokens': len(lexical),
            'native_word_ids': [101, *lexical, 101] + [0] * (126-len(lexical)),
            'spans': row['evidence'], 'provenance': row['provenance'],
            'target_origin': 'existing_stored_summary_unchanged_unreviewed', 'semantic_qualified': False})
    for key in ('source_sha256', 'source_component', 'source_id'):
        if {r[key] for r in selected['train']} & {r[key] for r in selected['test']}:
            raise ValueError('source split overlap')
    write_json(DEST / 'cases.json', selected)
    write_json(DEST / 'component_partitions.json', {'partitions': split, 'blocked_components': [group_names[k] for k in blocked],
        'components': {group_names[k]: {'raw_records': len(v)} for k, v in group_rows.items()}})
    write_json(DEST / 'source_conflicts.json', conflicts)
    native.vendor.verify(producer)
    result = {'status': 'engineering_shard_locally_bound_and_candidate_partitions_reserved',
        'shard_records': number, 'task_counts': dict(tasks), 'raw_records': len(raw), 'unique_raw_sources': len(seen),
        'source_components_before_exclusions': len(group_rows), 'blocked_source_components': len(blocked),
        'core_source_families_excluded': len(core_families), 'core_turn_hashes_excluded': len(excluded_turns),
        'core_turn_hashes_in_inventory': len(core_turns),
        'candidate_pairs': {k: len(v) for k, v in selected.items()},
        'eligible_source_components': {k: len({r['source_component'] for r in v}) for k, v in selected.items()},
        'exclusion_counts_can_overlap': dict(exclusions), 'all_target_text_unchanged': True,
        'span_text_hashes_verified_locally': True, 'database_hashes_bound_to_frozen_index': True,
        'full_turn_bytes_independently_restored': False, 'labels_individually_semantically_qualified': False,
        'training_admitted': False, 'model_training': False, 'new_requests': 0,
        'sealed_final_sources_opened_or_hashed': False, 'task_accuracy': None,
        'cases_sha256': file_hash(DEST / 'cases.json')}
    write_json(DEST / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
