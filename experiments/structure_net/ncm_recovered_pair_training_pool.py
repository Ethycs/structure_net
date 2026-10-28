"""Trace recovered raw pairs to original sources and protect fixed evaluations.

Produces review candidates, not automatically admitted training labels. The
completed census and existing evaluation reservation remain immutable.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import resource
import sqlite3

from experiments.structure_net.ncm_memory_data import file_hash, text_hash
from experiments.structure_net.ncm_source_split_integrity import source_family

BASE = Path('data/experiments/ncm_paper_regime')
ROOT = BASE/'20261027_recovered_pair_training_pool_v1'
TOTAL = BASE/'20261027_available_pair_total_v3'
PACKAGE = BASE/'20261027_available_pair_source_package_v1'
STAGES = [BASE/'20261027_available_pair_supplement_v1', BASE/'20261027_available_pair_deep_join_v2']
GRAPH = BASE/'20261024_clean_exchange_partitions_v2/component_index.json'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-recovered-pair-training-pool.md')
TEST = Path('tests/structure_net/test_ncm_recovered_pair_training_pool.py')
EXPECTED_EXPORT = '38ff6ca93b5d010c3879d5e18411865299c90be7f41f493a51faebc31fdf6960'
SCHEMA = 'nal.ncm-recovered-training-pool.v1'
MAX_RSS = 8_000_000_000


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    db.execute('BEGIN')
    return db


def check_memory():
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > MAX_RSS:
        raise ValueError('training-pool preparation exceeds 8GB host RSS')


def origin_family(origin):
    if origin.startswith(('longmemeval_s_cleaned.json:', 'longmemeval_oracle.json:')):
        return source_family(origin.split(':', 1)[1].rsplit(':', 1)[0])
    if origin.startswith('locomo:'):
        return ':'.join(origin.split(':')[:2])
    if origin.startswith('build:'):
        return 'original_build_session_8f7f7561'
    raise ValueError('unrecognized original source-package origin')


def pointer_proof(expected_hash, pointers, lookup):
    """Independently reconstruct only from original source-package whole turns."""
    if not isinstance(pointers, list) or not pointers:
        return None
    fragments, aliases = [], set()
    for pointer in pointers:
        if not isinstance(pointer, dict):
            return None
        required = ('turn_text_sha256', 'span_text_sha256', 'start_char', 'end_char')
        if any(k not in pointer for k in required):
            return None
        found = lookup(pointer['turn_text_sha256'])
        if found is None:
            return None
        value, tokens, origins = found
        start, end = pointer['start_char'], pointer['end_char']
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(value)
                or text_hash(value) != pointer['turn_text_sha256']):
            return None
        fragment = value[start:end]
        if text_hash(fragment) != pointer['span_text_sha256']:
            return None
        fragments.append(fragment)
        families = {origin_family(o) for o in origins}
        if tokens >= 192:
            aliases.add(('turn', pointer['turn_text_sha256']))
            aliases.update(('family', f) for f in families)
        else:
            # Generic short turns must not connect unrelated conversations.
            sid = pointer.get('source_id')
            if isinstance(sid, str) and source_family(sid) in families:
                aliases.add(('family', source_family(sid)))
    if text_hash('\n'.join(fragments)) != expected_hash or not aliases:
        return None
    return aliases


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def native_reasons(ids, specials):
    result = []
    if not 1 <= len(ids) <= 126:
        result.append('target_not_one_complete_native128_frame')
    if set(ids) & specials:
        result.append('native_special_or_unknown')
    return result


def schema(db):
    db.executescript('''PRAGMA journal_mode=WAL;
      CREATE TABLE inputs(sha TEXT PRIMARY KEY,source TEXT,tokens INTEGER);
      CREATE TABLE pairs(id TEXT PRIMARY KEY,input_sha TEXT,target_sha TEXT,target TEXT,target_tokens INTEGER);
      CREATE INDEX pair_input ON pairs(input_sha);
      CREATE TABLE aliases(input_sha TEXT,kind TEXT,value TEXT,PRIMARY KEY(input_sha,kind,value));
      CREATE TABLE proofs(input_sha TEXT,kind TEXT,origin TEXT,metadata TEXT,PRIMARY KEY(input_sha,kind,origin));
      CREATE TABLE decisions(input_sha TEXT PRIMARY KEY,component TEXT,reasons TEXT);
      CREATE TABLE outcomes(pair_id TEXT PRIMARY KEY,reasons TEXT,native_tokens INTEGER);
    ''')


def add_proof(db, sha, kind, origin, aliases, metadata):
    db.executemany('INSERT OR IGNORE INTO aliases VALUES(?,?,?)', [(sha, k, v) for k, v in aliases])
    db.execute('INSERT OR IGNORE INTO proofs VALUES(?,?,?,?)',
               (sha, kind, origin, json.dumps(metadata, ensure_ascii=False, sort_keys=True)))


def collect(db):
    expected = read(TOTAL/'candidate_export.json')
    if expected['sha256'] != EXPECTED_EXPORT or expected['pairs'] != 259636:
        raise ValueError('complete fixed recovered export required')
    h = hashlib.sha256(); counts = Counter()
    with Path(expected['path']).open('rb') as source:
        for line in source:
            h.update(line); row = json.loads(line); counts['all_export_rows'] += 1
            if row['training_admitted'] or row['split'] != 'unassigned':
                raise ValueError('inventory must remain unadmitted and unsplit')
            if row['input_kind'] != 'raw':
                counts['merge_rows_deferred'] += 1; continue
            sh, th = text_hash(row['source']), text_hash(row['target'])
            ident = text_hash(json.dumps(['raw', sh, th], separators=(',', ':')))
            if ident != row['pair_id'] or not 192 <= row['input_tokens'] <= 2048 or not 1 <= row['target_tokens'] <= 128:
                raise ValueError('exported pair identity or bounds differ')
            db.execute('INSERT OR IGNORE INTO inputs VALUES(?,?,?)', (sh, row['source'], row['input_tokens']))
            db.execute('INSERT INTO pairs VALUES(?,?,?,?,?)', (ident, sh, th, row['target'], row['target_tokens']))
            counts['raw_pairs'] += 1
    if h.hexdigest() != EXPECTED_EXPORT or counts['all_export_rows'] != 259636:
        raise ValueError('full export hash/count differs')
    db.commit(); inputs = {r[0] for r in db.execute('SELECT sha FROM inputs')}
    package = readonly(PACKAGE/'sources.sqlite')
    origins = defaultdict(list); tokens = {}
    for origin, sha, n in package.execute('''SELECT o.origin,o.sha,t.tokens FROM source_occurrences o
            JOIN source_texts t ON t.sha=o.sha WHERE o.kind='turn' '''):
        origins[sha].append(origin); tokens[sha] = n

    @lru_cache(maxsize=64)
    def lookup(sha):
        if sha not in origins:
            return None
        value = package.execute('SELECT text FROM source_texts WHERE sha=?', (sha,)).fetchone()[0]
        if text_hash(value) != sha:
            raise ValueError('original source text hash differs')
        return value, tokens[sha], origins[sha]

    for origin, sha in package.execute("SELECT origin,sha FROM source_occurrences WHERE kind='fragment'"):
        if sha in inputs:
            add_proof(db, sha, 'original_package_fragment', origin,
                      {('family', origin_family(origin))}, {'source_package_origin': origin})
            counts['exact_package_fragment_proofs'] += 1
    graph = read(GRAPH)
    for row in graph:
        if row['source_sha256'] in inputs and row['public_origins']:
            aliases = {('component', row['source_component'])}
            aliases.update(('family', f) for f in row['families'])
            aliases.update(('turn', t) for t in row['informative_turns'])
            aliases.update(('group', g) for g in row['groups'])
            add_proof(db, row['source_sha256'], 'registered_public_source_graph', row['id'], aliases,
                      {'source_component': row['source_component'], 'public_origins': row['public_origins']})
            counts['exact_graph_source_proofs'] += 1
    del graph
    selections = []
    for stage in STAGES:
        source = readonly(stage/'pairs.sqlite'); digest = hashlib.sha256(); visited = 0
        query = '''SELECT l.id,p.input_sha,l.pointers FROM labels l
          JOIN label_results r ON r.id=l.id JOIN pairs p ON p.id=r.pair_id
          WHERE p.input_kind='raw' AND p.input_tokens BETWEEN 192 AND 2048
            AND p.target_tokens BETWEEN 1 AND 128 AND p.exact_copy=0 ORDER BY l.id'''
        for key, sh, encoded in source.execute(query):
            digest.update((json.dumps([key, sh, encoded], separators=(',', ':'))+'\n').encode()); visited += 1
            if sh not in inputs:
                raise ValueError('source-stage fitting raw input absent from completed union')
            pointers = json.loads(encoded); aliases = pointer_proof(sh, pointers, lookup)
            if aliases is None:
                counts['pointer_proof_not_original_package_verified'] += 1
            else:
                add_proof(db, sh, 'original_package_pointer_reconstruction', stage.name+':'+key, aliases, pointers)
                counts['pointer_proofs_verified'] += 1
            if visited % 25000 == 0:
                db.commit(); check_memory(); print(json.dumps({'stage': stage.name, 'pointer_records': visited, 'counts': counts}), flush=True)
        source.close(); db.commit()
        selections.append({'database': str(stage/'pairs.sqlite'), 'query': query,
                           'rows': visited, 'ordered_selection_sha256': digest.hexdigest()})
    package.close(); lookup.cache_clear()
    counts['distinct_raw_inputs'] = len(inputs)
    counts['inputs_with_original_source_proof'] = db.execute('SELECT COUNT(DISTINCT input_sha) FROM proofs').fetchone()[0]
    value = {'status': 'original_source_provenance_collected', 'counts': dict(counts),
             'input_SQL_selection_bindings': selections, 'training_admitted': False}
    write(ROOT/'provenance_result.json', value); print(json.dumps(value), flush=True)


def guard(db):
    from experiments.structure_net import ncm_shared_summary_inventory as inventory
    from experiments.structure_net import ncm_shared_summary_source_guard as text_guard
    protected, protected_paths, protected_counts = inventory.protected_metadata()
    index = read(GRAPH)
    original_aliases = defaultdict(set); blocked_components = set()
    for row in index:
        component = row['source_component']
        if row['blocked']:
            blocked_components.add(component)
        keys = [('hash', row['source_sha256'])]
        keys += [('family', f) for f in row['families']]
        keys += [('turn', t) for t in row['informative_turns']]
        keys += [('group', g) for g in row['groups']]
        for key in keys:
            original_aliases[key].add(component)
    del index
    components = Components(); first = {}; directly_blocked = set()
    proven = {r[0] for r in db.execute('SELECT DISTINCT input_sha FROM proofs')}
    sources = [r[0] for r in db.execute('SELECT sha FROM inputs ORDER BY sha')]
    protected_map = {'family': protected['families'], 'group': protected['groups'], 'turn': protected['turns'], 'hash': protected['hashes']}

    def link(sha, kind, value):
        key = (kind, value)
        if key in first:
            components.union(sha, first[key])
        else:
            first[key] = sha
        if value in protected_map.get(kind, ()):
            directly_blocked.add(sha)
        for c in original_aliases.get(key, ()):
            components.union(sha, 'old:'+c)
            if c in blocked_components:
                directly_blocked.add(sha)
        if kind == 'component':
            components.union(sha, 'old:'+value)
            if value in blocked_components:
                directly_blocked.add(sha)

    for sha in sources:
        components.find(sha); link(sha, 'hash', sha)
    for sha, kind, value in db.execute('SELECT input_sha,kind,value FROM aliases ORDER BY input_sha,kind,value'):
        link(sha, kind, value)
    cohorts, _ = text_guard.development.source_inputs()
    held = [r for name in text_guard.development.COHORTS for r in cohorts[name]] + read(inventory.COMPONENT)
    if len(held) != 192 or protected_counts['reserved_components'] != 512:
        raise ValueError('all fixed protected source cohorts required')
    text_index = text_guard.TextIndex([r['source'] for r in held])
    text_conflicts = 0
    with (ROOT/'open_heldout_conflicts.jsonl').open('x') as output:
        for i, (sha, source) in enumerate(db.execute('SELECT sha,source FROM inputs ORDER BY sha'), 1):
            matches = text_index.matches(source)
            if matches:
                directly_blocked.add(sha); text_conflicts += 1
                output.write(json.dumps({'source_sha256': sha, 'conflicts': [{'heldout_id': held[j]['id'], **p} for j, p in matches]})+'\n')
            if i % 10000 == 0:
                check_memory(); print(json.dumps({'source_text_guards': i, 'of': len(sources), 'conflicts': text_conflicts}), flush=True)
    blocked_roots = {components.find(s) for s in directly_blocked}
    for sha in sources:
        reasons = []
        if sha not in proven:
            reasons.append('original_source_provenance_unresolved')
        if components.find(sha) in blocked_roots:
            reasons.append('protected_or_previously_blocked_source_component')
        db.execute('INSERT INTO decisions VALUES(?,?,?)', (sha, components.find(sha), json.dumps(reasons)))
    db.commit()
    tokenizer = inventory.native.tokenizer(); special = set(tokenizer.all_special_ids)
    counts = Counter(); retained_sources = set(); retained_components = set()
    output_path = ROOT/'review_candidates_train.jsonl'; digest = hashlib.sha256()
    with output_path.open('xb') as output:
        query = '''SELECT p.id,p.input_sha,p.target_sha,p.target,p.target_tokens,i.source,i.tokens,d.component,d.reasons
          FROM pairs p JOIN inputs i ON i.sha=p.input_sha JOIN decisions d ON d.input_sha=p.input_sha ORDER BY p.id'''
        for i, (pid, sh, th, target, target_tokens, source, source_tokens, component, encoded) in enumerate(db.execute(query), 1):
            reasons = json.loads(encoded); lexical = tokenizer.encode(target, add_special_tokens=False)
            reasons += native_reasons(lexical, special); counts['observed_pairs'] += 1
            db.execute('INSERT INTO outcomes VALUES(?,?,?)', (pid, json.dumps(reasons), len(lexical)))
            counts.update(reasons)
            if not reasons:
                origins = db.execute('SELECT kind,origin FROM proofs WHERE input_sha=? ORDER BY kind,origin', (sh,)).fetchall()
                row = {'id': pid, 'source_sha256': sh, 'target_sha256': th, 'source': source,
                       'target_text': target, 'source_component': component, 'source_cl100k_tokens': source_tokens,
                       'stored_target_tokens': target_tokens, 'native_lexical_tokens': len(lexical),
                       'source_proof_references': origins, 'split': 'TRAIN_review_candidate',
                       'training_admitted': False, 'semantic_label_quality_reviewed': False}
                line = (json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n').encode()
                output.write(line); digest.update(line); counts['retained_pairs'] += 1
                retained_sources.add(sh); retained_components.add(component)
            if i % 10000 == 0:
                db.commit(); check_memory(); print(json.dumps({'native_targets_checked': i, 'counts': counts}), flush=True)
    db.commit()
    if counts['observed_pairs'] != 244587 or db.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0] != 244587:
        raise ValueError('complete recovered raw-pair outcomes required')
    protected_bindings = {str(p): file_hash(p) for p in [*protected_paths, text_guard.development.SOURCE/'cases.json']}
    value = {'schema': SCHEMA, 'status': 'source_guarded_native_fit_review_pool_complete',
             'counts_exclusions_can_overlap': dict(counts), 'retained_distinct_inputs': len(retained_sources),
             'retained_source_components': len(retained_components), 'protected_metadata': protected_counts,
             'open_heldout_text_conflict_inputs': text_conflicts, 'blocked_connected_components': len(blocked_roots),
             'export': {'path': str(output_path), 'sha256': digest.hexdigest(), 'bytes': output_path.stat().st_size},
             'protected_input_bindings': protected_bindings, 'merge_candidates_deferred': 15049,
             'training_admitted': False, 'semantic_label_quality_reviewed': False,
             'native_tokenizer_capacity_checked_not_codec_reconstruction': True,
             'final_source_files_opened_or_hashed': False, 'new_teacher_or_judge_requests': 0,
             'source_split_rule': 'fixed existing protected cohorts; whole connected groups excluded from TRAIN review pool',
             'sealed_final_full_text_overlap_audit_completed': False,
             'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    write(ROOT/'result.json', value); print(json.dumps(value), flush=True)


def main():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only training-pool preparation required')
    inputs = [TOTAL/'result.json', TOTAL/'inventory_audit.json', TOTAL/'candidate_export.json',
              PACKAGE/'producer.json', PACKAGE/'result.json', PACKAGE/'sources.sqlite', GRAPH,
              *[s/'result.json' for s in STAGES]]
    files = [Path(__file__), PLAN, TEST, Path('pixi.lock'), Path('pyproject.toml'),
             Path('experiments/structure_net/ncm_memory_data.py'), Path('experiments/structure_net/ncm_source_split_integrity.py'),
             Path('experiments/structure_net/ncm_shared_summary_inventory.py'), Path('experiments/structure_net/ncm_shared_summary_source_guard.py')]
    proof = {'schema': SCHEMA, 'stage': 'provenance_native_capacity_and_fixed_source_guards',
             'inputs': {str(p): file_hash(p) for p in inputs}, 'producer_files': {str(p): file_hash(p) for p in files},
             'export_sha256': EXPECTED_EXPORT, 'training_admitted': False, 'maximum_RSS_bytes': MAX_RSS}
    ROOT.mkdir(); write(ROOT/'producer.json', proof)
    try:
        db = sqlite3.connect(ROOT/'pool.sqlite'); schema(db); collect(db); guard(db); db.close()
        if any(file_hash(p) != h for p, h in proof['producer_files'].items()):
            raise ValueError('producer changed during preparation')
    except Exception as error:
        write(ROOT/'failure.json', {'error_type': type(error).__name__, 'error': str(error), 'training_admitted': False})
        raise


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args(); main()
