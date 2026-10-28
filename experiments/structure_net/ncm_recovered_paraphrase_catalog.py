"""Build indexed conversational supervision without flattening role/date variants."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sqlite3
import struct
import time

from experiments.structure_net import ncm_recovered_pair_context as context
from experiments.structure_net import ncm_recovered_pair_training_pool as pool

ROOT = pool.BASE/'20261028_recovered_paraphrase_catalog_v1'
SAMPLE = pool.BASE/'20261027_recovered_label_sample_v1'
AUDIT = pool.BASE/'20261027_recovered_pair_training_pool_audit_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-paraphrase-catalog.md')
TEST = Path('tests/structure_net/test_ncm_recovered_paraphrase_catalog.py')
SCHEMA = 'nal.ncm-recovered-paraphrase-catalog.v1'
MAX_RSS = 4_000_000_000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def frame(tokenizer, text):
    lexical = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if (not 1 <= len(lexical) <= 126 or set(lexical) & set(tokenizer.all_special_ids)
            or any(type(v) is not int or not 0 <= v < 30522 for v in lexical)):
        raise ValueError('complete native target without special or unknown tokens required')
    ids = [101, *lexical, 101]+[0]*(126-len(lexical))
    return ids, hashlib.sha256(struct.pack('<128H', *ids)).hexdigest(), len(lexical)


def original_context(records):
    roles = {r['role'] for r in records if r['role'] is not None}
    dates = {r['created_at'] for r in records if r['created_at'] is not None}
    reasons = []
    if not roles: reasons.append('missing_original_role')
    elif len(roles) != 1: reasons.append('conflicting_original_roles')
    elif not roles <= {'user', 'assistant'}: reasons.append('unsupported_original_role')
    if len(dates) > 1: reasons.append('conflicting_original_dates')
    return (next(iter(roles)) if len(roles) == 1 else None,
            next(iter(dates)) if len(dates) == 1 else None, reasons)


def source_record(row, role, created_at):
    """An allowlist: neither target variants nor their provenance affect the input."""
    binding = {'source_sha256': row['source_sha256'], 'source_component': row['source_component'],
               'role': role, 'created_at': created_at}
    identity = digest(binding)
    return {'id': identity, 'source_id': identity, 'group_id': row['source_component'],
        'source_component': row['source_component'], 'source_sha256': row['source_sha256'],
        'source': row['source'], 'source_cl100k_tokens': row['source_cl100k_tokens'],
        'source_truncated': False, 'split': 'TRAIN_noisy_candidate',
        'original_role': role, 'original_created_at': created_at,
        'spans': [{'source_id': identity, 'role': role, 'created_at': created_at,
            'span_text_sha256': row['source_sha256'], 'start_char': 0, 'end_char': len(row['source'])}]}


def exclusion_reasons(row, records, held_components, frame_sha, held_frames):
    role, date, reasons = original_context(records)
    if row['target_text'] in row['source']: reasons.append('exact_substring_target')
    if row['source_component'] in held_components: reasons.append('reserved_diagnostic_component')
    if frame_sha in held_frames: reasons.append('reserved_diagnostic_native_frame')
    return role, date, reasons


def receipt_paths(root):
    receipt = pool.read(root/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 0: raise ValueError('actual completed parent required')
    for path, expected in receipt.get('artifacts', {}).items():
        if pool.file_hash(path) != expected: raise ValueError('parent receipt artifact changed')
    result = pool.read(root/'result.json')
    for path, expected in result.get('artifacts', {}).items():
        if pool.file_hash(path) != expected: raise ValueError('parent result artifact changed')
    return [root/'actual_tool_receipt.json', root/'result.json']


def write_record(stream, index, row):
    payload = (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n').encode()
    entry = {'id': row['id'], 'offset_bytes': stream.tell(), 'length_bytes': len(payload),
             'record_sha256': hashlib.sha256(payload).hexdigest()}
    stream.write(payload); index.write(json.dumps(entry)+'\n')


def readback(path, index, expected_count):
    count = 0
    with path.open('rb') as raw, index.open() as entries:
        for entry in map(json.loads, entries):
            if entry['offset_bytes'] != raw.tell(): raise ValueError('noncontiguous JSONL offset')
            payload = raw.read(entry['length_bytes'])
            if hashlib.sha256(payload).hexdigest() != entry['record_sha256']:
                raise ValueError('JSONL bytes differ')
            row = json.loads(payload)
            if row['id'] != entry['id']: raise ValueError('JSONL identity differs')
            count += 1
        if raw.read(1) or count != expected_count: raise ValueError('incomplete indexed population')
    return count


def run():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only catalog required')
    from experiments.structure_net import ncm_pixi
    training, environment = ncm_pixi.configure()
    tokenizer = training.native.tokenizer()
    import tiktoken
    storage = tiktoken.get_encoding('cl100k_base')
    inputs = []
    for parent in (pool.ROOT, AUDIT, context.ROOT, SAMPLE): inputs += receipt_paths(parent)
    export = pool.ROOT/'review_candidates_train.jsonl'
    if pool.file_hash(export) != context.POOL_SHA: raise ValueError('complete guarded pool changed')
    held = pool.read(SAMPLE/'cases.json')
    components = {r['source_component'] for r in held}
    held_frames = {frame(tokenizer, r['target_text'])[1] for r in held}
    if len(held) != 128 or len(components) != 128: raise ValueError('complete fixed diagnostic sample required')
    inputs += [export, context.ROOT/'contexts.sqlite', SAMPLE/'cases.json', Path('pixi.lock'), Path('pyproject.toml')]
    inputs += sorted(p for p in (training.native.INVENTORY/'bert-tokenizer').iterdir() if p.is_file())
    sources = [Path(__file__), PLAN, TEST, Path(context.__file__), Path(pool.__file__), Path(ncm_pixi.__file__),
               Path(training.native.__file__)]
    proof = {'schema': SCHEMA, 'environment': environment,
        'source_hashes': {str(p): pool.file_hash(p) for p in sources},
        'input_hashes': {str(p): pool.file_hash(p) for p in dict.fromkeys(inputs)},
        'config': {'all_candidates': 104503, 'whole_diagnostic_components_excluded': sorted(components),
            'held_native_frames': sorted(held_frames), 'max_RSS_bytes': MAX_RSS,
            'target_variant_policy': 'retain_all; future uniform_source_sampling_with_deterministic_variant_rotation',
            'conditioning': 'exact_source_original_role_original_timestamp',
            'training_admitted': False, 'semantic_quality_confirmed': False}}
    ROOT.mkdir(); (ROOT/'producer_source').mkdir()
    for i, path in enumerate(sources): (ROOT/'producer_source'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    pool.write(ROOT/'producer.json', proof)
    db = sqlite3.connect(ROOT/'catalog.sqlite')
    db.executescript('''PRAGMA journal_mode=WAL; PRAGMA cache_size=-65536;
      CREATE TABLE sources(id TEXT PRIMARY KEY,source_sha TEXT,component TEXT,role TEXT,created_at TEXT,record TEXT);
      CREATE TABLE pairs(id TEXT PRIMARY KEY,source_id TEXT,source_sha TEXT,target_sha TEXT,record TEXT);
      CREATE INDEX source_hash ON sources(source_sha); CREATE INDEX pair_source ON pairs(source_sha);
      CREATE TABLE exclusions(id TEXT PRIMARY KEY,reasons TEXT);''')
    ctx = context.connect(context.ROOT/'contexts.sqlite')
    counts = Counter(); reasons_count = Counter(); started = time.monotonic()
    with export.open() as rows, (ROOT/'excluded.jsonl').open('x') as excluded:
        for row in map(json.loads, rows):
            counts['observed'] += 1
            if (row['training_admitted'] or row['split'] != 'TRAIN_review_candidate'
                    or pool.text_hash(row['source']) != row['source_sha256']
                    or pool.text_hash(row['target_text']) != row['target_sha256']):
                raise ValueError('unaltered complete unadmitted pair required')
            n = len(storage.encode(row['source'], disallowed_special=()))
            t = len(storage.encode(row['target_text'], disallowed_special=()))
            ids, frame_sha, lexical = frame(tokenizer, row['target_text'])
            if (n != row['source_cl100k_tokens'] or t != row['stored_target_tokens']
                    or lexical != row['native_lexical_tokens'] or not 192 <= n <= 2048 or not 1 <= t <= 128):
                raise ValueError('full source/target token bounds differ')
            binding = ctx.execute('SELECT source_sha,target_sha,component,substring FROM pairs WHERE id=?', (row['id'],)).fetchone()
            if binding != (row['source_sha256'], row['target_sha256'], row['source_component'], int(row['target_text'] in row['source'])):
                raise ValueError('original metadata pair binding differs')
            records = [dict(zip(('role','created_at','identity','request_sha','origin','record','provenance'), r))
                for r in ctx.execute('SELECT role,created_at,identity,request_sha,origin,record,provenance FROM contexts WHERE pair_id=? ORDER BY origin,record', (row['id'],))]
            role, date, reasons = exclusion_reasons(row, records, components, frame_sha, held_frames)
            if reasons:
                counts['excluded'] += 1; reasons_count.update(reasons)
                db.execute('INSERT INTO exclusions VALUES(?,?)', (row['id'], json.dumps(reasons)))
                excluded.write(json.dumps({'id': row['id'], 'reasons': reasons})+'\n')
            else:
                source = source_record(row, role, date)
                source_json = json.dumps(source, ensure_ascii=False, sort_keys=True, allow_nan=False)
                old = db.execute('SELECT record FROM sources WHERE id=?', (source['id'],)).fetchone()
                if old and old[0] != source_json: raise ValueError('conditioned source identity collision')
                db.execute('INSERT OR IGNORE INTO sources VALUES(?,?,?,?,?,?)',
                    (source['id'], row['source_sha256'], row['source_component'], role, date, source_json))
                target = {k: row[k] for k in ('id','source_sha256','source_component','target_sha256','target_text','source_proof_references')}
                target.update(source_record_id=source['id'], stored_tokens=t, native_lexical_tokens=lexical,
                    native_word_ids=ids, native_frame_sha256=frame_sha, original_context_records=records,
                    split='TRAIN_noisy_candidate', reference_kind='historical_paraphrase_not_factual_gold',
                    training_admitted=False, semantic_quality_confirmed=False)
                db.execute('INSERT INTO pairs VALUES(?,?,?,?,?)', (row['id'], source['id'], row['source_sha256'],
                    row['target_sha256'], json.dumps(target, ensure_ascii=False, allow_nan=False)))
                counts['retained_pairs'] += 1
            if counts['observed'] % 10000 == 0:
                db.commit()
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > MAX_RSS: raise ValueError('host RSS cap exceeded')
                print(json.dumps({'counts': dict(counts), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    ctx.close(); db.commit()
    if counts['observed'] != 104503 or counts['excluded']+counts['retained_pairs'] != counts['observed']:
        raise ValueError('complete candidate accounting required')
    pair_count, source_count = [db.execute('SELECT count(*) FROM '+table).fetchone()[0] for table in ('pairs','sources')]
    for table, count in (('pairs', pair_count), ('sources', source_count)):
        path, index = ROOT/(table+'.jsonl'), ROOT/(table+'_index.jsonl')
        with path.open('xb') as out, index.open('x') as idx:
            for raw, in db.execute('SELECT record FROM '+table+' ORDER BY id'):
                write_record(out, idx, json.loads(raw))
        readback(path, index, count)
    if db.execute('SELECT count(*) FROM pairs p LEFT JOIN sources s ON p.source_id=s.id WHERE s.id IS NULL OR p.source_sha!=s.source_sha').fetchone()[0]:
        raise ValueError('incomplete source/target join')
    distinct = groups = 0
    with (ROOT/'source_groups.jsonl').open('x') as out:
        for sh, in db.execute('SELECT DISTINCT source_sha FROM sources ORDER BY source_sha'):
            variants = [{'pair_id': p, 'source_record_id': s} for p,s in db.execute('SELECT id,source_id FROM pairs WHERE source_sha=? ORDER BY id',(sh,))]
            component_set = {r[0] for r in db.execute('SELECT DISTINCT component FROM sources WHERE source_sha=?',(sh,))}
            if len(component_set) != 1: raise ValueError('same source text crosses components')
            out.write(json.dumps({'source_sha256': sh, 'source_component': next(iter(component_set)), 'variants': variants})+'\n')
            distinct += 1; groups += len(variants)
    if groups != pair_count: raise ValueError('source grouping lost target variants')
    coverage = {'pairs': pair_count, 'distinct_source_texts': distinct, 'conditioned_sources': source_count,
        'components': db.execute('SELECT count(DISTINCT component) FROM sources').fetchone()[0],
        'conditioned_source_roles': dict(db.execute('SELECT role,count(*) FROM sources GROUP BY role')),
        'conditioned_sources_missing_date': db.execute('SELECT count(*) FROM sources WHERE created_at IS NULL').fetchone()[0],
        'source_texts_with_multiple_dates': db.execute('SELECT count(*) FROM (SELECT source_sha FROM sources GROUP BY source_sha HAVING count(DISTINCT created_at)>1)').fetchone()[0]}
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()
    for collection in (proof['source_hashes'], proof['input_hashes']):
        for path, sha in collection.items():
            if pool.file_hash(path) != sha: raise ValueError('producer input changed during preparation')
    result = {'schema': SCHEMA, 'status': 'guarded_noisy_paraphrases_indexed_and_read_back',
        'counts': dict(counts), 'exclusion_reasons_nonexclusive': dict(reasons_count), 'coverage': coverage,
        'all_byte_offsets_verified': True, 'all_source_target_joins_verified': True,
        'training_admitted': False, 'semantic_quality_confirmed': False, 'model_calls': 0,
        'task_accuracy': None, 'final_test_opened': False, 'elapsed_seconds': time.monotonic()-started,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'artifacts': {str(p): pool.file_hash(p) for p in ROOT.iterdir() if p.is_file()}}
    if result['peak_RSS_bytes'] > MAX_RSS: raise ValueError('host RSS cap exceeded')
    pool.write(ROOT/'result.json', result); print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args(); existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed:
            pool.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
