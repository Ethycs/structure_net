"""Stream a unique codec-text corpus with whole-component development exclusion."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sqlite3
import struct

from experiments.structure_net import ncm_recovered_pair_training_pool as pool
from experiments.structure_net import ncm_recovered_label_sample as sample
from experiments.structure_net import ncm_public_training_catalog as catalog

ROOT = pool.BASE/'20261027_recovered_codec_corpus_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-recovered-codec-corpus.md')
MAX_RSS = 4_000_000_000


def packed_frame(tokenizer, text):
    lexical = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    ids = catalog.native_frame(lexical, tokenizer.all_special_ids)
    raw = struct.pack('<128H', *ids)
    return hashlib.sha256(raw).hexdigest(), raw, len(lexical)


def run():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only codec corpus required')
    for parent in (pool.ROOT, sample.ROOT):
        receipt = pool.read(parent/'actual_tool_receipt.json')
        if receipt['actual_tool_exit_code'] != 0:
            raise ValueError('actual completed parent required')
        for path, sha in receipt['artifacts'].items():
            if pool.file_hash(path) != sha: raise ValueError('parent artifact differs')
    audited = pool.BASE/'20261027_recovered_pair_training_pool_audit_v1'
    if pool.read(audited/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('independent parent export readback required')
    result = pool.read(pool.ROOT/'result.json'); export = Path(result['export']['path'])
    if pool.file_hash(export) != result['export']['sha256']:
        raise ValueError('audited review export differs')
    from experiments.structure_net import ncm_pixi
    training, environment = ncm_pixi.configure(); tokenizer = training.native.tokenizer()
    held = pool.read(sample.ROOT/'cases.json')
    components = {r['source_component'] for r in held}
    held_frames = {packed_frame(tokenizer, r['target_text'])[0] for r in held}
    if len(held) != 128 or len(components) != 128:
        raise ValueError('unchanged complete128-component codec development sample required')
    paths = [Path(__file__), PLAN, Path(pool.__file__), Path(sample.__file__), Path(catalog.__file__),
        Path('pixi.lock'), Path('pyproject.toml'), export,
        *[p/n for p in (pool.ROOT, sample.ROOT, audited) for n in ('result.json', 'actual_tool_receipt.json')],
        sample.ROOT/'cases.json']
    proof = {'schema': 'nal.ncm-recovered-codec-corpus.v1', 'stage': 'unique_native_text_frames',
        'environment': environment, 'inputs': {str(p): pool.file_hash(p) for p in paths},
        'semantic_pair_training_admitted': False, 'codec_development_components': sorted(components),
        'codec_development_frame_hashes': sorted(held_frames), 'max_RSS_bytes': MAX_RSS}
    ROOT.mkdir(); pool.write(ROOT/'producer.json', proof)
    db = sqlite3.connect(ROOT/'frames.sqlite')
    db.executescript('''PRAGMA journal_mode=WAL; PRAGMA cache_size=-65536;
      CREATE TABLE frames(sha TEXT PRIMARY KEY,raw BLOB,target TEXT,lexical INTEGER);
      CREATE TABLE provenance(frame_sha TEXT,pair_id TEXT PRIMARY KEY,component TEXT,target_sha TEXT);
      CREATE INDEX by_frame ON provenance(frame_sha);''')
    counts = Counter()
    with export.open() as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line); counts['observed_pairs'] += 1
            sha, raw, lexical = packed_frame(tokenizer, row['target_text'])
            if lexical != row['native_lexical_tokens']: raise ValueError('native target length differs')
            if row['source_component'] in components:
                counts['excluded_development_component'] += 1; continue
            if sha in held_frames:
                counts['excluded_duplicate_development_frame'] += 1; continue
            inserted = db.execute('INSERT OR IGNORE INTO frames VALUES(?,?,?,?)', (sha, raw, row['target_text'], lexical)).rowcount
            counts['new_unique_frames' if inserted else 'duplicate_native_frames'] += 1
            db.execute('INSERT INTO provenance VALUES(?,?,?,?)', (sha, row['id'], row['source_component'], row['target_sha256']))
            counts['retained_pair_provenance'] += 1
            if number % 10000 == 0:
                db.commit()
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > MAX_RSS: raise ValueError('host memory cap exceeded')
                print(json.dumps({'scanned': number, 'counts': dict(counts)}), flush=True)
    db.commit()
    if counts['observed_pairs'] != 104503: raise ValueError('entire reviewed export required')
    binary, index = ROOT/'frames.uint16.bin', ROOT/'index.jsonl'; count = 0
    with binary.open('xb') as output, index.open('x') as positions:
        for sha, raw, text, lexical in db.execute('SELECT sha,raw,target,lexical FROM frames ORDER BY sha'):
            origins = db.execute('SELECT pair_id,component,target_sha FROM provenance WHERE frame_sha=? ORDER BY pair_id', (sha,)).fetchall()
            if sha in held_frames or any(r[1] in components for r in origins): raise ValueError('codec development overlap')
            row = {'position': count, 'offset_bytes': output.tell(), 'length_bytes': 256,
                'frame_sha256': sha, 'native_lexical_tokens': lexical, 'representative_target_text': text,
                'pair_provenance': origins, 'purpose': 'codec_text_reconstruction', 'semantic_pair_training_admitted': False}
            output.write(raw); positions.write(json.dumps(row, ensure_ascii=False)+'\n'); count += 1
    if count != counts['new_unique_frames'] or binary.stat().st_size != 256*count:
        raise ValueError('all unique frames must be exported')
    checked = 0
    with binary.open('rb') as raw_file, index.open() as entries:
        for line in entries:
            row = json.loads(line)
            if raw_file.tell() != row['offset_bytes'] or row['position'] != checked: raise ValueError('frame order or offset differs')
            raw = raw_file.read(256)
            if hashlib.sha256(raw).hexdigest() != row['frame_sha256']: raise ValueError('readback frame hash differs')
            if packed_frame(tokenizer, row['representative_target_text'])[:2] != (row['frame_sha256'], raw): raise ValueError('text/frame binding differs')
            if row['frame_sha256'] in held_frames or any(r[1] in components for r in row['pair_provenance']): raise ValueError('readback found codec development overlap')
            checked += 1
        if raw_file.read(1) or checked != count: raise ValueError('incomplete readback')
    db.close()
    for path, sha in proof['inputs'].items():
        if pool.file_hash(path) != sha: raise ValueError('input changed during preparation')
    value = {'schema': proof['schema'], 'status': 'unique_codec_corpus_exported_and_read_back',
        'counts': dict(counts), 'frames': count, 'frame_width': 128, 'dtype': 'little_endian_uint16',
        'readback_frames': checked, 'development_components_excluded': 128,
        'development_unique_frames_excluded': len(held_frames), 'model_training': False,
        'new_model_calls': 0, 'semantic_pair_training_admitted': False, 'task_accuracy': None,
        'final_test_opened': False, 'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'artifacts': {str(p): pool.file_hash(p) for p in (ROOT/'producer.json', binary, index)}}
    pool.write(ROOT/'result.json', value); print(json.dumps(value), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args(); existed = ROOT.exists()
    try: run()
    except Exception as error:
        if ROOT.exists() and not existed: pool.write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
