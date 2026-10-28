"""Combine prepared TRAIN catalogs by content while retaining every original record."""
import hashlib
import json
from pathlib import Path
import struct

BASE = Path('data/experiments/ncm_paper_regime')
ROOT = BASE/'20261028_combined_prepared_pair_inventory_v1'
PUBLIC = BASE/'20261026_public_training_catalog_v1'
RECOVERED = BASE/'20261028_recovered_paraphrase_catalog_v1'
ANCHORS = BASE/'20261027_expanded_teacher_catalog_prepare_v3'
TEST = Path('tests/structure_net/test_ncm_combined_pair_inventory.py')
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-combined-prepared-pair-inventory.md')
CPU = BASE/'20261028_combined_pair_inventory_CPU_v1.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def completed(root):
    receipt = read(root/'actual_tool_receipt.json')
    if (receipt['actual_tool_exit_code'] != 0
            or receipt['artifacts'].get(str(root/'result.json')) != sha(root/'result.json')):
        raise ValueError('completed result binding required')
    return read(root/'result.json')


def add(groups, row, domain, pointer):
    if (row.get('split') not in ('TRAIN', 'TRAIN_noisy_candidate')
            or not 1 <= row['stored_tokens'] <= 128
            or not 1 <= row['native_lexical_tokens'] <= 126
            or len(row['native_word_ids']) != 128):
        raise ValueError('complete original TRAIN target within both token limits required')
    for name in ('source_sha256', 'target_sha256'):
        if len(row[name]) != 64 or any(c not in '0123456789abcdef' for c in row[name]):
            raise ValueError('exact source/target SHA256 identities required')
    if hashlib.sha256(row['target_text'].encode()).hexdigest() != row['target_sha256']:
        raise ValueError('target text binding differs')
    if 'source' in row and hashlib.sha256(row['source'].encode()).hexdigest() != row['source_sha256']:
        raise ValueError('source text binding differs')
    words, n = row['native_word_ids'], row['native_lexical_tokens']
    if (any(type(x) is not int or not 0 <= x < 30522 for x in words)
            or words[0] != 101 or words[n+1] != 101 or any(words[n+2:])
            or any(x in (0, 100, 101, 102, 103) for x in words[1:n+1])):
        raise ValueError('complete native frame required')
    frame = hashlib.sha256(struct.pack('<128H', *words)).hexdigest()
    key = (row['source_sha256'], row['target_sha256'])
    identity = hashlib.sha256((':'.join(key)).encode()).hexdigest()
    group = groups.setdefault(key, {'id': identity, 'source_sha256': key[0],
        'target_sha256': key[1], 'native_frame_sha256_uint16': frame,
        'stored_tokens': row['stored_tokens'], 'native_lexical_tokens': n, 'origins': []})
    if (group['native_frame_sha256_uint16'] != frame or group['stored_tokens'] != row['stored_tokens']):
        raise ValueError('same exact pair has inconsistent target representation')
    group['origins'].append({'domain': domain, 'original_id': row['id'],
        'source_component': row['source_component'], 'original_split': row['split'],
        'source_record_id': row.get('source_record_id'), **pointer})


def run():
    if ROOT.exists():
        raise ValueError('fresh combined inventory required')
    cpu = read(CPU)
    if cpu['actual_tool_exit_code'] != 0 or cpu['tests_passed'] != 3:
        raise ValueError('successful union and split contracts required')
    for p, digest in cpu['artifacts'].items():
        if sha(p) != digest:
            raise ValueError('qualified inventory source changed')
    configs = [('public_xsum', PUBLIC, 'training.jsonl', 162353),
               ('recovered_conversation', RECOVERED, 'pairs.jsonl', 43894),
               ('reviewed_anchor', ANCHORS, 'cases.json', 2384)]
    audits = [BASE/'20261026_public_training_catalog_audit_v1',
              BASE/'20261028_recovered_paraphrase_training_prepare_v1',
              BASE/'20261027_expanded_teacher_catalog_audit_v3']
    for root in audits:
        state = completed(root)
        if not (state.get('training_data_admitted') or state.get('data_admitted')):
            raise ValueError('each original data catalog must have completed admission')
    inputs = [Path(__file__).resolve(), TEST, PLAN, CPU, Path('pixi.lock')]
    inputs += [root/name for root in [*(c[1] for c in configs), *audits]
               for name in ('result.json', 'actual_tool_receipt.json')]
    for _, root, name, _ in configs:
        state = completed(root)
        path = root/name
        if state['artifacts'].get(str(path)) != sha(path):
            raise ValueError('original complete catalog payload changed')
        inputs.append(path)
    # Retain access to the recovered source/context records without flattening them.
    state = completed(RECOVERED)
    for name in ('sources.jsonl', 'sources_index.jsonl'):
        path = RECOVERED/name
        if state['artifacts'].get(str(path)) != sha(path):
            raise ValueError('original recovered source catalog changed')
        inputs.append(path)
    proof = {'stage': 'prepared_TRAIN_catalog_union_only',
        'config': {'exact_pair_key': ['source_sha256', 'target_sha256'],
            'retain_every_original_record_and_context': True, 'new_split_assignment': False,
            'new_training_admitted': False, 'model_calls': 0, 'judge_calls': 0,
            'final_test_opened': False}, 'inputs': {str(p): sha(p) for p in inputs}}
    ROOT.mkdir()
    write(ROOT/'producer.json', proof)
    groups, counts, domain_sources = {}, {}, {}
    for domain, root, name, expected in configs:
        path = root/name
        count, sources = 0, set()
        if name.endswith('.jsonl'):
            with path.open('rb') as stream:
                while True:
                    offset = stream.tell()
                    payload = stream.readline()
                    if not payload:
                        break
                    row = json.loads(payload)
                    pointer = {'path': str(path), 'format': 'jsonl', 'offset_bytes': offset,
                        'length_bytes': len(payload), 'record_sha256': hashlib.sha256(payload).hexdigest()}
                    add(groups, row, domain, pointer)
                    count += 1
                    sources.add(row['source_sha256'])
        else:
            for i, row in enumerate(read(path)):
                add(groups, row, domain, {'path': str(path), 'format': 'json_array', 'index': i})
                count += 1
                sources.add(row['source_sha256'])
        if count != expected:
            raise ValueError('complete original domain population required')
        counts[domain] = count
        domain_sources[domain] = sources
    with (ROOT/'pairs_index.jsonl').open('x') as stream:
        for row in groups.values():
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    all_sources = set().union(*domain_sources.values())
    overlap = {a+'__'+b: len(domain_sources[a] & domain_sources[b])
               for i, a in enumerate(domain_sources) for b in list(domain_sources)[i+1:]}
    for p, digest in proof['inputs'].items():
        if sha(p) != digest:
            raise ValueError('input changed during inventory')
    value = {'status': 'prepared_catalog_union_indexed', 'input_rows': counts,
        'total_original_rows': sum(counts.values()), 'distinct_content_pairs': len(groups),
        'duplicate_content_pair_rows': sum(counts.values())-len(groups),
        'distinct_source_texts': len(all_sources),
        'domain_distinct_source_texts': {k: len(v) for k, v in domain_sources.items()},
        'cross_domain_exact_source_overlap': overlap,
        'distinct_native_target_frames': len({v['native_frame_sha256_uint16'] for v in groups.values()}),
        'all_original_records_preserved': sum(len(v['origins']) for v in groups.values()) == sum(counts.values()),
        'training_admitted': False, 'union_near_duplicate_split_audit_completed': False,
        'summary_quality_rechecked': False, 'new_training_or_model_calls': 0,
        'new_judge_calls': 0, 'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(ROOT/n): sha(ROOT/n) for n in ('producer.json', 'pairs_index.jsonl')}}
    write(ROOT/'result.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'artifacts'}))


if __name__ == '__main__':
    existed = ROOT.exists()
    try:
        run()
    except Exception as error:
        if ROOT.exists() and not existed and not (ROOT/'result.json').exists() and not (ROOT/'failure.json').exists():
            write(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
