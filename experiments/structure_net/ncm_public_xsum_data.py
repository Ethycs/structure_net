"""Lossless public TRAIN inventory; neither data admission nor model training.

Export: pixi run --locked -e parquet --manifest-path environments/ncm-parquet/pixi.toml export-xsum
Inspect: CUDA_VISIBLE_DEVICES='' pixi run --locked -e ncm python -m experiments.structure_net.ncm_public_xsum_data inspect
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path('data/experiments/ncm_paper_regime')
DOWNLOAD = ROOT/'20261026_public_xsum_train_download_v1'
EXPORT = ROOT/'20261026_public_xsum_train_export_v1'
INSPECT = ROOT/'20261026_public_xsum_train_inspection_v1'
DATA_ENV = Path('environments/ncm-parquet')
ROOT_HASHES = {
    'pyproject.toml': '0f268085bdb851aa6f070c06abae27e9b16e5c69d361267090be61168b1ec31b',
    'pixi.lock': '417b8825845225cd1e6d2d1ad6ae764a837f77d8f1c1443813294a9021b28f5b',
}
EXPECTED_ROWS = 204045
SHARD_SHA = '0bba72d330f2a35ba20321444e54724e487b49f2eab6b26625b5f282d1c797c9'
BOUNDARIES = {'training_data_admitted': False, 'new_model_training': False,
    'new_judge_requests': 0, 'final_test_opened': False, 'task_accuracy': None,
    'label_factual_quality_verified': False}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def verify(hashes):
    for name, expected in hashes.items():
        if sha(name) != expected:
            raise ValueError('input changed: '+str(name))


def completed(root):
    receipt = read(root/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 0:
        raise ValueError('actual successful completion required: '+str(root))
    result = read(root/'result.json')
    verify(receipt.get('artifacts', {}))
    verify(result.get('artifacts', {}))
    return result


def start(root, stage, inputs, environment, config):
    if Path.cwd().resolve() != PROJECT:
        raise ValueError('run from the repository root')
    verify(ROOT_HASHES)
    sources = []
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(PROJECT/'experiments') and path.is_file():
                sources.append(path)
    hashes = {str(p): sha(p) for p in dict.fromkeys([Path(__file__),
        Path('tests/structure_net/test_ncm_public_xsum_data.py'),
        *map(Path, ROOT_HASHES), *inputs, *sources])}
    root.mkdir()
    (root/'producer_source.py').write_bytes(Path(__file__).read_bytes())
    write(root/'producer.json', {'schema': 'nal.ncm-public-xsum-data.v1',
        'stage': stage, 'classification': 'data_preparation_not_model_quality',
        'input_hashes': hashes, 'environment': environment, 'config': config, **BOUNDARIES})
    return hashes


def finish(root, hashes, result):
    verify(hashes)
    result = {**result, **BOUNDARIES,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'artifacts': {str(p): sha(p) for p in sorted(root.iterdir()) if p.is_file()}}
    write(root/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


def validate_row(row):
    if set(row) != {'document', 'summary', 'id'}:
        raise ValueError('unexpected public dataset fields')
    if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
        raise ValueError('incomplete or non-string public example')
    if not row['id'].isascii() or not row['id'].isdigit():
        raise ValueError('expected BBC numeric article identifier')


def export():
    import pyarrow
    import pyarrow.parquet as pq
    prefix = (DATA_ENV/'.pixi/envs/parquet').resolve()
    if (Path(sys.prefix).resolve() != prefix or os.environ.get('PIXI_ENVIRONMENT_NAME') != 'parquet'
            or not Path(pyarrow.__file__).resolve().is_relative_to(prefix)
            or pyarrow.__version__ != '18.1.0'):
        raise ValueError('isolated locked Pixi parquet environment required')
    previous = completed(DOWNLOAD)
    path = DOWNLOAD/'train.parquet'
    if previous['sha256'] != SHARD_SHA or sha(path) != SHARD_SHA or previous['published_expected_rows'] != EXPECTED_ROWS:
        raise ValueError('pinned official TRAIN shard required')
    table = pq.ParquetFile(path)
    if table.metadata.num_rows != EXPECTED_ROWS or set(table.schema_arrow.names) != {'document', 'summary', 'id'}:
        raise ValueError('published complete schema/count required')
    hashes = start(EXPORT, 'lossless_export', [path, DOWNLOAD/'result.json', DOWNLOAD/'actual_tool_receipt.json',
        DATA_ENV/'pixi.toml', DATA_ENV/'pixi.lock'],
        {'python': sys.version, 'prefix': sys.prefix, 'pyarrow': pyarrow.__version__},
        {'batch_rows': 256, 'threads': False, 'original_strings_retained': True,
         'no_truncation_or_target_rewrite': True, 'dataset': 'EdinburghNLP/xsum',
         'revision': previous['revision'], 'split': 'train'})
    ids, sources, summaries = set(), Counter(), Counter()
    total = source_bytes = summary_bytes = 0
    begin = time.monotonic()
    with (EXPORT/'rows.jsonl').open('x') as out:
        for batch in table.iter_batches(batch_size=256, columns=['document', 'summary', 'id'], use_threads=False):
            for row in batch.to_pylist():
                validate_row(row)
                if row['id'] in ids:
                    raise ValueError('duplicate article ID')
                ids.add(row['id'])
                sources[text_sha(row['document'])] += 1
                summaries[text_sha(row['summary'])] += 1
                source_bytes += len(row['document'].encode('utf-8'))
                summary_bytes += len(row['summary'].encode('utf-8'))
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                total += 1
            if total % 32768 == 0:
                print(json.dumps({'stage': 'export', 'rows': total}), flush=True)
    if total != EXPECTED_ROWS:
        raise ValueError('incomplete parsed TRAIN population')
    finish(EXPORT, hashes, {'status': 'complete_lossless_TRAIN_export', 'rows': total,
        'unique_article_ids': len(ids), 'unique_source_strings': len(sources),
        'source_duplicate_extra_rows': sum(n-1 for n in sources.values()),
        'unique_summary_strings': len(summaries),
        'summary_duplicate_extra_rows': sum(n-1 for n in summaries.values()),
        'source_utf8_bytes': source_bytes, 'summary_utf8_bytes': summary_bytes,
        'elapsed_seconds': time.monotonic()-begin, 'parsed_rows_verified': True,
        'source_disjointness_established': False})


def token_reasons(source_tokens, target_tokens, lexical, specials):
    reasons = []
    if source_tokens < 192:
        reasons.append('application_bypass_below192')
    if source_tokens > 2048:
        reasons.append('source_above2048')
    if not 0 < target_tokens <= 128:
        reasons.append('target_outside1to128')
    if not 0 < len(lexical) <= 126:
        reasons.append('native_target_outside1to126')
    if set(lexical) & specials:
        reasons.append('native_special_or_unknown')
    return reasons


def inspect():
    # Existing package/tokenizer and metadata guards are reused without changing them.
    from experiments.structure_net import ncm_pixi
    training, environment = ncm_pixi.configure()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or training.torch.cuda.is_initialized():
        raise ValueError('CPU-only data preparation required')
    import tiktoken
    from experiments.structure_net import ncm_shared_summary_inventory as inventory
    from experiments.structure_net import ncm_shared_summary_source_guard as guard
    from experiments.structure_net import ncm_broad_teacher_catalog as catalog
    previous = completed(EXPORT)
    if previous['rows'] != EXPECTED_ROWS:
        raise ValueError('complete public TRAIN export required')
    protected, paths, counts = inventory.protected_metadata()
    cohorts, _ = guard.development.source_inputs()
    held = [r for name in guard.development.COHORTS for r in cohorts[name]] + read(inventory.COMPONENT)
    if len(held) != 192 or counts['reserved_components'] != 512:
        raise ValueError('all protected open sources and sealed metadata required')
    text_index = guard.TextIndex([r['source'] for r in held])
    train = read(catalog.DEST/'cases.json')
    if len(train) != 1376:
        raise ValueError('complete current TRAIN lineage required')
    train_index = guard.TextIndex([r['source'] for r in train])
    train_hashes = {r['source_sha256'] for r in train}
    tokenizer, encoding = training.native.tokenizer(), tiktoken.get_encoding('cl100k_base')
    specials = set(tokenizer.all_special_ids)
    token_files = [p for p in sorted((training.native.INVENTORY/'bert-tokenizer').iterdir()) if p.is_file()]
    inputs = [EXPORT/'rows.jsonl', EXPORT/'result.json', EXPORT/'actual_tool_receipt.json',
        *paths, guard.development.SOURCE/'cases.json', catalog.DEST/'cases.json', *token_files,
        Path(inventory.__file__), Path(guard.__file__), Path(guard.integrity.__file__), Path(ncm_pixi.__file__)]
    hashes = start(INSPECT, 'token_and_open_source_screen', inputs, environment,
        {'expected_rows': EXPECTED_ROWS, 'source_cl100k_range': [192, 2048],
         'target_cl100k_limit': 128, 'native_lexical_limit': 126,
         'text_rule': 'normalized equality or >=32 common distinct5grams and >=80percent shorter containment',
         'open_protected_sources': 192, 'protected_metadata': counts,
         'current_TRAIN_sources': 1376, 'all_strings_preserved': True,
         'sealed_final_full_text_comparison': False,
         'public_train_validation_test_near_duplicate_comparison': False,
         'source_family_mapping': 'public BBC article ID; not session/conversation lineage'})
    counters = Counter()
    histograms = {k: Counter() for k in ('source_cl100k', 'target_cl100k', 'target_native', 'eligible_source_cl100k', 'eligible_target_native')}
    seen_sources = set()
    begin = time.monotonic()
    with (EXPORT/'rows.jsonl').open() as stream, (INSPECT/'row_metadata.jsonl').open('x') as out:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            validate_row(row)
            source_hash = text_sha(row['document'])
            source_tokens = len(encoding.encode(row['document'], disallowed_special=()))
            target_tokens = len(encoding.encode(row['summary'], disallowed_special=()))
            lexical = tokenizer.encode(row['summary'], add_special_tokens=False, truncation=False)
            reasons = token_reasons(source_tokens, target_tokens, lexical, specials)
            if not reasons:
                counters['token_range_fit'] += 1
            # Use metadata hashes for all protected components; never open final text.
            if source_hash in protected['hashes'] or source_hash in protected['turns']:
                reasons.append('protected_exact_source_or_turn')
            if source_hash in train_hashes:
                reasons.append('current_TRAIN_exact_source')
            if source_hash in seen_sources:
                reasons.append('public_TRAIN_exact_source_duplicate')
            seen_sources.add(source_hash)
            # Run content comparisons even for out-of-range rows: account for every row.
            held_hits = text_index.matches(row['document'])
            train_hits = train_index.matches(row['document'])
            if held_hits:
                reasons.append('open_protected_text_overlap')
            if train_hits:
                reasons.append('current_TRAIN_text_overlap')
            counters.update(reasons)
            counters['rows'] += 1
            if not reasons:
                counters['screen_eligible_not_admitted'] += 1
                histograms['eligible_source_cl100k'][source_tokens] += 1
                histograms['eligible_target_native'][len(lexical)] += 1
            for key, n in [('source_cl100k', source_tokens), ('target_cl100k', target_tokens), ('target_native', len(lexical))]:
                histograms[key][n] += 1
            out.write(json.dumps({'row_number': number, 'article_id': row['id'],
                'source_sha256': source_hash, 'summary_sha256': text_sha(row['summary']),
                'source_cl100k_tokens': source_tokens, 'target_cl100k_tokens': target_tokens,
                'native_lexical_tokens': len(lexical), 'screen_eligible_not_admitted': not reasons,
                'reasons': reasons,
                'heldout_conflicts': [{'id': held[i]['id'], **detail} for i, detail in held_hits],
                'current_TRAIN_conflicts': [{'id': train[i]['id'], **detail} for i, detail in train_hits]}, allow_nan=False)+'\n')
            if number % 8192 == 0:
                print(json.dumps({'stage': 'inspect', 'rows': number,
                    'eligible': counters['screen_eligible_not_admitted'],
                    'elapsed_seconds': time.monotonic()-begin}), flush=True)
    if counters['rows'] != EXPECTED_ROWS:
        raise ValueError('incomplete inspected TRAIN population')
    write(INSPECT/'histograms.json', {k: dict(sorted(v.items())) for k, v in histograms.items()})
    finish(INSPECT, hashes, {'status': 'all_public_TRAIN_rows_screened_admission_pending',
        'counts': dict(counters), 'elapsed_seconds': time.monotonic()-begin,
        'source_disjointness_established': False,
        'open_source_and_protected_exact_hash_checks_completed': True,
        'sealed_final_full_text_overlap_audit_completed': False,
        'public_split_near_duplicate_audit_completed': False,
        'limitations': ['News references are not verified factual gold for the application.',
            'Token eligibility is not reconstruction or task accuracy.',
            'BBC article IDs do not supply conversation/component lineage.',
            'Semantic or paraphrased overlap is not ruled out by the text screen.']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('export', 'inspect'))
    stage = parser.parse_args().stage
    target = EXPORT if stage == 'export' else INSPECT
    existed = target.exists()
    try:
        export() if stage == 'export' else inspect()
    except Exception as error:
        if not existed and target.exists():
            write(target/'failure.json', {'error_type': type(error).__name__, 'error': str(error), **BOUNDARIES})
        raise
