"""Full XSum paired pretraining catalog, with source and native-target guards."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time

from experiments.structure_net import ncm_public_xsum_data as data
from experiments.structure_net.ncm_public_xsum_data_v2 import EXPORT, INSPECT

HYPOTHESIS = 'ncm-public-training-catalog-v1'
DEST = data.ROOT/'20261026_public_training_catalog_v1'
AUDIT = data.ROOT/'20261026_public_training_catalog_audit_v1'
CODEC = data.ROOT/'20261026_public_xsum_codec_prepare_v1'
OLD_CODEC = data.ROOT/'20261025_codec_roundtrip_prepare_v1'
EXISTING = data.ROOT/'20261025_broad_teacher_catalog_v1'
INSPECTION_AUDIT = data.ROOT/'20261026_public_xsum_train_inspection_audit_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-training-catalog-plan.md')
TEST = Path('tests/structure_net/test_ncm_public_training_catalog.py')
REVISION = '7d4d486c2f8ef850b1a11aead99b894ff3dd7da9'
BOUNDARIES = {'model_training': False, 'new_judge_requests': 0, 'task_accuracy': None,
    'final_test_opened': False, 'source_disjointness_established': False,
    'label_factual_quality_verified': False, 'training_data_admitted': False}


def native_frame(lexical, specials):
    if (not 0 < len(lexical) <= 126 or any(type(v) is not int for v in lexical)
            or set(lexical) & set(specials)):
        raise ValueError('complete native lexical target without special/unknown tokens required')
    return [101, *lexical, 101]+[0]*(126-len(lexical))


def frame_hash(frame):
    if (len(frame) != 128 or frame[0] != 101 or any(type(v) is not int or v < 0 for v in frame)
            or 101 not in frame[1:]):
        raise ValueError('complete native128 frame required')
    end = frame.index(101, 1)
    if end < 2 or any(v in (0, 100, 101, 102) for v in frame[1:end]) or any(frame[end+1:]):
        raise ValueError('native frame content/EOS/PAD layout differs')
    return hashlib.sha256(json.dumps(frame, separators=(',', ':')).encode()).hexdigest()


def pair_rows():
    with (EXPORT/'rows.jsonl').open() as sources, (INSPECT/'row_metadata.jsonl').open() as metadata:
        for number, pair in enumerate(itertools.zip_longest(sources, metadata), 1):
            if None in pair:
                raise ValueError('complete original export/metadata pairing required')
            row, meta = map(json.loads, pair)
            if (row['id'] != meta['article_id'] or meta['row_number'] != number
                    or data.text_sha(row['document']) != meta['source_sha256']
                    or data.text_sha(row['summary']) != meta['summary_sha256']
                    or meta['screen_eligible_not_admitted'] != (not meta['reasons'])):
                raise ValueError('original source/target/row binding differs')
            yield row, meta


def source_only(row):
    keys = ('id', 'source_id', 'group_id', 'source_component', 'source_sha256', 'source',
            'spans', 'split', 'source_cl100k_tokens', 'source_truncated', 'public_provenance')
    return {key: row[key] for key in keys}


def paired_record(row, meta, frame):
    family = 'xsum_'+REVISION+'_train_'+row['id']
    identity = data.text_sha(family)
    source_hash = meta['source_sha256']
    return {'id': identity, 'source_id': family, 'group_id': family, 'source_component': family,
        'source': row['document'], 'source_sha256': source_hash, 'source_truncated': False,
        'spans': [{'source_id': family, 'role': 'article', 'turn_id': row['id'],
            'turn_text_sha256': source_hash, 'span_text_sha256': source_hash,
            'start_char': 0, 'end_char': len(row['document'])}],
        'split': 'TRAIN', 'source_cl100k_tokens': meta['source_cl100k_tokens'],
        'target_text': row['summary'], 'target_sha256': meta['summary_sha256'],
        'stored_tokens': meta['target_cl100k_tokens'], 'native_lexical_tokens': meta['native_lexical_tokens'],
        'native_word_ids': frame, 'native_frame_sha256': frame_hash(frame),
        'reference_kind': 'original_public_XSum_reference_not_factual_gold',
        'public_provenance': {'dataset': 'EdinburghNLP/xsum', 'revision': REVISION,
            'split': 'train', 'article_id': row['id'], 'original_row_number': meta['row_number']}}


def write_indexed(stream, index, row):
    payload = (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n').encode('utf-8')
    offset = stream.tell()
    stream.write(payload)
    entry = {k: row[k] for k in ('id', 'source_sha256', 'target_sha256', 'native_frame_sha256',
        'source_cl100k_tokens', 'stored_tokens', 'native_lexical_tokens')}
    entry.update(offset_bytes=offset, length_bytes=len(payload),
        record_sha256=hashlib.sha256(payload).hexdigest(), public_provenance=row['public_provenance'])
    index.write(json.dumps(entry, allow_nan=False)+'\n')
    return entry


def additional_reasons(row, meta, frame, protected, seen_frames, hits):
    reasons = []
    key = frame_hash(frame)
    if row['id'] in protected['public_article_ids']: reasons.append('public_development_article_id')
    if meta['source_sha256'] in protected['public_source_hashes']: reasons.append('public_development_exact_source')
    if meta['summary_sha256'] in protected['target_hashes']: reasons.append('protected_or_existing_TRAIN_target_text')
    if key in protected['native_frame_hashes']: reasons.append('protected_or_existing_TRAIN_native_target')
    if hits: reasons.append('public_development_text_overlap')
    if key in seen_frames: reasons.append('public_TRAIN_native_target_duplicate')
    return reasons


def setup():
    from experiments.structure_net import ncm_pixi
    from experiments.structure_net import ncm_shared_summary_source_guard as guard
    import tiktoken
    training, environment = ncm_pixi.configure()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or training.torch.cuda.is_initialized():
        raise ValueError('CPU-only catalog preparation required')
    for root in (EXPORT, INSPECT, INSPECTION_AUDIT, CODEC):
        data.completed(root)
    public = [r for r in data.read(CODEC/'cases.json') if r['split'] != 'TRAIN_control']
    if dict(Counter(r['split'] for r in public)) != {'representative': 256, 'long_stress': 32, 'digit_stress': 32}:
        raise ValueError('all320 fixed public codec diagnostics required')
    old = [r for r in data.read(OLD_CODEC/'cases.json') if r['split'] == 'reserved_component']
    existing = data.read(EXISTING/'cases.json')
    if len(old) != 64 or len(existing) != 1376:
        raise ValueError('all64 reserved application targets and1376 current TRAIN targets required')
    tokenizer = training.native.tokenizer()
    protected_targets = public+old+existing
    for row in protected_targets:
        lexical = tokenizer.encode(row['target_text'], add_special_tokens=False, truncation=False)
        if (native_frame(lexical, tokenizer.all_special_ids) != row['native_word_ids']
                or data.text_sha(row['target_text']) != row['target_sha256']):
            raise ValueError('protected target text/frame changed')
    protected = {'public_article_ids': {r['id'] for r in public},
        'public_source_hashes': {r['source_sha256'] for r in public},
        'target_hashes': {r['target_sha256'] for r in protected_targets},
        'native_frame_hashes': {frame_hash(r['native_word_ids']) for r in protected_targets}}
    by_id = {r['id']: r for r in public}
    held = []
    for row, meta in pair_rows():
        if row['id'] in by_id:
            expected = by_id[row['id']]
            if (not meta['screen_eligible_not_admitted'] or meta['source_sha256'] != expected['source_sha256']
                    or meta['summary_sha256'] != expected['target_sha256']
                    or meta['row_number'] != expected['original_row_number']):
                raise ValueError('public diagnostic original article binding changed')
            case = paired_record(row, meta, expected['native_word_ids'])
            case.update(split='PUBLIC_DEVELOPMENT', diagnostic_cohort=expected['split'])
            held.append(case)
    if len(held) != 320:
        raise ValueError('complete public development source texts required')
    index = guard.TextIndex([r['source'] for r in held])
    return training, environment, tokenizer, tiktoken.get_encoding('cl100k_base'), protected, held, index


def freeze(environment, tokenizer_assets):
    sources = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):
                sources.add(p)
    paths = [EXPORT/'rows.jsonl', INSPECT/'row_metadata.jsonl', CODEC/'cases.json',
        OLD_CODEC/'cases.json', EXISTING/'cases.json', PLAN, TEST, Path('pyproject.toml'), Path('pixi.lock')]
    paths += [root/name for root in (EXPORT, INSPECT, INSPECTION_AUDIT, CODEC)
              for name in ('result.json', 'actual_tool_receipt.json')]
    paths += tokenizer_assets
    DEST.mkdir(); (DEST/'producer_source').mkdir()
    for i, path in enumerate(sorted(sources)):
        (DEST/'producer_source'/f'{i}_{path.name}').write_bytes(path.read_bytes())
    result = {'schema': 'nal.ncm-public-training-catalog.v1', 'hypothesis_id': HYPOTHESIS,
        'environment': environment, 'config': {'original_rows': 204045, 'original_candidates': 163051,
            'public_development_cases': 320, 'reserved_application_target_cases': 64,
            'existing_TRAIN_target_cases': 1376, 'retain_all_eligible_pairs': True,
            'native_target_deduplication': 'first_original_eligible_row', 'full_source_retained': True},
        'source_hashes': {str(p): data.sha(p) for p in sorted(sources)},
        'input_hashes': {str(p): data.sha(p) for p in paths}, **BOUNDARIES}
    data.write(DEST/'producer.json', result)
    return result


def prepare():
    if DEST.exists(): raise ValueError('fresh append-only catalog root required')
    training, environment, tokenizer, encoding, protected, held, index = setup()
    assets = [p for p in sorted((training.native.INVENTORY/'bert-tokenizer').iterdir()) if p.is_file()]
    proof = freeze(environment, assets)
    data.write(DEST/'protection.json', {k: sorted(v) for k, v in protected.items()})
    with (DEST/'public_development.jsonl').open('x') as out:
        for row in held: out.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    counts, seen_frames = Counter(), set()
    began = time.monotonic()
    with (DEST/'training.jsonl').open('xb') as out, (DEST/'training_index.jsonl').open('x') as idx, \
            (DEST/'exclusions.jsonl').open('x') as excluded:
        for row, meta in pair_rows():
            counts['original_rows'] += 1
            if not meta['screen_eligible_not_admitted']:
                counts['prior_screen_exclusions'] += 1
                continue
            counts['original_candidates'] += 1
            lexical = tokenizer.encode(row['summary'], add_special_tokens=False, truncation=False)
            source_tokens = len(encoding.encode(row['document'], disallowed_special=()))
            target_tokens = len(encoding.encode(row['summary'], disallowed_special=()))
            if (source_tokens != meta['source_cl100k_tokens'] or target_tokens != meta['target_cl100k_tokens']
                    or len(lexical) != meta['native_lexical_tokens']
                    or data.token_reasons(source_tokens, target_tokens, lexical, set(tokenizer.all_special_ids))):
                raise ValueError('complete candidate token geometry changed')
            frame = native_frame(lexical, tokenizer.all_special_ids)
            hits = index.matches(row['document'])
            reasons = additional_reasons(row, meta, frame, protected, seen_frames, hits)
            if reasons:
                counts['additional_excluded_rows'] += 1
                counts.update(reasons)
                excluded.write(json.dumps({'article_id': row['id'], 'original_row_number': meta['row_number'],
                    'reasons': reasons, 'public_development_conflicts': [{'id': held[i]['id'], **detail} for i, detail in hits]},
                    allow_nan=False)+'\n')
            else:
                case = paired_record(row, meta, frame)
                write_indexed(out, idx, case)
                seen_frames.add(case['native_frame_sha256'])
                counts['retained_pairs'] += 1
                counts['source_cl100k_tokens'] += source_tokens
                counts['native_lexical_tokens'] += len(lexical)
            if counts['original_candidates'] % 8192 == 0:
                print(json.dumps({'stage': 'catalog', 'candidates': counts['original_candidates'],
                    'retained': counts['retained_pairs'], 'elapsed_seconds': time.monotonic()-began}), flush=True)
    if (counts['original_rows'] != 204045 or counts['original_candidates'] != 163051
            or counts['retained_pairs']+counts['additional_excluded_rows'] != 163051):
        raise ValueError('complete mutually exclusive row accounting required')
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    result = {'status': 'full_public_training_catalog_prepared_audit_pending', 'hypothesis_id': HYPOTHESIS,
        'counts': dict(counts), 'public_development_cases': len(held),
        'retained_all_remaining_pairs': True, 'elapsed_seconds': time.monotonic()-began,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'artifacts': {str(p): data.sha(p) for p in sorted(DEST.glob('*')) if p.is_file()}, **BOUNDARIES}
    data.write(DEST/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare',))
    parser.parse_args()
    existed = DEST.exists()
    try:
        prepare()
    except Exception as error:
        if not existed and DEST.exists():
            data.write(DEST/'failure.json', {'exception': type(error).__name__, 'message': str(error), **BOUNDARIES})
        raise
