"""Replay every public training pair, native target, exclusion and byte offset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time

from experiments.structure_net import ncm_public_training_catalog as catalog

data = catalog.data


def accounting(counts):
    if (counts['original_rows'] != 204045 or counts['original_candidates'] != 163051
            or counts['original_rows'] != counts['original_candidates']+counts['prior_screen_exclusions']
            or counts['original_candidates'] != counts['retained_pairs']+counts['additional_excluded_rows']
            or counts['retained_pairs'] <= 0):
        raise ValueError('complete mutually exclusive row accounting required')


def verify_index(entry, row, payload, offset):
    if (entry['offset_bytes'] != offset or entry['length_bytes'] != len(payload)
            or entry['record_sha256'] != hashlib.sha256(payload).hexdigest()
            or any(row[key] != entry[key] for key in ('id', 'source_sha256', 'target_sha256',
                'native_frame_sha256', 'source_cl100k_tokens', 'stored_tokens', 'native_lexical_tokens', 'public_provenance'))):
        raise ValueError('indexed UTF8 record binding differs')


def audit():
    if catalog.AUDIT.exists(): raise ValueError('fresh independent catalog audit required')
    original = data.completed(catalog.DEST)
    proof = data.read(catalog.DEST/'producer.json')
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    for i, (name, expected) in enumerate(sorted(proof['source_hashes'].items())):
        path = catalog.DEST/'producer_source'/f'{i}_{Path(name).name}'
        if data.sha(path) != expected: raise ValueError('frozen producer snapshot changed')
    training, environment, tokenizer, encoding, protected, held, index = catalog.setup()
    if data.read(catalog.DEST/'protection.json') != {key: sorted(value) for key, value in protected.items()}:
        raise ValueError('source/target/native-frame protection changed')
    with (catalog.DEST/'public_development.jsonl').open() as stream:
        saved_held = [json.loads(line) for line in stream]
    if saved_held != held or len(held) != 320:
        raise ValueError('all320 original public development sources/references required')
    counts, seen, seen_ids, seen_sources = Counter(), set(), set(), set()
    started = time.monotonic()
    with (catalog.DEST/'training.jsonl').open('rb') as pairs, \
            (catalog.DEST/'training_index.jsonl').open() as offsets, \
            (catalog.DEST/'exclusions.jsonl').open() as exclusions:
        for source, meta in catalog.pair_rows():
            counts['original_rows'] += 1
            if not meta['screen_eligible_not_admitted']:
                counts['prior_screen_exclusions'] += 1
                continue
            counts['original_candidates'] += 1
            native = tokenizer.encode(source['summary'], add_special_tokens=False, truncation=False)
            source_tokens = len(encoding.encode(source['document'], disallowed_special=()))
            target_tokens = len(encoding.encode(source['summary'], disallowed_special=()))
            if (not 192 <= source_tokens <= 2048 or not 1 <= target_tokens <= 128
                    or not 1 <= len(native) <= 126 or set(native) & set(tokenizer.all_special_ids)
                    or (source_tokens, target_tokens, len(native)) !=
                       (meta['source_cl100k_tokens'], meta['target_cl100k_tokens'], meta['native_lexical_tokens'])):
                raise ValueError('candidate token bounds or counts differ')
            frame = [101, *native, 101]+[0]*(126-len(native))
            fingerprint = hashlib.sha256(json.dumps(frame, separators=(',', ':')).encode()).hexdigest()
            hits = index.matches(source['document'])
            reasons = []
            if source['id'] in protected['public_article_ids']: reasons.append('public_development_article_id')
            if meta['source_sha256'] in protected['public_source_hashes']: reasons.append('public_development_exact_source')
            if meta['summary_sha256'] in protected['target_hashes']: reasons.append('protected_or_existing_TRAIN_target_text')
            if fingerprint in protected['native_frame_hashes']: reasons.append('protected_or_existing_TRAIN_native_target')
            if hits: reasons.append('public_development_text_overlap')
            if fingerprint in seen: reasons.append('public_TRAIN_native_target_duplicate')
            if reasons:
                counts['additional_excluded_rows'] += 1; counts.update(reasons)
                expected = {'article_id': source['id'], 'original_row_number': meta['row_number'], 'reasons': reasons,
                    'public_development_conflicts': [{'id': held[i]['id'], **detail} for i, detail in hits]}
                if json.loads(exclusions.readline()) != expected:
                    raise ValueError('ordered complete exclusion evidence differs')
            else:
                offset = pairs.tell(); payload = pairs.readline()
                row, entry = json.loads(payload), json.loads(offsets.readline())
                verify_index(entry, row, payload, offset)
                family = 'xsum_'+catalog.REVISION+'_train_'+source['id']
                if (row['id'] != data.text_sha(family) or row['id'] in seen_ids or row['source_sha256'] in seen_sources
                        or any(row[key] != family for key in ('source_id', 'group_id', 'source_component'))
                        or row['source'] != source['document'] or row['target_text'] != source['summary']
                        or row['source_sha256'] != meta['source_sha256'] or row['target_sha256'] != meta['summary_sha256']
                        or row['native_word_ids'] != frame or row['native_frame_sha256'] != fingerprint
                        or row['split'] != 'TRAIN' or row['source_truncated']
                        or row['source_cl100k_tokens'] != source_tokens or row['stored_tokens'] != target_tokens
                        or row['native_lexical_tokens'] != len(native)
                        or row['reference_kind'] != 'original_public_XSum_reference_not_factual_gold'
                        or row['public_provenance'] != {'dataset': 'EdinburghNLP/xsum', 'revision': catalog.REVISION,
                            'split': 'train', 'article_id': source['id'], 'original_row_number': meta['row_number']}):
                    raise ValueError('full retained public pair differs from original data')
                expected_span = {'source_id': family, 'role': 'article', 'turn_id': source['id'],
                    'turn_text_sha256': meta['source_sha256'], 'span_text_sha256': meta['source_sha256'],
                    'start_char': 0, 'end_char': len(source['document'])}
                if row['spans'] != [expected_span]: raise ValueError('complete original source span differs')
                seen.add(fingerprint); seen_ids.add(row['id']); seen_sources.add(row['source_sha256'])
                counts['retained_pairs'] += 1
                counts['source_cl100k_tokens'] += source_tokens
                counts['native_lexical_tokens'] += len(native)
            if counts['original_candidates'] % 16384 == 0:
                print(json.dumps({'stage': 'audit', 'candidates_replayed': counts['original_candidates'],
                    'retained': counts['retained_pairs'], 'elapsed_seconds': time.monotonic()-started}), flush=True)
        if pairs.read(1) or offsets.read(1) or exclusions.read(1):
            raise ValueError('unaccounted catalog/index/exclusion records')
    accounting(counts)
    if dict(counts) != original['counts']:
        raise ValueError('exhaustive catalog counts differ')
    paths = [p for p in sorted(catalog.DEST.iterdir()) if p.is_file()]
    paths += [Path(__file__), Path('tests/structure_net/test_ncm_public_training_catalog_audit.py')]
    result = {'schema': 'nal.ncm-public-training-catalog-audit.v1', 'hypothesis_id': catalog.HYPOTHESIS,
        'classification': 'full_public_paired_catalog_admitted_for_pretraining',
        'counts': dict(counts), 'public_development_cases': 320,
        'protected_application_target_cases': 64, 'existing_TRAIN_target_cases': 1376,
        'all_original_rows_replayed': True, 'all_candidate_tokens_recomputed': True,
        'all_public_development_text_overlaps_recomputed': True, 'all_byte_offsets_replayed': True,
        'all_remaining_pairs_retained': True, 'strings_rewritten_or_truncated': False,
        'training_data_admitted': True, 'admission_scope': 'public_reference_conditional_pretraining_only',
        'label_factual_quality_verified': False, 'model_training': False, 'new_judge_requests': 0,
        'task_accuracy': None, 'final_test_opened': False, 'source_disjointness_established': False,
        'sealed_final_full_text_overlap_audit_completed': False,
        'public_validation_or_test_payload_read': False,
        'elapsed_seconds': time.monotonic()-started,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'environment': environment, 'artifacts': {str(p): data.sha(p) for p in paths}}
    data.verify(proof['input_hashes']); data.verify(proof['source_hashes'])
    catalog.AUDIT.mkdir()
    data.write(catalog.AUDIT/'campaign_results.json', result)
    data.write(catalog.AUDIT/'result.json', {'status': result['classification'],
        'training_data_admitted': True, 'admission_scope': result['admission_scope'], 'counts': dict(counts),
        'artifacts': {str(catalog.AUDIT/'campaign_results.json'): data.sha(catalog.AUDIT/'campaign_results.json')}})
    print(json.dumps({k: v for k, v in result.items() if k not in ('artifacts', 'environment')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('audit',))
    parser.parse_args()
    audit()
