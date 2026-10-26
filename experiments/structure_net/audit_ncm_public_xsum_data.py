"""Replay public data counts/bindings and sample token counts, without admission."""
import argparse
from collections import Counter
import itertools
import math
import os
from pathlib import Path

from experiments.structure_net import ncm_public_xsum_data as data
from experiments.structure_net.ncm_public_xsum_data_v2 import EXPORT, INSPECT

DEST = data.ROOT/'20261026_public_xsum_train_inspection_audit_v1'


def describe(histogram):
    items = sorted((int(k), n) for k, n in histogram.items())
    total = sum(n for _, n in items)
    if not total:
        return {'rows': 0}
    result = {'rows': total, 'min': items[0][0], 'max': items[-1][0],
              'mean': sum(k*n for k, n in items)/total}
    for name, quantile in [('p10', .1), ('p50', .5), ('p90', .9), ('p99', .99)]:
        threshold, running = math.ceil(quantile*total), 0
        for k, n in items:
            running += n
            if running >= threshold:
                result[name] = k
                break
    return result


def main():
    from experiments.structure_net import ncm_pixi
    import tiktoken
    training, environment = ncm_pixi.configure()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or training.torch.cuda.is_initialized():
        raise ValueError('CPU-only audit required')
    exported, inspected = data.completed(EXPORT), data.completed(INSPECT)
    if exported['rows'] != data.EXPECTED_ROWS:
        raise ValueError('complete public TRAIN required')
    inputs = [EXPORT/'rows.jsonl', EXPORT/'result.json', EXPORT/'actual_tool_receipt.json',
        INSPECT/'row_metadata.jsonl', INSPECT/'histograms.json', INSPECT/'result.json', INSPECT/'actual_tool_receipt.json']
    hashes = data.start(DEST, 'independent_data_replay', inputs, environment,
        {'all_row_text_id_hash_and_arithmetic_replay': True,
         'token_replay_sample': '256 evenly spaced original physical rows, plus every special-token rejection',
         'no_repeated_all_pair_text_overlap_computation': True})
    tokenizer, encoding = training.native.tokenizer(), tiktoken.get_encoding('cl100k_base')
    sample = {1+i*(data.EXPECTED_ROWS-1)//255 for i in range(256)}
    counts, replayed, digits, source_hashes = Counter(), 0, 0, set()
    histograms = {key: Counter() for key in data.read(INSPECT/'histograms.json')}
    with (EXPORT/'rows.jsonl').open() as sources, (INSPECT/'row_metadata.jsonl').open() as metadata:
        for number, pair in enumerate(itertools.zip_longest(sources, metadata), 1):
            if None in pair:
                raise ValueError('export/metadata row count differs')
            row, meta = map(data.json.loads, pair)
            assert meta['row_number'] == number and meta['article_id'] == row['id']
            assert meta['source_sha256'] == data.text_sha(row['document'])
            assert meta['summary_sha256'] == data.text_sha(row['summary'])
            reasons = meta['reasons']
            assert len(reasons) == len(set(reasons))
            assert ('public_TRAIN_exact_source_duplicate' in reasons) == (meta['source_sha256'] in source_hashes)
            source_hashes.add(meta['source_sha256'])
            counts['rows'] += 1
            counts.update(reasons)
            numerical = {'application_bypass_below192', 'source_above2048', 'target_outside1to128',
                         'native_target_outside1to126', 'native_special_or_unknown'}
            counts['token_range_fit'] += not (numerical & set(reasons))
            assert meta['screen_eligible_not_admitted'] == (not reasons)
            if not reasons:
                counts['screen_eligible_not_admitted'] += 1
                digits += any(c.isdigit() for c in row['summary'])
                histograms['eligible_source_cl100k'][meta['source_cl100k_tokens']] += 1
                histograms['eligible_target_native'][meta['native_lexical_tokens']] += 1
            for key, field in [('source_cl100k', 'source_cl100k_tokens'), ('target_cl100k', 'target_cl100k_tokens'), ('target_native', 'native_lexical_tokens')]:
                histograms[key][meta[field]] += 1
            if number in sample or 'native_special_or_unknown' in reasons:
                source_tokens = len(encoding.encode(row['document'], disallowed_special=()))
                target_tokens = len(encoding.encode(row['summary'], disallowed_special=()))
                lexical = tokenizer.encode(row['summary'], add_special_tokens=False, truncation=False)
                assert (source_tokens, target_tokens, len(lexical)) == (meta['source_cl100k_tokens'], meta['target_cl100k_tokens'], meta['native_lexical_tokens'])
                assert set(data.token_reasons(source_tokens, target_tokens, lexical, set(tokenizer.all_special_ids))) == numerical & set(reasons)
                replayed += 1
    assert dict(counts) == inspected['counts'] and counts['rows'] == data.EXPECTED_ROWS
    assert {k: {str(n): v for n, v in h.items()} for k, h in histograms.items()} == data.read(INSPECT/'histograms.json')
    data.finish(DEST, hashes, {'status': 'data_bindings_and_counts_replayed_admission_pending',
        'counts': dict(counts), 'token_count_rows_replayed': replayed,
        'eligible_summaries_containing_a_digit': digits,
        'token_distributions': {k: describe(h) for k, h in histograms.items()},
        'source_disjointness_established': False, 'all_pair_text_overlap_independently_recomputed': False,
        'next_gate': 'prospective public-TRAIN codec reconstruction check and training/exposure admission'})


if __name__ == '__main__':
    argparse.ArgumentParser(__doc__).parse_args()
    main()
