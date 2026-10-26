import io
import json

import pytest

from experiments.structure_net import ncm_public_training_catalog as catalog


def fixture():
    row = {'id': '123', 'document': 'An unchanged source: café.', 'summary': 'A summary.'}
    meta = {'source_sha256': catalog.data.text_sha(row['document']),
        'summary_sha256': catalog.data.text_sha(row['summary']), 'source_cl100k_tokens': 192,
        'target_cl100k_tokens': 3, 'native_lexical_tokens': 2, 'row_number': 7}
    frame = catalog.native_frame([200, 201], {0, 100, 101, 102, 103})
    return row, meta, frame


def test_native_frame_complete_and_canonical():
    frame = catalog.native_frame(list(range(200, 326)), {0, 100, 101, 102, 103})
    assert len(frame) == 128 and frame[-1] == 101
    assert catalog.frame_hash(frame) == catalog.frame_hash(list(frame))
    for ids in ([], [200]*127, [200, 100], [200.1]):
        with pytest.raises(ValueError):
            catalog.native_frame(ids, {0, 100, 101, 102, 103})
    bad = catalog.native_frame([200], {0, 100, 101, 102, 103}); bad[-1] = 4
    with pytest.raises(ValueError, match='layout'):
        catalog.frame_hash(bad)


def test_guard_preserves_all_overlapping_reasons_and_native_target_deduplication():
    row, meta, frame = fixture()
    protected = {'public_article_ids': {'123'}, 'public_source_hashes': {meta['source_sha256']},
        'target_hashes': {meta['summary_sha256']}, 'native_frame_hashes': {catalog.frame_hash(frame)}}
    reasons = catalog.additional_reasons(row, meta, frame, protected, {catalog.frame_hash(frame)}, [(0, {})])
    assert len(reasons) == 6 and len(set(reasons)) == 6
    empty = {key: set() for key in protected}
    assert catalog.additional_reasons(row, meta, frame, empty, set(), []) == []
    assert catalog.additional_reasons(row, meta, frame, empty, {catalog.frame_hash(frame)}, []) == [
        'public_TRAIN_native_target_duplicate']


def test_pair_projection_keeps_source_and_provenance_but_removes_all_targets():
    row, meta, frame = fixture()
    pair = catalog.paired_record(row, meta, frame)
    assert pair['source'] == row['document'] and pair['target_text'] == row['summary']
    source = catalog.source_only({**pair, 'oracle_codes': [1], 'teacher_latent': 'forbidden'})
    assert not ({'target_text', 'target_sha256', 'native_word_ids', 'oracle_codes', 'teacher_latent'} & source.keys())
    assert source['source_sha256'] == meta['source_sha256'] and source['source_truncated'] is False
    assert source['public_provenance']['original_row_number'] == 7
    assert source['spans'][0]['end_char'] == len(row['document'])


def test_index_is_utf8_byte_accurate_and_supports_random_access():
    row, meta, frame = fixture()
    pair = catalog.paired_record(row, meta, frame)
    payload, index = io.BytesIO(), io.StringIO()
    a = catalog.write_indexed(payload, index, pair)
    b = catalog.write_indexed(payload, index, {**pair, 'source': 'Second source — unchanged.'})
    assert a['offset_bytes'] == 0 and b['offset_bytes'] == a['length_bytes']
    for entry, expected in ((a, pair), (b, {**pair, 'source': 'Second source — unchanged.'})):
        payload.seek(entry['offset_bytes'])
        raw = payload.read(entry['length_bytes'])
        assert catalog.hashlib.sha256(raw).hexdigest() == entry['record_sha256']
        assert json.loads(raw) == expected
    assert len(index.getvalue().splitlines()) == 2
