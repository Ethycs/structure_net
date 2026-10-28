import io
import json

import pytest

from experiments.structure_net import ncm_recovered_paraphrase_catalog as study


def row():
    return {'id': 'pair', 'source_sha256': 'sourcehash', 'source_component': 'component',
            'source': 'A source passage.', 'source_cl100k_tokens': 200, 'target_text': 'A summary.'}


def records(role='assistant', date='2023-01-01'):
    return [{'role': role, 'created_at': date}]


def test_conflicting_or_missing_original_context_is_rejected():
    assert study.original_context([])[2] == ['missing_original_role']
    assert study.original_context(records()+records('user'))[2] == ['conflicting_original_roles']
    assert study.original_context(records()+records(date='2024-01-01'))[2] == ['conflicting_original_dates']
    assert study.original_context(records(date=None)) == ('assistant', None, [])


def test_conditioned_source_never_depends_on_targets_but_preserves_dates():
    a = study.source_record(row(), 'assistant', '2023-01-01')
    changed = {**row(), 'id': 'another-pair', 'target_text': 'Another summary.', 'native_word_ids': [1], 'oracle_codes': [2]}
    assert study.source_record(changed, 'assistant', '2023-01-01') == a
    assert study.source_record(row(), 'assistant', '2023-01-02')['id'] != a['id']
    assert study.source_record(row(), 'user', '2023-01-01')['id'] != a['id']
    assert not {'target_text','native_word_ids','oracle_codes','summarizer_identity'} & a.keys()


def test_diagnostic_exclusion_covers_all_component_variants():
    for target in ('A summary.', 'A different summary.'):
        value = {**row(), 'target_text': target}
        assert 'reserved_diagnostic_component' in study.exclusion_reasons(value, records(), {'component'}, 'frame', set())[2]
    assert 'reserved_diagnostic_native_frame' in study.exclusion_reasons(row(), records(), set(), 'frame', {'frame'})[2]
    assert 'exact_substring_target' in study.exclusion_reasons({**row(), 'target_text': 'source'}, records(), set(), 'frame', set())[2]


class Tokenizer:
    all_special_ids = [0, 100, 101, 102]
    def __init__(self, ids): self.ids = ids
    def encode(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False, 'truncation': False}
        return self.ids


def test_native_frame_keeps_complete_target_and_true_end_marker():
    ids, _, size = study.frame(Tokenizer([1000]*126), 'target')
    assert size == 126 and ids == [101]+[1000]*126+[101]
    for bad in ([], [1000]*127, [100], [30522]):
        with pytest.raises(ValueError): study.frame(Tokenizer(bad), 'target')


def test_indexed_readback_detects_corrupt_or_unindexed_bytes(tmp_path):
    path, index = tmp_path/'rows.jsonl', tmp_path/'index.jsonl'
    with path.open('xb') as raw, index.open('x') as idx:
        study.write_record(raw, idx, {'id': 'one', 'text': 'α'})
        study.write_record(raw, idx, {'id': 'two', 'text': 'β'})
    assert study.readback(path, index, 2) == 2
    with path.open('ab') as raw: raw.write(b'extra')
    with pytest.raises(ValueError, match='incomplete'): study.readback(path, index, 2)
