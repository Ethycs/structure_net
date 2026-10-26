from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_public_source_metadata as metadata


def test_pilot_is_fixed_distinct_and_covers_range():
    lengths = [330, 192, 256, 384, 512, 768, 1024, 1536, 2048, 2000]
    entries = [{'id': str(i), 'source_cl100k_tokens': v} for i, v in enumerate(lengths)]
    assert metadata.pilot_positions(entries) == list(range(9))
    with pytest.raises(ValueError): metadata.pilot_positions(entries[:8])


def test_source_preparation_never_receives_label_oracle_fields():
    pair = {key: 'value' for key in ('id', 'source_id', 'group_id', 'source_component', 'source_sha256',
            'source', 'spans', 'split', 'source_cl100k_tokens', 'public_provenance')}
    pair.update(source_truncated=False, target_text='forbidden', native_word_ids=[1], oracle_codes=[2])
    def prepare(row, *tokens):
        assert not {'target_text', 'native_word_ids', 'oracle_codes'} & row.keys()
        return {**row, 'source_ids': [1, 2], 'alignment': [[[0, 1.]], [[1, 1.]]]}
    helper = SimpleNamespace(prepare_one=prepare, FORBIDDEN={'target_text', 'native_word_ids', 'oracle_codes'})
    result = metadata.prepare_source(pair, (None, None, None), helper)
    assert result['source'] == pair['source'] and not result['source_truncated']
