import pytest

from experiments.structure_net.ncm_expanded_word_data import combine_rows


def source(identifier):
    return {key: identifier+key for key in ('id', 'source_id', 'group_id', 'source_sha256', 'source', 'spans')}


def test_combination_preserves_original_targets_and_accepts_only_complete_matching_sources():
    original = {**source('old'), 'word_ids': [1, 2], 'concept_ids': [3], 'target_text': 'old target'}
    new = source('new')
    label = {**new, 'text': 'complete target', 'target': 'complete target', 'teacher_valid': True}
    rows = combine_rows([original], [new], [label])
    assert rows[0] == original and rows[1]['target_text'] == label['text']
    assert 'word_ids' not in rows[1] and rows[1]['split'] == 'train'
    with pytest.raises(ValueError, match='complete source'):
        combine_rows([original], [new], [{**label, 'source': 'different fragment'}])


def test_combination_rejects_group_overlap_and_duplicate_accepted_labels():
    original = source('old')
    new = source('new')
    label = {**new, 'text': 'target', 'target': 'target', 'teacher_valid': True}
    with pytest.raises(ValueError, match='disjoint'):
        combine_rows([], [new], [label, label])
    overlap = {**new, 'group_id': original['group_id']}
    with pytest.raises(ValueError, match='disjoint'):
        combine_rows([original], [overlap], [{**label, 'group_id': original['group_id']}])
