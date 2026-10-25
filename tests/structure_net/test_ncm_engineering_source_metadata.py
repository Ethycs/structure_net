from copy import deepcopy

import pytest

from experiments.structure_net import ncm_engineering_source_metadata as run


def row():
    return {'id': 'id', 'source': 'Complete source', 'source_sha256': run.text_hash('Complete source'),
        'source_id': 'session', 'group_id': 'group', 'source_component': 'component', 'spans': [],
        'input_tokens': 200, 'target_text': 'Hidden target', 'native_word_ids': [101, 7, 101],
        'provenance': {'target': 'Hidden metadata'}, 'target_sha256': 'target'}


def test_targets_cannot_influence_source_geometry_input():
    a = row()
    b = deepcopy(a)
    b.update(target_text='Different secret label', native_word_ids=[9], target_sha256='changed')
    b['provenance']['target'] = 'Other metadata'
    assert run.source_only(a, 'test') == run.source_only(b, 'test')
    assert not run.geometry.FORBIDDEN & run.source_only(a, 'test').keys()
    assert run.source_only(a, 'test')['split'] == 'test'


def test_source_tamper_is_rejected_before_tokenization():
    value = row()
    value['source'] += ' altered'
    with pytest.raises(ValueError, match='unaltered complete source'):
        run.source_only(value, 'train')
