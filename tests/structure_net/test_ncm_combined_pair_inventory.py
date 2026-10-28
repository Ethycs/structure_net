import hashlib

import pytest

from experiments.structure_net.ncm_combined_pair_inventory import add


def row(text='summary', source='source', split='TRAIN'):
    return {'id': 'original', 'source': source, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'target_text': text, 'target_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'source_component': 'original-group', 'source_record_id': 'role-and-date-context',
        'split': split, 'stored_tokens': 1, 'native_lexical_tokens': 1,
        'native_word_ids': [101, 1045, 101]+[0]*125}


def test_exact_pair_union_retains_both_domains_and_contexts():
    groups = {}
    add(groups, row(), 'public', {'path': 'a'})
    add(groups, {**row(), 'source_record_id': 'different-context'}, 'conversation', {'path': 'b'})
    assert len(groups) == 1
    origins = next(iter(groups.values()))['origins']
    assert [r['domain'] for r in origins] == ['public', 'conversation']
    assert [r['source_record_id'] for r in origins] == ['role-and-date-context', 'different-context']


def test_alternative_targets_and_distinct_sources_are_not_discarded():
    groups = {}
    for value in (row(), row(text='alternative'), row(source='another source')):
        add(groups, value, 'conversation', {})
    assert len(groups) == 3


def test_nontrain_and_changed_source_are_rejected():
    with pytest.raises(ValueError, match='TRAIN'):
        add({}, row(split='DEV'), 'public', {})
    with pytest.raises(ValueError, match='source text binding'):
        add({}, {**row(), 'source': 'changed'}, 'public', {})
