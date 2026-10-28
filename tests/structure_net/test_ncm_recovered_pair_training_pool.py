import pytest

from experiments.structure_net import ncm_recovered_pair_training_pool as pool


def pointer(text, source_id='sharegpt_demo_0'):
    h = pool.text_hash(text)
    return {'turn_text_sha256': h, 'span_text_sha256': h, 'start_char': 0,
            'end_char': len(text), 'source_id': source_id}


def test_pointer_reconstruction_rejects_missing_changed_and_target_only_sources():
    source = 'The project has three milestones.'
    p = pointer(source)
    lookup = lambda h: (source, 192, ['longmemeval_s_cleaned.json:sharegpt_demo_0:4']) if h == pool.text_hash(source) else None
    aliases = pool.pointer_proof(pool.text_hash(source), [p], lookup)
    assert aliases == {('turn', pool.text_hash(source)), ('family', 'sharegpt_demo')}
    assert pool.pointer_proof(pool.text_hash(source), [p], lambda h: None) is None
    assert pool.pointer_proof(pool.text_hash('different'), [p], lookup) is None
    assert pool.pointer_proof(pool.text_hash(source), [{**p, 'end_char': len(source)+1}], lookup) is None
    assert pool.pointer_proof(pool.text_hash(source), [{**p, 'span_text_sha256': '0'*64}], lookup) is None


def test_short_generic_turn_does_not_connect_every_public_session():
    source = 'Thanks!'; p = pointer(source)
    lookup = lambda h: (source, 2, ['longmemeval_s_cleaned.json:sharegpt_demo_0:4',
                                    'longmemeval_s_cleaned.json:sharegpt_other_9:2'])
    assert pool.pointer_proof(pool.text_hash(source), [p], lookup) == {('family', 'sharegpt_demo')}
    assert pool.pointer_proof(pool.text_hash(source), [{**p, 'source_id': 'unverified'}], lookup) is None


def test_multispan_proof_requires_every_original_turn_and_order():
    first, second = 'First fact', 'Second fact'
    rows = {pool.text_hash(s): (s, 200, ['longmemeval_s_cleaned.json:sharegpt_demo_0:0']) for s in [first, second]}
    sha = pool.text_hash(first+'\n'+second)
    assert pool.pointer_proof(sha, [pointer(first), pointer(second)], rows.get)
    assert pool.pointer_proof(sha, [pointer(second), pointer(first)], rows.get) is None
    del rows[pool.text_hash(second)]
    assert pool.pointer_proof(sha, [pointer(first), pointer(second)], rows.get) is None


def test_component_bridge_propagates_protection_independent_of_order():
    for edges in [[('train-a', 'alias'), ('alias', 'heldout'), ('train-b', 'train-a')],
                  [('train-b', 'train-a'), ('alias', 'heldout'), ('train-a', 'alias')]]:
        d = pool.Components()
        for a, b in edges:
            d.union(a, b)
        blocked = {d.find('heldout')}
        assert d.find('train-a') in blocked and d.find('train-b') in blocked
        assert d.find('independent') not in blocked


def test_native_capacity_boundaries_and_public_origin_families():
    assert pool.native_reasons([11]*126, {0, 100, 101}) == []
    assert pool.native_reasons([11]*127, {0, 100, 101})
    assert pool.native_reasons([100], {0, 100, 101}) == ['native_special_or_unknown']
    assert pool.origin_family('longmemeval_oracle.json:answer_1234abcd_abs_3:11') == 'answer_1234abcd'
    assert pool.origin_family('locomo:3:session_12:4') == 'locomo:3'
    with pytest.raises(ValueError):
        pool.origin_family('fixture:1')
