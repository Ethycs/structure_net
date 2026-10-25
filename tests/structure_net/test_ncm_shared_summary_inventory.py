from experiments.structure_net.ncm_shared_summary_inventory import reasons


def row():
    return {'source_id': 'train', 'group_id': 'g', 'source_sha256': 'h',
        'spans': [{'source_id': 'train', 'turn_text_sha256': 't'}],
        'input_tokens': 2048, 'target_tokens': 128, 'target': 'text'}


def guard():
    return {key: set() for key in ('families', 'groups', 'hashes', 'turns')}


def test_exact_current_limits_are_eligible_without_trimming():
    assert reasons(row(), [5] * 126, guard(), {0, 100, 101, 102, 103}) == []
    r = row()
    r['input_tokens'] = 2049
    assert 'input_above2048_application_tokens' in reasons(r, [5] * 126, guard(), set())
    assert 'target_not_one_complete_native128_frame' in reasons(row(), [5] * 127, guard(), set())


def test_family_aliases_and_shared_turns_protect_all_reserved_cohorts():
    r, p = row(), guard()
    r['source_id'] = 'answer_abcd1234_2'
    p['families'].add('answer_abcd1234')
    p['turns'].add('t')
    assert set(reasons(r, [5], p, set())) == {'protected_source_family', 'protected_informative_turn'}


def test_application_bypass_and_unrepresentable_targets_are_counted():
    r = row()
    r['input_tokens'] = 191
    assert set(reasons(r, [100], guard(), {100})) == {'application_bypass_below192', 'native_target_special_or_UNK'}
