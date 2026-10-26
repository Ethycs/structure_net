import pytest

from experiments.structure_net import ncm_public_xsum_codec as probe


def population():
    return [{'id': str(i), 'source_sha256': 'source'+str(i), 'native_lexical_tokens': 10+i%117,
             'target_text': 'A claim with 2021.' if i%2 else 'A claim without a digit.'} for i in range(512)]


def test_selection_reproducible_without_order_or_score_selection():
    rows = population()
    selected = probe.select(rows)
    assert selected == probe.select(list(reversed(rows)))
    assert len(selected) == len({r['id'] for r in selected}) == 320
    first = sorted(rows, key=lambda r: probe.rank(r['id']))[:256]
    assert [r['id'] for r in selected[:256]] == [r['id'] for r in first]
    remaining = [r for r in rows if r['id'] not in {x['id'] for x in first}]
    long = sorted(remaining, key=lambda r: (-r['native_lexical_tokens'], probe.rank(r['id'])))[:32]
    assert [r['id'] for r in selected[256:288]] == [r['id'] for r in long]
    assert all('2021' in r['target_text'] for r in selected[288:])


def test_selection_rejects_duplicate_sources_and_unfilled_quotas():
    rows = population()
    rows[-1]['source_sha256'] = rows[0]['source_sha256']
    with pytest.raises(ValueError, match='unique'):
        probe.select(rows)
    with pytest.raises(ValueError, match='quotas'):
        probe.select(population()[:300])


def test_framing_retains_full_target_and_rejects_truncation():
    assert probe.frame([200]*126, {0, 100, 101, 102}) == [101]+[200]*126+[101]
    for ids in ([], [200]*127, [100], [101]):
        with pytest.raises(ValueError): probe.frame(ids, {0, 100, 101, 102})


def passing():
    return {'cases': 256, 'exact_rendered': 244, 'content_matches': 990, 'content_tokens': 1000,
            'has_digits': 20, 'digit_cases_preserved': 19, 'invalid_outputs': 0, 'empty': 0}


def test_joint_gate_inclusive_and_reconstruction_only():
    assert probe.gate(passing())
    for key, value in [('exact_rendered', 243), ('content_matches', 989),
                       ('digit_cases_preserved', 18), ('invalid_outputs', 1), ('empty', 1)]:
        assert not probe.gate({**passing(), key: value})
    assert probe.BOUNDARIES['source_only_task_accuracy'] is None
    assert probe.BOUNDARIES['final_task_goal_achieved'] is False


@pytest.mark.parametrize('changes', [{'cases': 255}, {'has_digits': 0}])
def test_partial_sample_cannot_pass(changes):
    with pytest.raises(ValueError): probe.gate({**passing(), **changes})


def test_partial_or_duplicate_outputs_cannot_aggregate():
    with pytest.raises(ValueError, match='complete'):
        probe.aggregate([])
    with pytest.raises(ValueError, match='duplicate'):
        probe.aggregate([{'id': 'a', 'arm': 'released'}]*2)
