import pytest
from dataclasses import replace
from neural_architecture_lab.word_games_70tps_hypothesis import SeventyTpsEvidence


def evidence(rate):
    return SeventyTpsEvidence('candidate', 56.0, rate, 24, 24, 'analysis.json', 'a' * 64,
                             'report.md', 'b' * 64)


@pytest.mark.parametrize('rate,reached', [(69.999, False), (70.0, True), (75.0, True)])
def test_speed_threshold_does_not_confirm_semantic_quality(rate, reached):
    record = evidence(rate).record()
    assert record['speed_target_reached'] is reached
    assert not record['confirmed']
    assert not record['population_accuracy_established']
    assert not record['production_deployed']


@pytest.mark.parametrize('rate', [float('nan'), float('inf'), 0.0, -1.0])
def test_invalid_rates_are_rejected(rate):
    with pytest.raises(ValueError):
        evidence(rate)


def test_fast_but_invalid_selector_cannot_pass_quality_gate():
    record = replace(evidence(71.1), valid_selectors=0).record()
    assert record['speed_target_reached']
    assert record['confirmation_status'] == 'exploratory_70tps_reached_existing_checks_failed'
    assert not record['existing_checks_pass']
    assert not record['qualified_70tps_reached']
