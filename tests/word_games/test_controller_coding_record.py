import runpy

import pytest

Record=runpy.run_path('src/neural_architecture_lab/word_games_controller_coding_hypothesis.py')['ControllerCodingEvidence']


def test_even_perfect_regression_score_is_not_general_accuracy():
    result=Record(8,8,8,32,32,0).record()
    assert result['tested'] and result['observed_task_count_parity']
    assert not result['confirmed'] and not result['general_accuracy_established']


@pytest.mark.parametrize('values',[(7,7,7,28,28,0),(8,8,8,31,32,0),(8,0,0,32,0,0),(8,8,8,32,32,25)])
def test_incomplete_or_inconsistent_counts_are_rejected(values):
    with pytest.raises(ValueError):Record(*values)
