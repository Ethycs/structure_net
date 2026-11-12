import runpy

import pytest

Record=runpy.run_path('src/neural_architecture_lab/word_games_relabel_coding_hypothesis.py')['RelabelCodingEvidence']


def test_complete_regression_parity_is_not_general_accuracy_or_secrecy():
    record=Record(8,8,288,288,0).record()
    assert record['tested'] and record['paired_case_disagreements']==0
    assert not record['confirmed'] and not record['general_accuracy_established'] and not record['information_secrecy_tested']


@pytest.mark.parametrize('args',[(8,8,287,288,0),(0,8,288,288,0),(8,8,288,288,289),(True,8,36,288,0)])
def test_invalid_counts_rejected(args):
    with pytest.raises(ValueError):Record(*args)
