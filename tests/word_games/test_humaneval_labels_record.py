import runpy

import pytest

Record = runpy.run_path('src/neural_architecture_lab/word_games_humaneval_labels_hypothesis.py')['HumanEvalLabelsEvidence']


def test_perfect_sample_is_not_universal_confirmation():
    record = Record(32,32,32).record()
    assert record['tested'] and not record['confirmed']
    assert record['direct_only'] == record['relabeled_only'] == 0
    assert not record['general_accuracy_established'] and not record['information_secrecy_tested']


@pytest.mark.parametrize('counts', [(33,32,32),(True,0,0),(31,30,20),(0,2,1),(-1,0,0)])
def test_invalid_or_impossible_pairs_rejected(counts):
    with pytest.raises(ValueError): Record(*counts)
