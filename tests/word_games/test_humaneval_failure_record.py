import runpy

import pytest

Record=runpy.run_path('src/neural_architecture_lab/word_games_failure_diagnosis_hypothesis.py')['FailureDiagnosisEvidence']


def test_local_repairs_cannot_confirm_general_accuracy_or_autonomous_repair():
    record=Record(5,0,0,1,2).record()
    assert record['tested'] and not record['confirmed']
    assert record['saved_repairs_passing']==5 and record['primary_accuracy_unchanged']
    assert not record['general_accuracy_established'] and not record['autonomous_repair_demonstrated']


@pytest.mark.parametrize('counts',[(6,0,0,0,0),(5,3,0,0,0),(True,0,0,0,0),(5,0,0,-1,0)])
def test_invalid_counts_rejected(counts):
    with pytest.raises(ValueError):Record(*counts)
