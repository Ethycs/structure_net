import pytest
from pathlib import Path
import runpy

StructuredCodingEvidence = runpy.run_path(str(Path(__file__).resolve().parents[2] /
    'src/neural_architecture_lab/word_games_structured_coding_hypothesis.py'))['StructuredCodingEvidence']


def test_complete_record_does_not_claim_general_accuracy_or_secrecy():
    r=StructuredCodingEvidence(85176,8,8,4,32,64,0).record()
    assert r['tested'] is True and r['confirmed'] is False
    assert r['general_secrecy_proved'] is False and r['swe_bench_evaluated'] is False
    assert r['native_correct']==8 and r['translated_correct']==4


@pytest.mark.parametrize('cases,tasks,native',[(85175,8,8),(85176,7,7),(85176,8,9)])
def test_partial_or_invalid_evidence_cannot_be_registered(cases,tasks,native):
    with pytest.raises(ValueError):
        StructuredCodingEvidence(cases,tasks,native,0,0,64,0)
