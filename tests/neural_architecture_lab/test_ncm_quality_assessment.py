import pytest
from neural_architecture_lab.ncm_quality_assessment import AssessedQualityEvidence,CLASSIFICATION


def test_registry_separates_observed_strict_grade_from_unmeasured_accuracy():
    r=AssessedQualityEvidence('campaign','a'*64,'assessment','b'*64,'report','c'*64).record()
    assert r['tested'] and not r['confirmed'] and r['underpowered']
    assert r['confirmation_status']==CLASSIFICATION
    assert r['operational_classification']=='independent_summary_quality_below_95'
    assert not r['judge_qualified_for_accuracy'] and not r['application_accuracy_tested']
    assert r['direct_experiment_count']==480 and r['human_reference_diagnostic_outputs']==40
    with pytest.raises(ValueError):AssessedQualityEvidence('c','bad','a','b'*64,'r','c'*64).record()
