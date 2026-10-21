import pytest
from neural_architecture_lab.ncm_llama_hypothesis import classification,NcmEvidence,graph_classification,NcmGraphEvidence


def test_codec_and_teacher_are_positive_controls_before_speed():
    assert classification(972)=='training_label_capacity_or_control_failed'
    assert classification(973,60)=='codec_fidelity_failed'
    assert classification(1024,61,94,100,100,100)=='teacher_control_failed'
    assert classification(1024,64,100,99,100,100)=='distilled_quality_failed'


def test_joint_speed_and_quality_thresholds_are_inclusive():
    assert classification(1024,64,95,95,2,2)=='task_specific_ncm_prototype_meets_speed_and_quality'
    assert classification(1024,64,100,100,2,.99)=='distilled_quality_retained_speed_target_unmet'
    with pytest.raises(ValueError):classification(1024,64,100,100,float('nan'),2)


def test_completed_index_does_not_claim_confirmation_or_general_capability():
    r=NcmEvidence('codec_fidelity_failed','campaign','a'*64,'report','b'*64,0).record()
    assert r['tested'] and not r['confirmed'] and r['underpowered']
    assert not r['full_application_context_tested'] and not r['application_accuracy_tested']


def test_graph_success_requires_all_outputs_equal_and_inherited_quality():
    assert graph_classification(299,100,100,10,10)=='output_parity_failed'
    assert graph_classification(300,100,99,10,10)=='inherited_model_quality_failed'
    assert graph_classification(300,100,100,2,2)=='frozen_ncm_graph_prototype_meets_speed_and_quality'
    assert graph_classification(300,100,100,2,1.99)=='speed_target_unmet'


def test_graph_record_distinguishes_new_and_reused_evidence():
    r=NcmGraphEvidence('speed_target_unmet','graph','a'*64,'report','b'*64,'parent','c'*64).record()
    assert r['direct_experiment_count']==300 and r['reused_control_timing_cells']==900
    assert r['tested'] and not r['confirmed'] and r['used_runtime_validation_sources']
    assert not r['full_application_context_tested']
