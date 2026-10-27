from copy import deepcopy

from experiments.neural_architecture_lab import ncm_task_supervision_results as report


def comparison():
    arm={'valid':128,'passed':0,'task_accuracy':0.0,'primary_admitted':False,
         'joint_passed':None,'DEV_qualified':False}
    return {'arms':{'existing':deepcopy(arm),'expanded':deepcopy(arm)},
        'paired_Terra':{'both_pass':0,'both_fail':128,'existing_only':0,'expanded_only':0,'unresolved':0},
        'additional_passes':0,'descriptive_supervision_benefit':False,'selected_DEV_candidate':None}


def test_report_shows_actual_zero_accuracy_and_distinguishes_unmeasured_final():
    value=comparison();value['classification']=report.classification(value)
    text=report.report_text(value)
    assert '| existing | 0/128 | 0.00% | 128/128 |' in text
    assert '| expanded | 0/128 | 0.00% | 128/128 |' in text
    assert 'Final accuracy remains unmeasured' in text
    assert value['classification']=='supervision_benefit_not_established_DEV_failed'


def test_invalid_judgments_remain_unresolved_in_report():
    value=comparison();value['arms']['expanded'].update(valid=127,task_accuracy=None)
    value['paired_Terra']['unresolved']=1;value['additional_passes']=None;value['descriptive_supervision_benefit']=None
    value['classification']=report.classification(value)
    assert value['classification']=='semantic_comparison_unresolved_invalid_judgments'
    assert '| expanded | 0/128 | unresolved | 127/128 |' in report.report_text(value)


def test_descriptive_gain_does_not_imply_dev_or_final_success():
    value=comparison();value.update(additional_passes=7,descriptive_supervision_benefit=True)
    assert report.classification(value)=='descriptive_supervision_benefit_DEV_failed'
    value['selected_DEV_candidate']='expanded'
    assert report.classification(value)=='DEV_candidate_qualified_final_unmeasured'
