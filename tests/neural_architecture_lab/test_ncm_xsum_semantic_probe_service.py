import pytest
from experiments.neural_architecture_lab import ncm_xsum_semantic_probe_service as service


def result():
    return {'status':'systems_semantic_pilot_passed','cases_per_arm':4,'metrics':{a:{} for a in service.study.ARMS},
            'inherited_first_judgments':4691,'model_training':False,'final_test_opened':False}


def test_completed_pilot_is_not_full_grading_or_a_quality_claim():
    value=result();assert service.valid_result('pilot',value)
    assert not service.valid_result('grade',value)
    value['status']='systems_semantic_pilot_failed';assert service.valid_result('pilot',value)
    value['cases_per_arm']=3;assert not service.valid_result('pilot',value)


def test_only_cpu_and_fixed_fresh_judge_stage_commands():
    assert service.ENV['CUDA_VISIBLE_DEVICES']==''
    assert service.command('grade')[-1]=='grade'
    assert service.command('pilot')[1:5]==['run','--locked','-e','ncm']
    with pytest.raises(ValueError):service.command('train')
    value=result();value['model_training']=True;assert not service.valid_result('pilot',value)
