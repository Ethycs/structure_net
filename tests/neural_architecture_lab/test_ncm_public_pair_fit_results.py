from copy import deepcopy

import pytest

from experiments.neural_architecture_lab import ncm_public_pair_fit_results as results


def fixture():
    return {'hypothesis_id':results.study.HYPOTHESIS,'cases':32,'task_accuracy':None,
        'model_updates':0,'new_judge_requests':0,'final_test_opened':False,
        'semantic_generalization_established':False,'undertraining_causally_established':False,
        'scores':{arm:{mode:{'code_positions':20512,'rollout_correct_codes':count}
            for mode in ('matched','shuffled')} for arm,count in [('BROAD',100),('epoch2',200)]}}


def test_learning_direction_does_not_become_semantic_accuracy():
    value=fixture()
    for count,direction in [(200,'improved'),(100,'unchanged'),(99,'decreased')]:
        value['scores']['epoch2']['matched']['rollout_correct_codes']=count
        record=results.record_for(value)
        assert record['classification']=='public_TRAIN_rollout_'+direction
        assert record['matched_correct_code_change']==count-100
        assert record['task_accuracy_measured'] is False


def test_rejects_task_claims_and_incomplete_conditions():
    for key,replacement in [('task_accuracy',.95),('model_updates',1),('new_judge_requests',1),
            ('final_test_opened',True),('semantic_generalization_established',True),
            ('undertraining_causally_established',True),('cases',31)]:
        value=fixture();value[key]=replacement
        with pytest.raises(ValueError):results.record_for(value)
    value=deepcopy(fixture());del value['scores']['epoch2']['shuffled']
    with pytest.raises(ValueError):results.record_for(value)
