import math

import pytest

from experiments.structure_net.jspace_answer_mass import compare_records,decompose_records,report,summarize


def row(identity="a",logits=(1.,2.),answer=1,mass=.1):
    maximum=max(logits);lse=maximum+math.log(sum(math.exp(v-maximum) for v in logits))
    return dict(identity=identity,family="tiny",answer=answer,prediction=max(range(len(logits)),key=logits.__getitem__),
        logits=list(logits),ce=lse-logits[answer]-math.log(mass))


def test_exact_probability_factorization_and_conditional_argmax():
    source=row(mass=.00001)
    value=decompose_records([source])[0]
    assert value["full_vocabulary_ce"]==pytest.approx(value["conditional_label_ce"]+value["valid_label_mass_ce"])
    assert value["valid_label_probability"]==pytest.approx(.00001)
    assert value["conditional_correct_probability"]==pytest.approx(math.exp(2)/(math.exp(1)+math.exp(2)))
    assert summarize([value])["overall"]["accuracy"]==1
    assert summarize([value])["overall"]["prediction_counts"]=={"B":1}
    assert summarize([value])["overall"]["best_constant_label_correct"]==1


def test_better_label_mass_without_accuracy_improvement_is_identified():
    before=decompose_records([row(logits=(2.,1.),answer=1,mass=.00001)])
    after=decompose_records([row(logits=(2.,1.),answer=1,mass=.5)])
    comparison=compare_records(before,after)["overall"]
    assert comparison["wins"]==comparison["losses"]==0
    assert comparison["mean_conditional_ce_reduction"]==pytest.approx(0)
    assert comparison["fraction_full_ce_reduction_from_label_mass"]==pytest.approx(1)


def test_paired_accuracy_and_validation():
    before=decompose_records([row("a",(2.,1.),1),row("b",(2.,1.),0)])
    after=decompose_records([row("b",(2.,1.),0),row("a",(1.,2.),1)])
    result=compare_records(before,after)["overall"]
    assert result["wins"]==1 and result["losses"]==0
    assert result["label_transitions"]=={"A->A":1,"A->B":1}
    assert result["both_correct"]==1 and result["same_prediction"]==1
    assert abs(result["decomposition_residual"])<1e-12
    with pytest.raises(ValueError,match="identical"):compare_records(before,after[:1])
    bad=row();bad["ce"]=0
    with pytest.raises(ValueError,match="cannot be lower"):decompose_records([bad])
    with pytest.raises(ValueError,match="development"):report("confirmation.json")
