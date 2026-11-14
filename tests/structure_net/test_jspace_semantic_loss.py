import pytest
import torch
from torch.nn import functional as F

from experiments.structure_net.jspace_semantic_loss import SemanticLossConfig,semantic_recovery_loss


def inputs(dtype=torch.float64):
    torch.manual_seed(52)
    student=torch.randn(2,4,11,dtype=dtype,requires_grad=True)
    teacher=torch.randn(2,4,11,dtype=dtype,requires_grad=True)
    return student,teacher,dict(label_ids=[2,4,6,8],choice_counts=torch.tensor([2,4]),answers=torch.tensor([1,3]))


def test_formula_matches_explicit_per_case_full_and_conditional_objectives():
    student,teacher,kwargs=inputs()
    result=semantic_recovery_loss(student,teacher,**kwargs)
    full=[];kd=[];ce=[]
    for i,count in enumerate(kwargs["choice_counts"].tolist()):
        s=student[i,[1,3]];t=teacher[i,[1,3]].detach()
        p=(t/2).softmax(-1)
        full.append((p*((t/2).log_softmax(-1)-(s/2).log_softmax(-1))).sum(-1).mean())
        ids=kwargs["label_ids"][:count]
        sc=s[-1,ids];tc=t[-1,ids]
        kd.append(((tc/2).softmax(-1)*((tc/2).log_softmax(-1)-(sc/2).log_softmax(-1))).sum())
        ce.append(F.cross_entropy(sc[None],kwargs["answers"][i:i+1]))
    expected=torch.stack(full).mean()+torch.stack(kd).mean()+.5*torch.stack(ce).mean()
    torch.testing.assert_close(result["loss"],expected)
    torch.testing.assert_close(result["loss"],result["weighted_full_kl"]+result["weighted_choice_kl"]+result["weighted_choice_ce"])


def test_teacher_detached_format_slots_zero_gradient_full_vocabulary_still_trains():
    student,teacher,kwargs=inputs()
    result=semantic_recovery_loss(student,teacher,**kwargs)
    result["loss"].backward()
    assert teacher.grad is None
    assert torch.count_nonzero(student.grad[:,[0,2]])==0
    assert student.grad[:,1,10].abs().sum()>0  # Midpoint non-choice vocabulary.
    assert student.grad[:,3,10].abs().sum()>0  # Answer full vocabulary retained.
    assert torch.isfinite(student.grad).all()
    for key in ("accuracy","teacher_accuracy","valid_label_probability","choice_margin"):
        assert not result[key].requires_grad


def test_masked_choices_do_not_receive_conditional_gradients_even_with_huge_logits():
    student,teacher,kwargs=inputs()
    with torch.no_grad():student[0,-1,6]=1000;teacher[0,-1,8]=1000
    result=semantic_recovery_loss(student,teacher,**kwargs,
        config=SemanticLossConfig(full_kl_weight=0,choice_kl_weight=.5,choice_ce_weight=.5))
    result["loss"].backward()
    assert student.grad[0,-1,6]==student.grad[0,-1,8]==0
    assert result["prediction"][0]<2 and result["teacher_prediction"][0]<2
    assert torch.isfinite(student.grad).all()


def test_two_position_student_and_four_position_teacher_are_equivalent():
    student,teacher,kwargs=inputs()
    compact=student.detach()[:,[1,3]].clone().requires_grad_(True)
    original=semantic_recovery_loss(student,teacher,**kwargs)
    reduced=semantic_recovery_loss(compact,teacher,**kwargs)
    original["loss"].backward();reduced["loss"].backward()
    torch.testing.assert_close(original["loss"],reduced["loss"])
    torch.testing.assert_close(student.grad[:,[1,3]],compact.grad)


def test_format_and_label_mass_improvements_alone_cannot_change_conditional_loss():
    student,teacher,kwargs=inputs()
    modified=student.detach().clone()
    modified[:,[0,2]]=teacher.detach()[:,[0,2]]
    for i,count in enumerate(kwargs["choice_counts"].tolist()):
        modified[i,-1,kwargs["label_ids"][:count]]+=8
    before=semantic_recovery_loss(student,teacher,**kwargs)
    after=semantic_recovery_loss(modified,teacher,**kwargs)
    torch.testing.assert_close(before["choice_kl"],after["choice_kl"])
    torch.testing.assert_close(before["choice_ce"],after["choice_ce"])
    torch.testing.assert_close(before["accuracy"],after["accuracy"])
    assert after["valid_label_probability"]>before["valid_label_probability"]


def test_permutation_of_valid_labels_and_answer_preserves_loss_and_accuracy():
    student,teacher,kwargs=inputs()
    before=semantic_recovery_loss(student,teacher,**kwargs)
    # Permute within both valid sets, preserving their memberships.
    altered=dict(kwargs,label_ids=[4,2,8,6],answers=torch.tensor([0,2]))
    after=semantic_recovery_loss(student,teacher,**altered)
    torch.testing.assert_close(before["loss"],after["loss"])
    torch.testing.assert_close(before["accuracy"],after["accuracy"])


@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
def test_low_precision_inputs_compute_objective_in_fp32(dtype):
    student,teacher,kwargs=inputs(dtype)
    result=semantic_recovery_loss(student,teacher,**kwargs)
    assert result["loss"].dtype==torch.float32
    result["loss"].backward()
    assert torch.isfinite(student.grad).all()


def test_invalid_targets_counts_and_nonfinite_logits_rejected():
    student,teacher,kwargs=inputs()
    with pytest.raises(ValueError,match="permitted"):
        semantic_recovery_loss(student,teacher,**dict(kwargs,answers=torch.tensor([2,3])))
    with pytest.raises(ValueError,match="Distinct"):
        semantic_recovery_loss(student,teacher,**dict(kwargs,label_ids=[2,2,6,8]))
    with pytest.raises(ValueError):SemanticLossConfig(choice_ce_weight=.9)
    bad=student.detach().clone();bad[0,0,0]=float("nan")
    with pytest.raises(FloatingPointError):semantic_recovery_loss(bad,teacher,**kwargs)
