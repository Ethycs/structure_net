"""Prospective semantic-recovery objective; no model, optimizer or data loader.

Keep the original fixed full-vocabulary teacher and tied readout in the caller.
Distill midpoint text and answer positions, excluding the two universal format
positions. Explicit conditional-choice KD/CE supervises which permitted answer
is correct, independently of the probability assigned to answer letters.

The default 0.25 full-vocabulary KD / 0.25 choice KD / 0.50 choice CE weights
are a fixed, prospective control, not a claim of empirical optimality. Both KD
terms use temperature 2 and its squared gradient-scale correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SemanticLossConfig:
    temperature: float = 2.
    full_kl_weight: float = .25
    choice_kl_weight: float = .25
    choice_ce_weight: float = .50

    def __post_init__(self):
        weights=(self.full_kl_weight,self.choice_kl_weight,self.choice_ce_weight)
        if not math.isfinite(self.temperature) or self.temperature<=0:
            raise ValueError("Temperature must be positive and finite")
        if any(not math.isfinite(v) or v<0 for v in weights) or not math.isclose(sum(weights),1.):
            raise ValueError("Nonnegative loss weights must sum to one")


def _positions(logits):
    if logits.ndim!=3 or logits.shape[0]<1 or logits.shape[-1]<2 or logits.shape[1] not in (2,4):
        raise ValueError("Logits must be [batch,4,vocabulary] or [batch,2,vocabulary]")
    return logits[:,[1,3]] if logits.shape[1]==4 else logits


def semantic_recovery_loss(student,teacher,*,label_ids,choice_counts,answers,config=SemanticLossConfig()):
    """Return objective components and detached answer-ranking diagnostics.

    Student and teacher may independently contain all four cache positions or
    only [midpoint, answer], in that order. Four-position inputs select slots
    [1,3]; slots [0,2] receive no gradient. ``label_ids`` is the shared ordered
    vocabulary-ID list for A/B/C/D; ``choice_counts`` and ``answers`` are Long
    tensors [batch], and answers are choice indices, not vocabulary token IDs.
    Variable two/four-choice cases are masked before conditional normalization.

    KD targets are always detached. The caller must supply full-vocabulary
    logits: passing a four-row classifier violates the experiment contract.
    The loss itself cannot infer an external model's true vocabulary size.
    """
    if (not student.is_floating_point() or not teacher.is_floating_point()
            or student.device!=teacher.device):
        raise ValueError("Floating teacher/student logits on the same device required")
    selected_student,selected_teacher=_positions(student),_positions(teacher)
    if selected_student.shape!=selected_teacher.shape:
        raise ValueError("Teacher/student batch and vocabulary dimensions must agree")
    batch,_,vocabulary=selected_student.shape
    ids=torch.as_tensor(label_ids,device=student.device)
    if (ids.ndim!=1 or ids.dtype!=torch.long or len(ids)<2 or ids.unique().numel()!=len(ids)
            or bool((ids<0).any()) or bool((ids>=vocabulary).any())):
        raise ValueError("Distinct in-vocabulary Long label IDs required")
    for value in (choice_counts,answers):
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.long or value.shape!=(batch,) or value.device!=student.device:
            raise ValueError("Choice counts and answer indices must be same-device Long [batch] tensors")
    if (bool((choice_counts<2).any()) or bool((choice_counts>len(ids)).any())
            or bool((answers<0).any()) or bool((answers>=choice_counts).any())):
        raise ValueError("Each answer must belong to its permitted choice set")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise FloatingPointError("Nonfinite teacher/student logits")
    dtype=torch.float64 if student.dtype==torch.float64 or teacher.dtype==torch.float64 else torch.float32
    s=selected_student.to(dtype)
    t=selected_teacher.detach().to(dtype)
    temperature=config.temperature
    log_p=(t/temperature).log_softmax(-1)
    log_q=(s/temperature).log_softmax(-1)
    per_full=(log_p.exp()*(log_p-log_q)).sum(-1).mean(-1)

    valid=torch.arange(len(ids),device=student.device)[None]<choice_counts[:,None]
    s_choice=s[:,-1].index_select(-1,ids)
    t_choice=t[:,-1].index_select(-1,ids)
    s_masked=s_choice.masked_fill(~valid,float("-inf"))
    t_masked=t_choice.masked_fill(~valid,float("-inf"))
    choice_log_q=(s_masked/temperature).log_softmax(-1).masked_fill(~valid,0.)
    choice_log_p=(t_masked/temperature).log_softmax(-1).masked_fill(~valid,0.)
    choice_p=choice_log_p.exp().masked_fill(~valid,0.)
    # Finite replacements before multiplication avoid 0*(-inf-(-inf)) at
    # excluded choices and give zero gradients through those conditional rows.
    per_choice_kl=(choice_p*(choice_log_p-choice_log_q)).sum(-1)
    per_choice_ce=F.cross_entropy(s_masked,answers,reduction="none")
    weighted_full=config.full_kl_weight*temperature**2*per_full
    weighted_choice_kl=config.choice_kl_weight*temperature**2*per_choice_kl
    weighted_choice_ce=config.choice_ce_weight*per_choice_ce
    per_loss=weighted_full+weighted_choice_kl+weighted_choice_ce
    if not torch.isfinite(per_loss).all():raise FloatingPointError("Nonfinite semantic objective")
    result=dict(loss=per_loss.mean(),per_case_loss=per_loss,
        full_kl=per_full.mean(),choice_kl=per_choice_kl.mean(),choice_ce=per_choice_ce.mean(),
        weighted_full_kl=weighted_full.mean(),weighted_choice_kl=weighted_choice_kl.mean(),
        weighted_choice_ce=weighted_choice_ce.mean(),per_case_full_kl=per_full,
        per_case_choice_kl=per_choice_kl,per_case_choice_ce=per_choice_ce)
    with torch.no_grad():
        prediction=s_masked.argmax(-1)
        teacher_prediction=t_masked.argmax(-1)
        valid_mass=(s_masked.logsumexp(-1)-s[:,-1].logsumexp(-1)).exp()
        alternative=valid & (torch.arange(len(ids),device=s.device)[None]!=answers[:,None])
        correct=s_choice.gather(-1,answers[:,None]).squeeze(-1)
        margin=correct-s_choice.masked_fill(~alternative,float("-inf")).max(-1).values
        full_ce=F.cross_entropy(s[:,-1],ids[answers],reduction="none")
        result.update(prediction=prediction,teacher_prediction=teacher_prediction,
            accuracy=(prediction==answers).to(dtype).mean(),teacher_accuracy=(teacher_prediction==answers).to(dtype).mean(),
            teacher_agreement=(prediction==teacher_prediction).to(dtype).mean(),
            per_case_choice_margin=margin,choice_margin=margin.mean(),
            per_case_valid_label_probability=valid_mass,valid_label_probability=valid_mass.mean(),
            per_case_answer_full_vocab_ce=full_ce,answer_full_vocab_ce=full_ce.mean())
    return result
