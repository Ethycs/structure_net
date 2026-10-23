import torch
import torch.nn.functional as F

from experiments.structure_net.ncm_paired_word_control import noisy_mask, soft_consistency


def test_causal_noisy_mask_excludes_identical_previous_prefix_and_padding():
    gold = torch.tensor([[1, 2, 3, 9, 9]])
    state = torch.tensor([[1, 7, 3, 9, 9]])
    targets = torch.tensor([[1, 2, 3, 9, -100]])
    assert noisy_mask(state, gold, targets).tolist() == [[False, False, True, True, False]]
    assert not noisy_mask(gold, gold, targets).any()


def test_full_soft_ce_matches_dense_and_detaches_teacher():
    torch.manual_seed(1)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    teacher_logits = torch.randn(2, 4, 7, requires_grad=True)
    probabilities = teacher_logits.softmax(-1)
    mask = torch.tensor([[False, True, True, False], [False, False, True, True]])
    loss = soft_consistency(logits, probabilities, mask, chunk=2)
    expected = -(probabilities.detach() * logits.log_softmax(-1)).sum(-1)[mask].mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad.abs().sum() > 0
    assert teacher_logits.grad is None


def test_sequential_backward_equals_joint_paired_gradient():
    torch.manual_seed(2)
    weights = torch.randn(3, 5, requires_grad=True)
    clean_inputs, noisy_inputs = torch.randn(1, 4, 3), torch.randn(1, 4, 3)
    targets = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[False, True, True, True]])
    clean = clean_inputs @ weights
    teacher = clean.detach().softmax(-1)
    clean_loss = F.cross_entropy(clean.flatten(0, 1), targets.flatten()) * 10
    noisy_loss = soft_consistency(noisy_inputs @ weights, teacher, mask)
    (clean_loss + noisy_loss).backward()
    joint = weights.grad.clone(); weights.grad = None
    clean = clean_inputs @ weights
    teacher = clean.detach().softmax(-1)
    (F.cross_entropy(clean.flatten(0, 1), targets.flatten()) * 10).backward()
    soft_consistency(noisy_inputs @ weights, teacher, mask).backward()
    torch.testing.assert_close(weights.grad, joint)
