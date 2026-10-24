from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_dense_latent_word as run
from experiments.structure_net.ncm_generated_history_core import independent_word_branch, own_history


def setup():
    spec = importlib.util.spec_from_file_location("dense_word_latent_test_model",
        Path("tests/structure_net/test_ncm_generated_history_core.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forecaster, codec, features, mask, ids = module.model()
    branch = independent_word_branch(forecaster)
    run.prepare_branch(branch)
    return forecaster, branch, codec, features, mask, ids


def test_dense_training_updates_source_head_without_decoder_body_or_label_gradients():
    forecaster, branch, codec, features, mask, ids = setup()
    history = own_history(forecaster, features, mask, ids)
    fixed = deepcopy(forecaster.state_dict())
    decoder = deepcopy(codec.decoder.state_dict())
    teacher = torch.randn(2, 32, 16, requires_grad=True)
    words = torch.randint(0, 30, (2, 32))
    words[0, 25:] = 31
    optimizer = torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad], lr=.001)
    run.train_mode(branch)
    losses = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        latent, logits = run.prediction(branch, features, mask, history["word_conditioning"], ids)
        loss, parts = run.objective(latent, logits, teacher, words, pad=31)
        losses.append(float(loss.detach()))
        loss.backward()
        assert teacher.grad is None and features.grad is None
        assert all(p.grad is None for p in forecaster.parameters())
        assert all(p.grad is None for p in codec.parameters())
        assert all(p.grad is None for p in branch.word_decoder.parameters())
        assert branch.source_head.source_projection.weight.grad.abs().sum() > 0
        assert branch.word_projection.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in branch.generator.lm_heads.parameters())
        optimizer.step()
    assert losses[-1] < losses[0]
    assert all(torch.equal(v, forecaster.state_dict()[k]) for k, v in fixed.items())
    assert all(torch.equal(v, branch.word_decoder.state_dict()[k]) for k, v in decoder.items())


def test_setup_restores_original_decoder_instead_of_retaining_failed_adapted_weights():
    forecaster, branch, codec, *_ = setup()
    with torch.no_grad():
        next(branch.word_decoder.parameters()).add_(1)
    assert not all(torch.equal(v, codec.decoder.state_dict()[k]) for k, v in branch.word_decoder.state_dict().items())
    run.prepare_branch(branch)
    assert all(torch.equal(v, codec.decoder.state_dict()[k]) for k, v in branch.word_decoder.state_dict().items())
    assert all(not p.requires_grad for p in branch.word_decoder.parameters())
    run.train_mode(branch)
    assert branch.training and not branch.word_decoder.training
    for net in (branch.generator.transformer, branch.word_decoder):
        for module in net.modules():
            if isinstance(module, torch.nn.Dropout):
                assert module.p == 0
            value = getattr(module, "dropout", None)
            if isinstance(value, (float, int)):
                assert value == 0


def test_decoder_remains_fp32_inside_mixed_precision_forward():
    _, branch, _, *_ = setup()
    latent = torch.randn(2, 32, 16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = run.decoder_logits(branch, latent)
    assert logits.dtype == torch.float32
    expected = branch.word_decoder(latent.reshape(8, 8, 16)).reshape(2, 32, 32)[..., :31]
    assert torch.equal(logits, expected)
    logits.square().mean().backward()
    assert latent.grad is not None and torch.count_nonzero(latent.grad)


def test_objective_supervises_all_latents_but_ignores_pad_words():
    latent = torch.zeros(1, 3, 2, requires_grad=True)
    teacher = torch.ones_like(latent, requires_grad=True)
    logits = torch.tensor([[[1., 0.], [0., 1.], [9., -9.]]], requires_grad=True)
    words = torch.tensor([[0, 1, 2]])
    loss, parts = run.objective(latent, logits, teacher, words, pad=2)
    ce = F.cross_entropy(logits[0, :2], words[0, :2])
    torch.testing.assert_close(loss, 1 + .1 * ce)
    loss.backward()
    assert teacher.grad is None
    assert torch.count_nonzero(latent.grad[0, 2])  # Padding latents still have dense supervision.
    assert not torch.count_nonzero(logits.grad[0, 2])


def test_unsafe_decoder_modes_and_incomplete_supervision_are_rejected():
    _, branch, _, *_ = setup()
    branch.word_decoder.train()
    with pytest.raises(ValueError, match="frozen and eval"):
        run.decoder_logits(branch, torch.zeros(1, 32, 16))
    run.train_mode(branch, False)
    with pytest.raises(ValueError, match="complete latent"):
        run.decoder_logits(branch, torch.zeros(1, 31, 16))
    with pytest.raises(ValueError, match="complete finite"):
        run.objective(torch.zeros(1, 3, 2), torch.zeros(1, 3, 4), torch.zeros(1, 2, 2), torch.zeros(1, 3, dtype=torch.long))
