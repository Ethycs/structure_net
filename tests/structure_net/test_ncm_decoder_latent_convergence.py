from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
import torch

from experiments.structure_net import ncm_decoder_latent_convergence as run


def tiny_codec():
    spec = importlib.util.spec_from_file_location("latent_convergence_test_model",
        Path("tests/structure_net/test_ncm_generated_history_core.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forecaster, original, *_ = module.model()
    return original, forecaster.generator.vqvae_local[0]


def test_resuming_latent_and_moments_equals_uninterrupted_optimization():
    torch.manual_seed(151)
    original, facade = tiny_codec()
    cfg = run.Config(updates=8, prior_updates=3, word_vocab=31, pad=31)
    words = torch.randint(0, 30, (1, facade.context_length))
    words[0, -6:] = 31
    initial = run.prior.encoder_latent(facade, words)
    before = deepcopy(original.state_dict())
    full, full_trace = run.prior.refine(facade, initial, words, 8, cfg)
    partial, prefix = run.prior.refine(facade, initial, words, 3, cfg)
    copy = run.prior.to_cpu(partial)
    resumed, suffix = run.continue_latent(facade, partial, words, 5, cfg)
    assert run.prior.same(full, resumed)
    assert full_trace == prefix + suffix
    assert run.prior.same(partial, copy)
    assert run.prior.same(before, original.state_dict())
    assert all(p.grad is None for p in original.parameters())
    assert suffix[0]["anchor_MSE"] > 0  # Anchor remains at the original encoder output.


def test_continuation_requires_selected_state_and_fixed_additional_budget():
    _, facade = tiny_codec()
    cfg = run.Config(updates=8, prior_updates=3, word_vocab=31, pad=31)
    words = torch.randint(0, 30, (1, facade.context_length))
    initial = run.prior.encoder_latent(facade, words)
    saved, _ = run.prior.refine(facade, initial, words, 3, cfg)
    for steps in (0, 6):
        with pytest.raises(ValueError, match="fixed additional"):
            run.continue_latent(facade, saved, words, steps, cfg)
    wrong = deepcopy(saved)
    wrong["optimizer"]["state"][0]["step"] += 1
    with pytest.raises(ValueError, match="exact32-step"):
        run.continue_latent(facade, wrong, words, 1, cfg)


def test_fixed_config_inherits_original_objective_and_optimizer():
    config = run.Config()
    assert config.updates == 256 and config.prior_updates == 32
    assert config.lr == run.prior.Config().lr
    assert config.anchor_weight == run.prior.Config().anchor_weight
    with pytest.raises(ValueError, match="additional latent budget"):
        replace(config, updates=32)
