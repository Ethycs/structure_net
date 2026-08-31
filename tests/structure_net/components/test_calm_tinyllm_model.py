"""Contract tests for the CALM-style TinyLLM framework components."""

from pathlib import Path

import pytest
import torch

from structure_net.components.models import (
    CALMTinyLLMConfig,
    CALMTinyLLMModel,
    EnergyScoreObjective,
    TinyLLMConfig,
    TinyLLMModel,
)
from structure_net.components.trainers import ContinuousAutoregressiveTrainer
from structure_net.core import EvolutionContext


@pytest.fixture
def calm_config() -> CALMTinyLLMConfig:
    backbone = TinyLLMConfig(
        block_size=8,
        vocab_size=24,
        n_layer=2,
        n_head=2,
        n_embd=16,
        initialization_seed=5,
    )
    return CALMTinyLLMConfig(
        backbone=backbone,
        patch_size=2,
        latent_size=4,
        autoencoder_hidden_size=16,
        autoencoder_intermediate_size=32,
        autoencoder_dropout=0.1,
        generator_layers=1,
        noise_size=4,
        energy_samples=3,
        target_samples=4,
        initialization_seed=6,
    )


def test_config_round_trip_and_validation(calm_config):
    assert CALMTinyLLMConfig.from_dict(calm_config.to_dict()) == calm_config
    assert calm_config.token_block_size == 16
    with pytest.raises(ValueError, match="at least two"):
        CALMTinyLLMConfig(backbone=calm_config.backbone, energy_samples=1)
    with pytest.raises(ValueError, match="in \(0, 2\)"):
        CALMTinyLLMConfig(backbone=calm_config.backbone, energy_beta=2.0)


def test_component_initialization_seed_owns_all_parameters(calm_config):
    torch.manual_seed(1)
    first = CALMTinyLLMModel(calm_config)
    torch.manual_seed(999)
    second = CALMTinyLLMModel(calm_config)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(second.state_dict()[name], value, rtol=0, atol=0)


def test_autoencoder_shapes_tied_weights_and_backward(calm_config):
    model = CALMTinyLLMModel(calm_config)
    tokens = torch.randint(0, calm_config.backbone.vocab_size, (3, 8))
    output = model.autoencoder(tokens)

    assert output.logits.shape == (3, 4, 2, calm_config.backbone.vocab_size)
    assert output.mean.shape == (3, 4, calm_config.latent_size)
    assert model.autoencoder.decoder.token_weight is model.autoencoder.encoder.embedding.weight
    output.loss.backward()
    assert model.autoencoder.encoder.embedding.weight.grad is not None
    assert torch.isfinite(output.loss)


def test_energy_score_is_sample_permutation_invariant():
    objective = EnergyScoreObjective(beta=1.0, target_samples=5)
    predictions = torch.randn(4, 2, 3, 6, requires_grad=True)
    mean = torch.randn(2, 3, 6)
    log_std = torch.full_like(mean, -1.0)
    target_noise = torch.randn(5, 2, 3, 6)
    first = objective(predictions, mean, log_std, target_noise=target_noise)
    second = objective(
        predictions[torch.tensor([2, 0, 3, 1])],
        mean,
        log_std,
        target_noise=target_noise,
    )
    torch.testing.assert_close(first, second)
    first.backward()
    assert predictions.grad is not None


def test_conversion_copies_backbone_but_changes_interface(calm_config):
    source = TinyLLMModel(calm_config.backbone)
    converted = CALMTinyLLMModel.from_tinyllm(source, calm_config)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(converted.backbone.state_dict()[name], value, rtol=0, atol=0)

    tokens = torch.randint(0, calm_config.backbone.vocab_size, (2, 8))
    converted.set_autoencoder_frozen(True)
    output = converted(tokens, output_hidden_states=True)
    assert output.latent_predictions.shape == (3, 2, 3, 4)
    assert output.hidden_states is not None
    assert output.hidden_states.shape == (2, 3, 16)
    output.loss.backward()
    assert all(parameter.grad is None for parameter in converted.autoencoder.parameters())
    assert any(parameter.grad is not None for parameter in converted.input_adapter.parameters())
    assert not converted.get_architecture_summary()["source_checkpoint_function_preserved"]


def test_two_stage_trainers_isolate_parameters(calm_config):
    model = CALMTinyLLMModel.from_tinyllm(
        TinyLLMModel(calm_config.backbone), calm_config
    )
    tokens = torch.randint(0, calm_config.backbone.vocab_size, (4, 8))
    backbone_before = {
        name: value.detach().clone() for name, value in model.backbone.state_dict().items()
    }
    autoencoder_trainer = ContinuousAutoregressiveTrainer(
        phase="autoencoder", optimizer_kwargs={"lr": 1e-3}
    )
    ae_metrics = autoencoder_trainer.train_step(
        model, tokens, EvolutionContext(device="cpu")
    )
    assert ae_metrics["grad_norm"] > 0.0
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, backbone_before[name], rtol=0, atol=0)

    autoencoder_before = {
        name: value.detach().clone() for name, value in model.autoencoder.state_dict().items()
    }
    energy_trainer = ContinuousAutoregressiveTrainer(
        phase="energy", optimizer_kwargs={"lr": 1e-3}
    )
    energy_metrics = energy_trainer.train_step(
        model, tokens, EvolutionContext(device="cpu")
    )
    assert energy_metrics["grad_norm"] > 0.0
    assert model.autoencoder_frozen
    for name, value in model.autoencoder.state_dict().items():
        torch.testing.assert_close(value, autoencoder_before[name], rtol=0, atol=0)


def test_checkpoint_round_trip_is_exact_with_fixed_noise(calm_config, tmp_path: Path):
    model = CALMTinyLLMModel.from_tinyllm(
        TinyLLMModel(calm_config.backbone), calm_config
    ).eval()
    model.set_autoencoder_frozen(True)
    tokens = torch.randint(0, calm_config.backbone.vocab_size, (2, 8))
    prediction_noise = torch.randn(3, 2, 3, 4)
    target_noise = torch.randn(4, 2, 3, 4)
    expected = model(tokens, noise=prediction_noise, target_noise=target_noise)

    checkpoint = model.save_checkpoint(
        tmp_path / "calm.pt", metadata={"purpose": "round-trip-test"}
    )
    restored = CALMTinyLLMModel.from_checkpoint(checkpoint).eval()
    actual = restored(tokens, noise=prediction_noise, target_noise=target_noise)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.latent_predictions, expected.latent_predictions, rtol=0, atol=0
    )
    assert restored.checkpoint_metadata == {"purpose": "round-trip-test"}
    generated = restored.generate_next_chunk(
        tokens[:, :4], noise=torch.randn(1, 2, calm_config.noise_size)
    )
    assert generated.shape == (2, calm_config.patch_size)
