"""Tests for the TinyLLM-compatible model and delayed feedback growth."""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from structure_net.components.evolvers import FeedbackGrowthEvolver
from structure_net.components.models import (
    TINYLLM_PRESETS,
    TinyLLMConfig,
    TinyLLMModel,
    create_tinyllm_model,
)
from structure_net.components.strategies import RandomFeedbackGrowthStrategy
from structure_net.components.trainers import CausalLanguageModelTrainer
from structure_net.core import AnalysisReport, EvolutionContext, EvolutionPlan


@pytest.fixture
def tiny_config() -> TinyLLMConfig:
    return TinyLLMConfig(
        block_size=8,
        vocab_size=32,
        n_layer=3,
        n_head=2,
        n_embd=16,
    )


def test_published_presets_match_tinyllm_depth_rule():
    assert TINYLLM_PRESETS == {
        "d6": (6, 6, 384),
        "d8": (8, 8, 512),
        "d10": (10, 10, 640),
        "d11": (11, 11, 704),
        "d12": (12, 12, 768),
    }
    config = TinyLLMConfig.from_preset("30m")
    assert config.preset == "d6"
    assert config.n_embd == 64 * config.n_layer
    assert config.n_embd // config.n_head == 64


def test_huggingface_config_translation(tiny_config):
    hf_config = tiny_config.to_huggingface_config()
    restored = TinyLLMConfig.from_huggingface_config(hf_config)
    assert restored == tiny_config


def test_forward_loss_and_contract(tiny_config):
    model = TinyLLMModel(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 6))
    logits, loss = model(input_ids, input_ids, return_full_logits=True)

    assert logits.shape == (2, 6, tiny_config.vocab_size)
    assert loss is not None and loss.ndim == 0
    assert model.transformer["wte"].weight is model.lm_head.weight
    assert model.supports_dynamic_growth()
    assert "model.feedback_topology" in model.contract.provided_outputs


def test_attention_is_causal(tiny_config):
    model = TinyLLMModel(tiny_config).eval()
    original = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 8, 9]])

    with torch.no_grad():
        original_logits, _ = model(original, return_full_logits=True)
        changed_logits, _ = model(changed_future, return_full_logits=True)

    torch.testing.assert_close(original_logits[:, :2], changed_logits[:, :2])


def _upstream_reference_logits(model: TinyLLMModel, input_ids: torch.Tensor):
    """Independent form of the GPT-2 math used by TinyLLM's pinned llm.c."""
    positions = torch.arange(input_ids.shape[1], dtype=torch.long)
    value = F.embedding(input_ids, model.transformer["wte"].weight)
    value = value + F.embedding(positions, model.transformer["wpe"].weight)
    for block in model.transformer["h"]:
        normalized = F.layer_norm(
            value,
            (model.config.n_embd,),
            block.ln_1.weight,
            block.ln_1.bias,
            block.ln_1.eps,
        )
        query, key, content = F.linear(
            normalized, block.attn.c_attn.weight, block.attn.c_attn.bias
        ).chunk(3, dim=-1)
        batch, sequence, channels = query.shape
        head_size = channels // model.config.n_head

        def heads(tensor):
            return tensor.view(
                batch, sequence, model.config.n_head, head_size
            ).transpose(1, 2)

        query, key, content = map(heads, (query, key, content))
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_size)
        causal = torch.tril(torch.ones(sequence, sequence, dtype=torch.bool))
        attended = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        attended = (attended @ content).transpose(1, 2).reshape(batch, sequence, channels)
        value = value + F.linear(
            attended, block.attn.c_proj.weight, block.attn.c_proj.bias
        )
        normalized = F.layer_norm(
            value,
            (model.config.n_embd,),
            block.ln_2.weight,
            block.ln_2.bias,
            block.ln_2.eps,
        )
        hidden = F.linear(normalized, block.mlp.c_fc.weight, block.mlp.c_fc.bias)
        hidden = 0.5 * hidden * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (hidden + 0.044715 * hidden.pow(3))
            )
        )
        value = value + F.linear(
            hidden, block.mlp.c_proj.weight, block.mlp.c_proj.bias
        )
    value = F.layer_norm(
        value,
        (model.config.n_embd,),
        model.transformer["ln_f"].weight,
        model.transformer["ln_f"].bias,
        model.transformer["ln_f"].eps,
    )
    return F.linear(value, model.transformer["wte"].weight)


def test_baseline_matches_independent_upstream_gpt2_math(tiny_config):
    model = TinyLLMModel(tiny_config).eval()
    input_ids = torch.tensor([[1, 4, 2, 8], [3, 5, 7, 9]])
    actual, _ = model(input_ids, return_full_logits=True)
    expected = _upstream_reference_logits(model, input_ids)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_residual_projection_initialization_is_depth_scaled():
    config = TinyLLMConfig(
        block_size=8,
        vocab_size=32,
        n_layer=4,
        n_head=4,
        n_embd=64,
    )
    model = TinyLLMModel(config)
    residual_std = model.transformer["h"][0].attn.c_proj.weight.std().item()
    ordinary_std = model.transformer["h"][0].attn.c_attn.weight.std().item()
    assert residual_std == pytest.approx(0.02 / math.sqrt(8), rel=0.08)
    assert ordinary_std == pytest.approx(0.02, rel=0.04)


def test_feedback_is_delayed_sparse_and_zero_gate_preserves_baseline(tiny_config):
    model = TinyLLMModel(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 6))
    baseline, _ = model(input_ids, refinement_steps=0, return_full_logits=True)
    patch = model.add_feedback_connection(
        source_layer=2,
        target_layer=0,
        num_neurons=4,
        connection_density=0.25,
        gate_init=0.0,
        seed=5,
    )

    assert patch.active_connections == 32
    assert patch.source_mask.sum(dim=1).tolist() == [4, 4, 4, 4]
    assert patch.target_mask.sum(dim=0).tolist() == [4, 4, 4, 4]
    zero_gate, _ = model(input_ids, refinement_steps=1, return_full_logits=True)
    torch.testing.assert_close(zero_gate, baseline, rtol=0, atol=0)

    with torch.no_grad():
        patch.gate.fill_(0.5)
    refined, _ = model(input_ids, refinement_steps=1, return_full_logits=True)
    one_pass, _ = model(input_ids, refinement_steps=0, return_full_logits=True)
    assert not torch.equal(refined, one_pass)

    refined.sum().backward()
    assert torch.count_nonzero(patch.source_weight.grad[~patch.source_mask]) == 0
    assert torch.count_nonzero(patch.target_weight.grad[~patch.target_mask]) == 0


def test_feedback_refinement_remains_token_causal(tiny_config):
    model = TinyLLMModel(tiny_config).eval()
    model.add_feedback_connection(
        source_layer=2,
        target_layer=0,
        num_neurons=4,
        connection_density=0.5,
        gate_init=0.4,
        seed=9,
    )
    original = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 8, 9]])
    original_logits, _ = model(
        original, refinement_steps=2, return_full_logits=True
    )
    changed_logits, _ = model(
        changed_future, refinement_steps=2, return_full_logits=True
    )
    torch.testing.assert_close(original_logits[:, :2], changed_logits[:, :2])


def test_feedback_masks_round_trip_in_checkpoint(tiny_config):
    model = TinyLLMModel(tiny_config)
    original = model.add_feedback_connection(
        source_layer=2,
        target_layer=1,
        num_neurons=3,
        connection_density=0.5,
        seed=11,
    )
    state = model.state_dict()

    restored = TinyLLMModel(tiny_config)
    copied = restored.add_feedback_connection(
        source_layer=2,
        target_layer=1,
        num_neurons=3,
        connection_density=0.5,
        seed=99,
    )
    restored.load_state_dict(state)

    assert torch.equal(original.source_mask, copied.source_mask)
    assert torch.equal(original.target_mask, copied.target_mask)


def test_random_feedback_is_reproducible(tiny_config):
    first = create_tinyllm_model(
        tiny_config,
        feedback_connections=2,
        feedback_width=4,
        feedback_seed=17,
    )
    second = create_tinyllm_model(
        tiny_config,
        feedback_connections=2,
        feedback_width=4,
        feedback_seed=17,
    )
    assert first.get_feedback_topology() == second.get_feedback_topology()
    for left, right in zip(first.feedback_connections, second.feedback_connections):
        assert torch.equal(left.source_mask, right.source_mask)
        assert torch.equal(left.target_mask, right.target_mask)


def test_random_feedback_capacity_failure_is_atomic(tiny_config):
    model = TinyLLMModel(tiny_config)
    with pytest.raises(ValueError, match="only 3 unused"):
        model.add_random_feedback_connections(count=4, num_neurons=2, seed=1)
    assert len(model.feedback_connections) == 0


def test_huggingface_state_dict_round_trip_and_feedback_export_guard(tiny_config):
    original = TinyLLMModel(tiny_config).eval()
    hf_state = original.to_huggingface_state_dict()
    native_qkv = original.transformer["h"][0].attn.c_attn.weight
    assert hf_state["transformer.h.0.attn.c_attn.weight"].shape == (16, 48)
    torch.testing.assert_close(
        hf_state["transformer.h.0.attn.c_attn.weight"], native_qkv.transpose(0, 1)
    )
    restored = TinyLLMModel(tiny_config).eval()
    restored.load_huggingface_state_dict(hf_state)
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 5))

    with torch.no_grad():
        expected, _ = original(input_ids, return_full_logits=True)
        actual, _ = restored(input_ids, return_full_logits=True)
    torch.testing.assert_close(actual, expected)

    original.add_feedback_connection(
        source_layer=2,
        target_layer=0,
        num_neurons=2,
        seed=1,
    )
    with pytest.raises(RuntimeError, match="not representable"):
        original.to_huggingface_state_dict()


def test_feedback_evolver_updates_existing_optimizer(tiny_config):
    model = TinyLLMModel(tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    original_groups = len(optimizer.param_groups)
    plan = EvolutionPlan(
        {
            "type": "add_feedback_connection",
            "source_layer": 2,
            "target_layer": 0,
            "num_neurons": 4,
            "connection_density": 0.25,
            "gate_init": 0.0,
            "refinement_steps": 1,
            "seed": 3,
        }
    )

    result = FeedbackGrowthEvolver().apply_plan(plan, model, None, optimizer)

    assert result["success"]
    assert result["connections_added"] == 1
    assert result["optimizer_parameters_added"] == 129
    assert len(optimizer.param_groups) == original_groups + 1
    assert model.refinement_steps == 1


def test_topology_aware_checkpoint_reconstructs_feedback(tmp_path, tiny_config):
    source = TinyLLMModel(tiny_config).eval()
    patch = source.add_feedback_connection(
        source_layer=2,
        target_layer=0,
        num_neurons=3,
        connection_density=0.25,
        gate_init=0.2,
        seed=19,
    )
    patch.add_connections(3)
    source.refinement_steps = 2
    tokens = torch.randint(0, tiny_config.vocab_size, (2, 5))
    expected, _ = source(tokens, return_full_logits=True)
    checkpoint = source.save_checkpoint(
        tmp_path / "feedback.pt", metadata={"experiment_id": "unit-test"}
    )

    restored = TinyLLMModel.from_checkpoint(checkpoint).eval()
    actual, _ = restored(tokens, return_full_logits=True)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.refinement_steps == 2
    assert restored.get_feedback_topology() == source.get_feedback_topology()
    assert restored.checkpoint_metadata["experiment_id"] == "unit-test"


def test_topology_checkpoint_preserves_dtype_and_authoritative_metadata(
    tmp_path, tiny_config
):
    source = TinyLLMModel(tiny_config, name="SavedTinyLLM").to(torch.bfloat16)
    source.add_feedback_connection(
        source_layer=2,
        target_layer=0,
        num_neurons=2,
        seed=3,
    )
    checkpoint = source.save_checkpoint(
        tmp_path / "feedback-bf16.pt",
        metadata={"format": "not-authoritative", "path": "not-authoritative"},
    )
    restored = TinyLLMModel.from_checkpoint(checkpoint)

    assert restored.name == "SavedTinyLLM"
    assert next(restored.parameters()).dtype == torch.bfloat16
    assert restored.checkpoint_metadata["format"] == source.CHECKPOINT_FORMAT
    assert restored.checkpoint_metadata["path"] == str(checkpoint)


def test_checkpoint_rejects_unsafe_metadata_and_unknown_format(tmp_path, tiny_config):
    model = TinyLLMModel(tiny_config)
    with pytest.raises(TypeError, match="not JSON-compatible"):
        model.checkpoint_payload({"unsafe": object()})

    payload = model.checkpoint_payload()
    payload["format"] = "unknown"
    checkpoint = tmp_path / "unknown.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="Unsupported TinyLLM checkpoint format"):
        TinyLLMModel.from_checkpoint(checkpoint)


def test_causal_trainer_survives_online_feedback_growth(tiny_config):
    torch.manual_seed(7)
    model = TinyLLMModel(tiny_config)
    trainer = CausalLanguageModelTrainer(
        optimizer_kwargs={"lr": 1e-2, "weight_decay": 0.0}
    )
    batch = torch.arange(7).repeat(4, 1) % tiny_config.vocab_size
    context = EvolutionContext(device="cpu")
    first = trainer.train_step(model, batch, context)
    plan = EvolutionPlan(
        {
            "type": "add_random_feedback_connections",
            "count": 1,
            "num_neurons": 3,
            "connection_density": 0.25,
            "gate_init": 0.01,
            "refinement_steps": 1,
            "seed": 5,
        }
    )
    result = FeedbackGrowthEvolver().apply_plan(
        plan, model, trainer, trainer.optimizer
    )
    patch = model.feedback_connections[0]
    before = patch.source_weight.detach().clone()
    second = trainer.train_step(model, batch, context)

    assert result["optimizer_parameters_added"] > 0
    assert first["tokens"] == second["tokens"] == 24
    assert torch.isfinite(torch.tensor(second["loss"]))
    assert not torch.equal(before, patch.source_weight)


def test_random_strategy_emits_control_plan():
    strategy = RandomFeedbackGrowthStrategy(
        count=2,
        num_neurons=5,
        connection_density=0.1,
        seed=23,
    )
    plan = strategy.propose_plan(AnalysisReport(), EvolutionContext())

    assert plan["type"] == "add_random_feedback_connections"
    assert plan["placement"] == "random_backward"
    assert plan["seed"] == 23
    assert plan["count"] == 2


def test_feedback_endpoints_must_point_backward(tiny_config):
    model = TinyLLMModel(tiny_config)
    with pytest.raises(ValueError, match="target_block < source_block"):
        model.add_feedback_connection(
            source_layer=1,
            target_layer=2,
            num_neurons=2,
        )


def test_real_depth_endpoints_match_exact_prefixes(tiny_config):
    model = TinyLLMModel(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (3, 5))
    positions = torch.arange(input_ids.shape[1])
    embedded = model.transformer["wte"](input_ids) + model.transformer["wpe"](
        positions
    )
    prefix = embedded
    for depth in range(tiny_config.n_layer + 1):
        expected = model.lm_head(model.transformer["ln_f"](prefix)[:, [-1], :])
        actual = model.forward_at_depth(input_ids, float(depth))
        torch.testing.assert_close(actual, expected)
        if depth < tiny_config.n_layer:
            prefix = model.transformer["h"][depth](prefix)

    ordinary, _ = model(input_ids)
    torch.testing.assert_close(
        model.forward_at_depth(input_ids, float(tiny_config.n_layer)), ordinary
    )


def test_partial_block_gate_is_continuous_and_has_exact_endpoints(tiny_config):
    model = TinyLLMModel(tiny_config).eval()
    block = model.transformer["h"][0]
    value = torch.randn(2, 5, tiny_config.n_embd)

    torch.testing.assert_close(block.forward_gated(value, 0.0), value)
    torch.testing.assert_close(block.forward_gated(value, 1.0), block(value))
    below = block.forward_gated(value, 0.4999)
    above = block.forward_gated(value, 0.5001)
    assert torch.max(torch.abs(above - below)) < 1e-3
    with pytest.raises(ValueError, match="gate"):
        block.forward_gated(value, 1.1)


def test_real_depth_rejects_feedback_and_out_of_range_depth(tiny_config):
    model = TinyLLMModel(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 5))
    with pytest.raises(ValueError, match="depth"):
        model.forward_at_depth(input_ids, tiny_config.n_layer + 0.1)

    model.add_feedback_connection(
        source_block=2, target_block=0, width=2, seed=2
    )
    with pytest.raises(ValueError, match="feedback"):
        model.forward_at_depth(input_ids, 1.5)


def _write_llmc_checkpoint(
    path, model: TinyLLMModel, padded_vocab: int, version: int
) -> None:
    """Write the parameter ordering used by TinyLLM's pinned llm.c fork."""
    assert version in (3, 5)
    config = model.config
    header = np.zeros(256, dtype="<i4")
    header[:8] = [
        20240326,
        version,
        config.block_size,
        config.vocab_size,
        config.n_layer,
        config.n_head,
        config.n_embd,
        padded_vocab,
    ]

    def write(handle, tensor):
        tensor = tensor.detach().cpu()
        if version == 3:
            tensor.numpy().astype("<f4").tofile(handle)
        else:
            tensor.to(torch.bfloat16).view(torch.uint16).numpy().astype("<u2").tofile(handle)

    with path.open("wb") as handle:
        header.tofile(handle)
        write(handle, model.transformer["wte"].weight)
        padding = torch.zeros(padded_vocab - config.vocab_size, config.n_embd)
        write(handle, padding)
        write(handle, model.transformer["wpe"].weight)
        selectors = (
            lambda block: block.ln_1.weight,
            lambda block: block.ln_1.bias,
            lambda block: block.attn.c_attn.weight,
            lambda block: block.attn.c_attn.bias,
            lambda block: block.attn.c_proj.weight,
            lambda block: block.attn.c_proj.bias,
            lambda block: block.ln_2.weight,
            lambda block: block.ln_2.bias,
            lambda block: block.mlp.c_fc.weight,
            lambda block: block.mlp.c_fc.bias,
            lambda block: block.mlp.c_proj.weight,
            lambda block: block.mlp.c_proj.bias,
        )
        for select in selectors:
            for block in model.transformer["h"]:
                write(handle, select(block))
        write(handle, model.transformer["ln_f"].weight)
        write(handle, model.transformer["ln_f"].bias)


def test_llmc_v3_checkpoint_loader_preserves_logits(tmp_path, tiny_config):
    source = TinyLLMModel(tiny_config).eval()
    checkpoint = tmp_path / "tiny-model.bin"
    _write_llmc_checkpoint(checkpoint, source, padded_vocab=40, version=3)

    restored = TinyLLMModel.from_llmc_checkpoint(checkpoint).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 5))
    expected, _ = source(input_ids, return_full_logits=True)
    actual, _ = restored(input_ids, return_full_logits=True)

    torch.testing.assert_close(actual, expected)
    assert restored.checkpoint_metadata["format"] == "llm.c"
    assert restored.checkpoint_metadata["version"] == 3
    assert restored.checkpoint_metadata["padded_vocab_size"] == 40


def test_llmc_v5_bfloat16_checkpoint_loader(tmp_path, tiny_config):
    source = TinyLLMModel(tiny_config).eval()
    checkpoint = tmp_path / "tiny-model-bf16.bin"
    _write_llmc_checkpoint(checkpoint, source, padded_vocab=40, version=5)

    restored = TinyLLMModel.from_llmc_checkpoint(checkpoint).eval()
    for key, value in source.state_dict().items():
        expected = value.to(torch.bfloat16).to(torch.float32)
        torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)
    assert restored.checkpoint_metadata["version"] == 5


@pytest.mark.parametrize(
    ("magic", "version", "message"),
    [
        (0, 3, "magic number"),
        (20240326, 99, "Unsupported llm.c checkpoint version"),
    ],
)
def test_llmc_loader_rejects_unknown_headers(
    tmp_path, magic, version, message
):
    header = np.zeros(256, dtype="<i4")
    header[:8] = [magic, version, 8, 32, 3, 2, 16, 40]
    checkpoint = tmp_path / f"bad-{magic}-{version}.bin"
    header.tofile(checkpoint)
    with pytest.raises(ValueError, match=message):
        TinyLLMModel.from_llmc_checkpoint(checkpoint)


def test_llmc_loader_rejects_truncated_header(tmp_path):
    checkpoint = tmp_path / "truncated.bin"
    np.zeros(10, dtype="<i4").tofile(checkpoint)
    with pytest.raises(ValueError, match="Truncated llm.c checkpoint header"):
        TinyLLMModel.from_llmc_checkpoint(checkpoint)
