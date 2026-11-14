from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack
from experiments.structure_net.jspace_qwen_parallel_train import (
    BoundaryDataset, RecoveryConfig, SuffixTrainer, cpu_module_clone, output_objective, recursive_cpu,
)


def components():
    torch.manual_seed(73)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=36, vocab_size=16, attention_dropout=0., max_position_embeddings=16)
    config._attn_implementation = "eager"
    blocks = []
    for index in (34, 35):
        native = Qwen2DecoderLayer(config, layer_idx=index).eval().requires_grad_(False)
        basis = torch.linalg.qr(torch.randn(8, 8)).Q[:, :6]
        projection = {"rope_pairs": torch.tensor([[0]]), "value_basis": torch.eye(4)[None, :, :2]}
        block = AffineQwenBlock(native, basis, torch.randn(8, 8) / 10, head_projection=projection)
        relation = {"anchors": [0, 1], "mean": torch.zeros(12), "scale": torch.ones(12),
                    "coefficients": {"2": torch.randn(12, 6) / 100}}
        blocks.append(block.compile_polynomial(relation, [4, 5]))
    stack = CompactQwenStack(blocks, geometry_mode="fp32", pad_to_multiple=16, checkpoint_blocks=True)
    return stack, Qwen2RMSNorm(8), torch.nn.Parameter(torch.randn(16, 8) / 4, requires_grad=False), Qwen2RotaryEmbedding(config=config)


def dataset(tmp_path, original=True):
    torch.manual_seed(91)
    lengths = [5, 7, 4, 6, 5, 7, 4, 5, 7]; items = []; offset = 0
    for i, length in enumerate(lengths):
        positions = [0, (length - 1) // 3, 2 * (length - 1) // 3, length - 1]
        items.append({"identity": f"item{i}", "family": "toy", "split": "train" if i < 7 else "development",
            "input_ids": list(range(length)), "offset": offset, "stop": offset + length, "loss_positions": positions,
            "targets": [2, 3, 4, i % 2], "answer": i % 2, "choices": ["A", "B"]})
        offset += length
    boundary = torch.randn(offset, 8).bfloat16()
    accepted = torch.randn(len(items), 4, 16).bfloat16()
    tmp_path.mkdir(exist_ok=True)
    for name, value in (("boundary", boundary), ("accepted", accepted), ("original", accepted.flip(0))):
        np.save(tmp_path / (name + ".npy"), value.contiguous().view(torch.int16).numpy())
    kwargs = {"original_logits_path": tmp_path / "original.npy", "original_indices": list(reversed(range(len(items))))} if original else {}
    return BoundaryDataset(items, tmp_path / "boundary.npy", tmp_path / "accepted.npy", **kwargs)


def make_trainer(data, *, microbatch=2, accumulation=2, epochs=2):
    return SuffixTrainer(*components(), data, [0, 1], config=RecoveryConfig(dtype="float32", microbatch=microbatch,
        accumulation=accumulation, epochs=epochs, learning_rate=.001, optimizer_chunk_bytes=128))


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def test_padded_batch_preserves_each_unpadded_result_and_masks_padding(tmp_path):
    trainer = make_trainer(dataset(tmp_path))
    trainer.stack.eval()
    with torch.no_grad():
        batch = trainer.dataset.batch([0, 1], dtype=torch.float32)
        together = trainer.logits(batch)
        single = torch.cat([trainer.logits(trainer.dataset.batch([i], dtype=torch.float32)) for i in (0, 1)])
        torch.testing.assert_close(together, single, atol=2e-6, rtol=2e-5)
        batch["hidden"][0, 5:] = 1000
        torch.testing.assert_close(trainer.logits(batch), together, atol=0, rtol=0)
        assert batch["teacher"].dtype == torch.bfloat16
    trainer.dataset.close()


def test_partial_accumulation_matches_actual_three_case_window(tmp_path):
    data = dataset(tmp_path)
    batched = make_trainer(data, microbatch=3)
    accumulated = make_trainer(data, microbatch=2)
    first = batched.train_window([0, 1, 2]); second = accumulated.train_window([0, 1, 2])
    assert first["cases"] == second["cases"] == 3
    assert first["loss"] == pytest.approx(second["loss"], rel=2e-6)
    for p, q in zip(batched.parameters, accumulated.parameters):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)
    assert all(second["gradients_present_by_block"])
    assert accumulated.readout.grad is None and data.read_bytes["original"] == 0
    data.close()


def test_packed_development_does_not_mutate_live_parameters_or_master_state(tmp_path):
    trainer = make_trainer(dataset(tmp_path))
    trainer.train_window([0, 1])
    before = trainer.snapshot()
    masters = [state["master"].clone() for state in trainer.optimizer.state.values()]
    metrics, rows = trainer.packed_development()
    assert_state_equal(before["stack"], trainer.stack.state_dict())
    assert_state_equal(before["norm"], trainer.norm.state_dict())
    assert all(torch.equal(saved, state["master"]) for saved, state in zip(masters, trainer.optimizer.state.values()))
    assert metrics["count"] == 2 and len(rows) == 2 and metrics["selection_metric"] == "packed_development_ce"
    assert metrics["accepted_original_kl"] == pytest.approx(0., abs=1e-7)
    assert not trainer.stack.packed and not trainer.stack.stripped
    assert all(not p.is_inference() for p in trainer.parameters)
    trainer.dataset.close()


def test_fit_budget_selection_and_recursive_extra_state_reload(tmp_path):
    data = dataset(tmp_path / "data")
    trainer = make_trainer(data)
    report = trainer.fit(tmp_path / "fit")
    assert report["updates"] == 4 and report["completed_epochs"] == 2
    assert [w["cases"] for w in report["history"][1]["training"]] == [4, 3]
    expected_epoch = min(report["history"], key=lambda r: r["development"]["ce"])["epoch"]
    assert report["selected_epoch"] == expected_epoch
    state = torch.load(tmp_path / "fit/recovered_unpacked.pt", weights_only=True)
    assert state["stack"]["_extra_state"]["packed"] is False
    assert state["stack"]["_extra_state"]["layer_indices"] == (34, 35)
    fresh, norm, _, _ = components()
    fresh.load_state_dict(state["stack"], strict=True); norm.load_state_dict(state["norm"], strict=True)
    assert_state_equal(fresh.state_dict(), trainer.stack.state_dict())
    with pytest.raises(RuntimeError, match="Recovery finished"):
        trainer.train_window([0])
    data.close()


def test_objective_is_only_fixed_output_kl_plus_ce():
    torch.manual_seed(8)
    logits = torch.randn(2, 4, 11, requires_grad=True)
    teacher = torch.randn_like(logits).bfloat16(); targets = torch.randint(11, (2, 4))
    loss, kl, ce = output_objective(logits, teacher, targets, RecoveryConfig())
    reference = torch.nn.functional.kl_div((logits / 2).log_softmax(-1), (teacher.float() / 2).softmax(-1), reduction="none").sum(-1).mean()
    torch.testing.assert_close(kl, reference, atol=2e-7, rtol=1e-6)
    torch.testing.assert_close(loss, 2.8 * reference + .3 * torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten()))
    loss.backward(); assert torch.isfinite(logits.grad).all()


def test_dataset_rejects_heldout_rows_and_ambiguous_original_alignment(tmp_path):
    data = dataset(tmp_path)
    items = deepcopy(data.items); items[0]["split"] = "evaluation"
    with pytest.raises(ValueError, match="Held-out"):
        BoundaryDataset(items, tmp_path / "boundary.npy", tmp_path / "accepted.npy")
    with pytest.raises(ValueError, match="Explicit original-teacher"):
        BoundaryDataset(data.items, tmp_path / "boundary.npy", tmp_path / "accepted.npy", original_logits_path=tmp_path / "original.npy")
    trainer = make_trainer(data)
    with pytest.raises(ValueError, match="training cases only"):
        trainer.train_window(data.development)
    data.close()


def test_cpu_clone_and_recursive_state_own_their_storage():
    stack, *_ = components()
    clone = cpu_module_clone(stack)
    state = recursive_cpu(stack.state_dict())
    original = next(stack.parameters()); copied = next(clone.parameters())
    assert copied.data_ptr() != original.data_ptr()
    with torch.no_grad(): copied.add_(5)
    assert not torch.equal(copied, original)
    assert state["_extra_state"] == stack.get_extra_state()


def test_bf16_real_qwen_components_train_polynomial_bias_and_pack_norm_consistently(tmp_path):
    data = dataset(tmp_path)
    stack, norm, readout, rotary = components()
    readout.data = readout.data.bfloat16()
    trainer = SuffixTrainer(stack, norm, readout, rotary, data, [0, 1], config=RecoveryConfig(
        microbatch=2, accumulation=2, epochs=1, optimizer_chunk_bytes=128))
    result = trainer.train_window([0, 1, 2, 3])
    assert all(result["gradients_present_by_block"])
    assert all(p.dtype == torch.bfloat16 for p in trainer.parameters)
    assert trainer.norm.weight.dtype == trainer.readout.dtype == torch.bfloat16
    for block in trainer.stack.blocks:
        assert block.polynomial_bias in trainer.optimizer.state
        assert torch.count_nonzero(trainer.optimizer.state[block.polynomial_bias]["exp_avg"]) > 0
        assert block.anchor.dtype == block.anchor_energy.dtype == torch.float32
    metrics, rows = trainer.packed_development()
    assert metrics["precision"] == "bfloat16" and metrics["count"] == 2
    assert len(rows) == 2 and torch.isfinite(torch.tensor(metrics["ce"]))
    assert all(not p.is_inference() for p in trainer.parameters)
    data.close()
