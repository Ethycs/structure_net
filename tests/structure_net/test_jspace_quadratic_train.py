"""Real small Qwen blocks exercise frozen-prefix sequential training contracts."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_quadratic_qwen import QuadraticQwenBlock, FoldedReadout
from experiments.structure_net.jspace_quadratic_train import (
    CoordinateDataset, SequentialConfig, SequentialTrainer, feature_loss, logit_kl,
)


def components(layer=0, mode="quadratic"):
    torch.manual_seed(71)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=36, vocab_size=16, attention_dropout=0., max_position_embeddings=16)
    config._attn_implementation = "eager"
    native = Qwen2DecoderLayer(config, layer_idx=layer).eval().requires_grad_(False)
    basis = torch.linalg.qr(torch.randn(8, 8)).Q[:, :4]
    projection = {"rope_pairs": torch.tensor([[0]]), "value_basis": torch.eye(4)[None, :, :2]}
    affine = AffineQwenBlock(native, basis, torch.randn(1, 8) / 10, trainable=True,
        parameter_dtype=torch.float32, head_projection=projection)
    block = QuadraticQwenBlock(affine, mode=mode, learn_metric=True)
    readout = FoldedReadout(torch.randn(16, 8) / 4, affine.basis, affine.anchor, torch.ones(8), 1e-6) if layer == 35 else None
    return block, Qwen2RotaryEmbedding(config=config), readout


def dataset(path, block, final=False):
    torch.manual_seed(53)
    lengths = [5, 7, 4, 6, 5, 7, 4, 5, 7]
    items = []; offset = 0
    for i, length in enumerate(lengths):
        positions = [0, (length - 1)//3, 2*(length - 1)//3, length - 1]
        items.append(dict(identity=f"case{i}", family="toy", split="train" if i < 7 else "development",
            input_ids=list(range(length)), offset=offset, stop=offset+length, loss_positions=positions,
            targets=[2, 3, 4, i % 2], choices=["a", "b"], answer=i % 2))
        offset += length
    path.mkdir(parents=True, exist_ok=True)
    coordinates = torch.randn(offset, 4)
    native = torch.randn(offset, 8).bfloat16()
    logits = torch.randn(len(items), 4, 16).bfloat16()
    np.save(path / "coordinates.npy", coordinates.numpy())
    np.save(path / "teacher.npy", native.contiguous().view(torch.int16).numpy())
    np.save(path / "logits.npy", logits.contiguous().view(torch.int16).numpy())
    return CoordinateDataset(items, path / "coordinates.npy", path / "teacher.npy", basis=block.basis,
        anchor=block.anchor, teacher_logits_path=path / "logits.npy" if final else None)


def trainer(path, *, layer=0, microbatch=2, accumulation=2, epochs=2, mode="quadratic"):
    block, rotary, readout = components(layer, mode)
    data = dataset(path, block, final=layer == 35)
    return SequentialTrainer(block, rotary, data, readout=readout, label_ids=[0, 1], config=SequentialConfig(
        epochs=epochs, microbatch=microbatch, accumulation=accumulation,
        learning_rate=.001, optimizer_chunk_bytes=128))


def assert_state_equal(first, second):
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], torch.Tensor):
            assert torch.equal(first[key], second[key]), key
        else:
            assert first[key] == second[key]


def test_recipe_defaults_and_train_only_projection_normalizer(tmp_path):
    config = SequentialConfig()
    assert (config.epochs, config.microbatch, config.accumulation, config.learning_rate, config.temperature) == (12, 4, 4, 1.2e-4, 2.)
    model = trainer(tmp_path)
    data = model.dataset
    native = torch.from_numpy(np.array(data.teacher, copy=True)).view(torch.bfloat16).float()
    train_stop = data.items[data.training[-1]]["stop"]
    projected = (native[:train_stop] - data.anchor[0]) @ data.basis
    expected = float(projected.double().square().mean())
    assert data.fit_normalizer(microbatch=2) == pytest.approx(expected, rel=2e-7)
    assert data.read_bytes["coordinates"] == data.read_bytes["teacher_logits"] == 0
    assert data.read_bytes["teacher_boundary"] == train_stop * 8 * 2
    assert not data.basis.requires_grad and not data.anchor.requires_grad
    data.close()


@pytest.mark.parametrize("mode", ["quadratic", "softmax"])
def test_padding_does_not_change_valid_outputs_or_feature_weighting(tmp_path, mode):
    model = trainer(tmp_path, mode=mode)
    model.block.eval()
    with torch.no_grad():
        batch = model.dataset.batch([0, 1])
        outputs = model.forward(batch)
        original = outputs.clone()
        for row, index in enumerate((0, 1)):
            single = model.forward(model.dataset.batch([index]))
            torch.testing.assert_close(outputs[row, :single.shape[1]], single[0], atol=2e-6, rtol=2e-5)
        batch["coordinates"][0, 5:] = 1e5
        changed = model.forward(batch)
        torch.testing.assert_close(changed[0, :5], original[0, :5], atol=0, rtol=0)
        a, _, _, _ = feature_loss(original, batch, 1., final_layer=False)
        b, _, _, _ = feature_loss(changed, batch, 1., final_layer=False)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    model.dataset.close()


@pytest.mark.parametrize("layer", [0, 35])
def test_partial_accumulation_matches_actual_variable_length_window(tmp_path, layer):
    whole = trainer(tmp_path / "whole", layer=layer, microbatch=3)
    split = trainer(tmp_path / "split", layer=layer, microbatch=2)
    a, b = whole.train_window([0, 1, 2]), split.train_window([0, 1, 2])
    assert a["cases"] == b["cases"] == 3
    assert a["feature_positions"] == (12 if layer == 35 else 16)
    assert a["loss"] == pytest.approx(b["loss"], rel=4e-6)
    for p, q in zip(whole.parameters, split.parameters):
        torch.testing.assert_close(p, q, atol=3e-7, rtol=3e-5)
    assert all(v == 1 for v in split.gradient_updates.values())
    assert not any(p.requires_grad for p in split.rotary.parameters())
    whole.dataset.close(); split.dataset.close()


def test_final_objective_has_feature_and_full_vocab_kl_but_no_ce(tmp_path):
    model = trainer(tmp_path, layer=35)
    batch = model.dataset.batch([0, 1])
    values = model.losses(batch)
    reference = F.kl_div((values["logits"] / 2).log_softmax(-1), (batch["teacher_logits"] / 2).softmax(-1),
        reduction="none").sum(-1).mean()
    torch.testing.assert_close(values["kl"], reference, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(values["loss"], values["feature_mse"] / model.dataset.normalizer + 4 * reference)
    model.dataset.items[0]["answer"] = 1 - model.dataset.items[0]["answer"]
    model.dataset.items[0]["targets"] = [15] * 4
    changed = model.losses(batch)
    torch.testing.assert_close(values["loss"], changed["loss"], atol=0, rtol=0)
    before = model.readout.weight.detach().clone()
    model.train_window([0, 1])
    assert not torch.equal(before, model.readout.weight)
    assert model.gradient_updates["readout.weight"] == model.gradient_updates["readout.bias"] == 1
    assert model.gradient_updates["block.metric.log_metric"] == 1
    model.dataset.close()


def test_final_feature_energy_uses_exact_four_teacher_logit_positions(tmp_path):
    model = trainer(tmp_path, layer=35)
    data = model.dataset
    selected = []
    native = torch.from_numpy(np.array(data.teacher, copy=True)).view(torch.bfloat16).float()
    for i in data.training:
        case = data.items[i]
        positions = torch.tensor(case["loss_positions"]) + case["offset"]
        selected.append((native[positions] - data.anchor[0]) @ data.basis)
    expected = float(torch.cat(selected).double().square().mean())
    assert data.fit_normalizer(microbatch=2) == pytest.approx(expected, rel=2e-7)
    data.close()


@pytest.mark.parametrize("layer", [0, 35])
def test_budget_selection_disk_reload_and_freeze_current_stage(tmp_path, layer):
    model = trainer(tmp_path / "data", layer=layer)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path / "data").glob("*.npy")}
    report = model.fit(tmp_path / "fit")
    assert report["completed_epochs"] == 2 and report["updates"] == 4
    assert [u["cases"] for u in report["history"][1]["training"]] == [4, 3]
    assert report["selected_epoch"] == min(report["history"], key=lambda r: r["development"]["loss"])["epoch"]
    assert report["disk_reload_exact"]
    assert not any(p.requires_grad for p in model.parameters)
    assert all(p["gradient_updates"] == p["optimizer_updates"] == 4 for p in report["trainable_parameters"].values())
    state = torch.load(tmp_path / "fit/selected.pt", map_location="cpu", weights_only=True)
    fresh, _, readout = components(layer)
    fresh.load_state_dict(state["block"], strict=True)
    assert_state_equal(fresh.state_dict(), model.block.state_dict())
    if readout is not None:
        readout.load_state_dict(state["readout"], strict=True)
        assert_state_equal(readout.state_dict(), model.readout.state_dict())
    assert json.loads((tmp_path / "fit/epoch0_development.json").read_text())["metrics"] == report["initial"]
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path / "data").glob("*.npy")}
    with pytest.raises(RuntimeError, match="finished"):
        model.train_window([0])
    model.dataset.close()


def test_heldout_updates_and_wrong_geometry_are_rejected(tmp_path):
    model = trainer(tmp_path)
    data = model.dataset
    with pytest.raises(ValueError, match="training cases only"):
        model.train_window(data.development)
    items = deepcopy(data.items); items[0]["split"] = "evaluation"
    with pytest.raises(ValueError, match="Held-out"):
        CoordinateDataset(items, tmp_path / "coordinates.npy", tmp_path / "teacher.npy", basis=data.basis, anchor=data.anchor)
    other, rotary, _ = components()
    data.basis[0, 0] += .1
    with pytest.raises(ValueError, match="exact fixed geometry"):
        SequentialTrainer(other, rotary, data)
    data.close()
