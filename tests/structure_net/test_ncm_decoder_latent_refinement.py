from copy import deepcopy
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_decoder_latent_refinement as run


def tiny_codec():
    spec = importlib.util.spec_from_file_location("latent_refinement_test_model",
        Path("tests/structure_net/test_ncm_generated_history_core.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forecaster, original, *_ = module.model()
    return original, forecaster.generator.vqvae_local[0]


def test_refinement_reduces_word_loss_without_updating_codec_or_initial_latent():
    torch.manual_seed(99)
    original, facade = tiny_codec()
    config = run.Config(updates=8, word_vocab=31, pad=31, lr=.05)
    words = torch.randint(0, 30, (1, facade.context_length))
    words[0, -8:] = config.pad
    initial = run.encoder_latent(facade, words)
    before = deepcopy(original.state_dict())
    copy = initial.clone()
    with torch.no_grad():
        loss_before = F.cross_entropy(run.neural_logits(facade, initial, config).flatten(0, 1),
            words.flatten(), ignore_index=config.pad)
    saved, trace = run.refine(facade, initial, words, 8, config)
    with torch.no_grad():
        loss_after = F.cross_entropy(run.neural_logits(facade, saved["refined"], config).flatten(0, 1),
            words.flatten(), ignore_index=config.pad)
    assert loss_after < loss_before
    assert torch.equal(initial, copy)
    assert not torch.equal(saved["refined"], copy)
    assert run.same(before, original.state_dict())
    assert all(p.grad is None for p in original.parameters())
    assert len(trace) == 8 and all(row["gradient_norm"] > 0 for row in trace)
    assert int(saved["optimizer"]["state"][0]["step"]) == 8


def test_input_gradient_flows_through_frozen_decoder_and_training_codec_is_rejected():
    original, facade = tiny_codec()
    config = run.Config(word_vocab=31, pad=31)
    value = torch.randn(1, facade.context_length, facade.embed_dim, requires_grad=True)
    run.neural_logits(facade, value, config).square().mean().backward()
    assert value.grad is not None and torch.count_nonzero(value.grad)
    assert all(p.grad is None for p in original.parameters())
    original.train()
    with pytest.raises(ValueError, match="frozen"):
        run.neural_logits(facade, value, config)


def test_refinement_rejects_targetless_incomplete_or_out_of_budget_calls():
    _, facade = tiny_codec()
    config = run.Config(word_vocab=31, pad=31)
    value = torch.randn(1, facade.context_length, facade.embed_dim)
    targets = torch.full((1, facade.context_length), 31)
    with pytest.raises(ValueError, match="active targets"):
        run.refine(facade, value, targets, 2, config)
    targets[0, 0] = 2
    for updates in (0, 33):
        with pytest.raises(ValueError, match="fixed budget"):
            run.refine(facade, value, targets, updates, config)
    with pytest.raises(ValueError, match="complete"):
        run.refine(facade, value, targets[:, :-1], 2, config)


def test_exact_latent_optimizer_serialization_and_next_update(tmp_path):
    latent = torch.ones(1, 3, 2, requires_grad=True)
    opt = run.optimizer(latent)
    latent.square().mean().backward()
    opt.step()
    path = tmp_path / "latent.pt"
    torch.save(run.to_cpu({"initial": latent.detach() + 1, "refined": latent, "optimizer": opt.state_dict()}), path)
    saved = torch.load(path, weights_only=True)
    restored = saved["refined"].clone().requires_grad_(True)
    other = run.optimizer(restored)
    other.load_state_dict(saved["optimizer"])
    assert run.same(opt.state_dict(), other.state_dict())
    for value, optimizer in ((latent, opt), (restored, other)):
        optimizer.zero_grad(set_to_none=True)
        value.square().mean().backward()
        optimizer.step()
    assert torch.equal(latent, restored)
    assert run.same(opt.state_dict(), other.state_dict())


def saved_measurement():
    target = torch.tensor([[7, 50256] + [50257] * 190])
    initial = torch.zeros(1, 192, 384)
    latent = initial.clone().requires_grad_(True)
    opt = run.optimizer(latent)
    latent.sum().backward()
    opt.step()
    saved = run.to_cpu({"initial": initial, "refined": latent, "optimizer": opt.state_dict()})
    trace = [{"step": 1, "word_CE": .1, "anchor_MSE": 0., "gradient_norm": 1.,
        "target_word_sha256": run.clean.recovery.study.base.tensor_digest({"target": target}), "lr": run.Config().lr}]
    return saved, trace, target


def test_saved_trace_binds_labels_and_full_fixed_optimizer_budget():
    saved, trace, target = saved_measurement()
    run.validate_saved(saved, trace, target, 1)
    with pytest.raises(ValueError, match="optimizer step"):
        run.validate_saved(saved, trace, target, 2)
    trace[0]["target_word_sha256"] = "changed"
    with pytest.raises(ValueError, match="actual target"):
        run.validate_saved(saved, trace, target, 1)


def test_raw_replay_checks_complete_words_and_actual_latent_bytes():
    saved, trace, target = saved_measurement()
    source = {"id": "a", "word_ids": target[0].tolist(), "target_text": "w7",
        "source_sha256": "source", "diagnostic_group": "original_16"}
    arms = {}
    for arm, key in zip(run.ARMS, ("initial", "refined")):
        arms[arm] = {"raw_word_ids": [7, 50256] + [8] * 190, "correct": 2, "total": 2,
            "text": "w7", "global_EOS_position": 1, "exact_target_text": True, "CE": .1,
            "logits_sha256": "a" * 64,
            "latent_sha256": run.clean.recovery.study.base.tensor_digest({"latent": saved[key]})}
    item = {"id": "a", "target_word_ids": source["word_ids"], "target_text": "w7", "source_sha256": "source",
        "diagnostic_group": "original_16", "arms": arms, "trace": trace,
        "final_anchor_MSE": float(F.mse_loss(saved["refined"], saved["initial"]))}
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " ".join(f"w{i}" for i in ids))
    run.replay_case(item, source, saved, 1, tokenizer)
    changed = deepcopy(saved)
    changed["refined"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="latent digest"):
        run.replay_case(item, source, changed, 1, tokenizer)
    item["arms"][run.ARMS[1]]["raw_word_ids"][0] = 9
    with pytest.raises(ValueError, match="raw decoder"):
        run.replay_case(item, source, saved, 1, tokenizer)


def components(first=199, second=199):
    return {name: {"cases": count, "arms": {run.ARMS[1]: {"correct": correct, "total": total}}}
        for name, count, correct, total in (("original_16", 16, first, 200),
            ("additional_16", 16, second, 200), ("all_32", 32, first + second, 400))}


def test_gate_requires_inclusive_threshold_in_each_fixed_group():
    assert run.fidelity_gate(components())
    assert not run.fidelity_gate(components(198, 200))
    broken = components()
    broken["all_32"]["arms"][run.ARMS[1]]["total"] -= 1
    with pytest.raises(ValueError, match="compose"):
        run.fidelity_gate(broken)


def test_config_is_frozen_and_device_contract_forbids_silent_fallback(monkeypatch):
    with pytest.raises(FrozenInstanceError):
        run.Config().updates = 2
    with pytest.raises(ValueError):
        run.Config(anchor_weight=0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="physicalGPU2"):
        run.gpu()


def test_targets_remain_exact_ordered_train_members_without_loading_features():
    training = [{"id": str(i), "source_sha256": "source" + str(i), "target_text": "target" + str(i),
        "word_ids": [i, 50256] + [50257] * 190} for i in range(690)]
    selection, measured = [], []
    for i, row in enumerate(training[:32]):
        group = "original_16" if i < 16 else "additional_16"
        selection.append({"id": row["id"], "diagnostic_group": group, "source_sha256": row["source_sha256"]})
        measured.append({**row, "diagnostic_group": group, "target_word_ids": row["word_ids"]})
    assert [r["id"] for r in run.bound_targets(training, selection, measured)] == [str(i) for i in range(32)]
    changed = deepcopy(measured)
    changed[0]["target_word_ids"][0] += 1
    with pytest.raises(ValueError, match="target differs"):
        run.bound_targets(training, selection, changed)
    selection[0]["id"] = "689"
    with pytest.raises(ValueError, match="ordered32"):
        run.bound_targets(training, selection, measured)
