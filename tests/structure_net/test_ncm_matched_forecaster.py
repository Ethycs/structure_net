from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import ncm_matched_forecaster as run
from experiments.neural_architecture_lab import finalize_ncm_matched_forecaster as audit


def upstream(tmp_path, monkeypatch, gate=True):
    flow, archive = tmp_path / "upstream_flow", tmp_path / "upstream_archive"
    flow.mkdir(); archive.mkdir()
    monkeypatch.setattr(run.clean, "WORKFLOW", flow)
    monkeypatch.setattr(run.clean, "RESULTS", archive)
    classification = ("clean_word_fit_passed_matched_forecaster_still_required"
                      if gate else "clean_word_fit_gate_failed")
    campaign = {"classification": classification, "training_cases": 690,
        "semantic_quality_measured": False, "final_test_opened": False, "clean_fit_gate": gate}
    correct = 199 if gate else 198
    campaign["fitting"] = {"components": {
        "original_16": {"cases": 16, "arms": {"correct_history": {"correct": correct, "total": 200}}},
        "additional_16": {"cases": 16, "arms": {"correct_history": {"correct": correct, "total": 200}}},
        "all_32": {"cases": 32, "arms": {"correct_history": {"correct": 2 * correct, "total": 400}}},
    }}
    path = archive / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    digest = run.file_hash(path)
    parent = {"status": "clean_word_component_completed", "clean_fit_gate": gate,
        "campaign_sha256": digest, "stages": [{"stage": str(i), "exit_code": 0} for i in range(6)]}
    (flow / "result.json").write_text(json.dumps(parent))
    receipt = {"status": "registered_and_read_back", "record": {
        "campaign_sha256": digest, "confirmation_status": classification, "clean_fit_gate": gate}}
    (archive / (run.clean.HYPOTHESIS + "_registry_readback.json")).write_text(json.dumps(receipt))
    return flow, archive


@pytest.mark.parametrize("gate", [True, False])
def test_admission_uses_registered_upstream_gate(tmp_path, monkeypatch, gate):
    upstream(tmp_path, monkeypatch, gate)
    assert run.admission() is gate
    if not gate:
        with pytest.raises(ValueError):
            run.require_admission()


def test_admission_is_pending_and_rejects_inconsistent_or_nonzero_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(run.clean, "WORKFLOW", tmp_path / "missing")
    assert run.admission() is None
    flow, _ = upstream(tmp_path, monkeypatch)
    path = flow / "result.json"
    parent = json.loads(path.read_text())
    parent["clean_fit_gate"] = False
    path.write_text(json.dumps(parent))
    with pytest.raises(ValueError):
        run.admission()
    parent["clean_fit_gate"] = True
    parent["stages"][0]["exit_code"] = 1
    path.write_text(json.dumps(parent))
    with pytest.raises(ValueError):
        run.admission()


def tiny_model():
    spec = importlib.util.spec_from_file_location(
        "matched_forecaster_test_model", Path("tests/structure_net/test_ncm_generated_history_core.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model()


@pytest.mark.parametrize("extend_prefix", [False, True])
def test_concept_training_has_no_current_or_future_level_label_leakage(extend_prefix):
    net, _, features, mask, source_ids = tiny_model()
    if extend_prefix:
        features = torch.nn.functional.pad(features, (0, 0, 0, 8))
        mask = torch.nn.functional.pad(mask, (0, 8))
        source_ids = torch.nn.functional.pad(source_ids, (0, 8), value=31)
        run.set_source_frame(net.generator, 16, maximum=8192)
    codes = torch.randint(0, 16, (2, sum(net.generator.levels)))
    with torch.no_grad():
        expected = run.core.concept_logits(net, features, mask, codes, source_ids)
        position = 0
        for level, width in enumerate(net.generator.levels[:-1]):
            changed = codes.clone()
            changed[:, position:] = (changed[:, position:] + 1) % 16
            actual = run.core.concept_logits(net, features, mask, changed, source_ids)
            torch.testing.assert_close(actual[level], expected[level], atol=0, rtol=0)
            position += width


def test_concept_gradient_scope_keeps_word_and_codec_weights_fixed():
    net, codec, features, mask, source_ids = tiny_model()
    run.core.concept_scope(net)
    fixed = {name: value.detach().clone() for name, value in net.named_parameters() if not value.requires_grad}
    codes = torch.randint(0, 16, (2, sum(net.generator.levels)))
    words = torch.randint(0, 30, (2, net.generator.levels[-1]))
    logits = run.core.concept_logits(net, features, mask, codes, source_ids)
    loss, _ = run.core.concept_loss(logits, codes, words, net.generator.levels)
    loss.backward()
    assert features.grad is None
    assert net.source_head.source_projection.weight.grad.abs().sum() > 0
    assert net.generator.l0_queries.grad.abs().sum() > 0
    assert all(head.proj.weight.grad.abs().sum() > 0 for head in net.generator.lm_heads[:-1])
    assert all(value.grad is None for value in net.word_projection.parameters())
    assert all(value.grad is None for value in net.word_decoder.parameters())
    assert all(value.grad is None for value in codec.parameters())
    optimizer = torch.optim.AdamW([value for value in net.parameters() if value.requires_grad], lr=1e-3)
    optimizer.step()
    assert all(torch.equal(value, dict(net.named_parameters())[name]) for name, value in fixed.items())


def summary(original=99, additional=99):
    return {group: {"cases": cases, "levels": [
        {"level": level, "active_correct": correct, "active_total": total} for level in range(8)]}
        for group, cases, correct, total in (
            ("original_16", 16, original, 100), ("additional_16", 16, additional, 100),
            ("all_32", 32, original + additional, 200))}


def test_code_gate_is_inclusive_at_every_level_and_group():
    assert run.fitting_gate(summary())
    assert not run.fitting_gate(summary(98, 100))
    assert not run.fitting_gate(summary(100, 98))
    values = summary()
    values["all_32"]["levels"][3]["active_correct"] += 1
    with pytest.raises(ValueError):
        run.fitting_gate(values)


def test_active_code_counting_includes_eos_and_excludes_padding():
    row = {"word_ids": [7] * 47 + [50256] + [50257] * 144, "concept_ids": [0] * 800}
    codes = [0] * 608
    codes[0] = codes[4] = 1
    assert run.count_codes(codes, row)[0] == {"level": 0, "active_correct": 3, "active_total": 4}
    row["word_ids"] = [7] * 48 + [50256] + [50257] * 143
    assert run.count_codes(codes, row)[0] == {"level": 0, "active_correct": 6, "active_total": 8}
    with pytest.raises(ValueError):
        run.count_codes([0.] * 608, row)


def test_prediction_history_excludes_finest_labels_without_mutating_targets():
    codes = torch.arange(800).reshape(1, 800)
    expected = codes.clone()
    actual = run.prediction_history(codes)
    assert torch.equal(actual[:, :608], codes[:, :608])
    assert not torch.count_nonzero(actual[:, 608:])
    assert torch.equal(codes, expected)


def replay_fixture(monkeypatch):
    config = run.Config(updates=4, warmup=1, snapshot_interval=2)
    monkeypatch.setattr(run, "Config", lambda: config)
    rows = [{"id": "a", "source_frame": 256, "concept_ids": list(range(800)),
             "word_ids": [7, 50256] + [50257] * 190}]
    codes = torch.tensor([rows[0]["concept_ids"]])
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    noise = torch.Generator().manual_seed(config.noise_seed)
    concept_hash = run.clean.recovery.study.base.tensor_digest({"codes": codes})
    word_hash = run.clean.recovery.study.base.tensor_digest({"words": torch.tensor([rows[0]["word_ids"]])})
    trace = []
    for step in range(1, 5):
        torch.randint(1, (1,), generator=sampler)
        corrupted = bool(torch.rand((), generator=noise) < config.corrupt_probability)
        history = run.core.uniform_history(codes, run.WIDTHS + (192,), noise,
            config.maximum_uniform_rate) if corrupted else codes
        history = run.prediction_history(history)
        trace.append({"step": step, "id": "a", "source_frame": 256, "corrupted": corrupted,
            "history_sha256": run.clean.recovery.study.base.tensor_digest({"codes": history}),
            "target_concept_sha256": concept_hash, "target_word_sha256": word_hash,
            "lr": run.clean.rate(step, config), "loss": 1.125, "gradient_norm": 2.,
            "levels": [{"level": level, "active_CE": 1., "padded_CE": .5} for level in range(8)]})
    state = {"step": 4, "config": asdict(config),
        "initial_forecaster_checkpoint_sha256": run.clean.INITIAL_FORECASTER,
        "sampler_rng": sampler.get_state(), "noise_rng": noise.get_state(),
        "optimizer": {"state": {0: {"step": torch.tensor(4.)}}, "param_groups": [{
            "betas": (config.beta1, config.beta2), "weight_decay": config.weight_decay, "eps": config.optimizer_eps}]}}
    return rows, trace, state


def test_trace_replay_requires_exact_corruption_masks_targets_and_weighted_loss(monkeypatch):
    rows, trace, state = replay_fixture(monkeypatch)
    assert audit.replay_training(rows, trace, state)["sampler_noise_and_optimizer_steps_exact"]
    for field, value in (("history_sha256", "wrong"), ("target_word_sha256", "wrong"), ("loss", 9.)):
        changed = deepcopy(trace)
        changed[1][field] = value
        with pytest.raises(ValueError):
            audit.replay_training(rows, changed, state)
    changed_state = deepcopy(state)
    changed_state["noise_rng"] = torch.Generator().manual_seed(999).get_state()
    with pytest.raises(ValueError):
        audit.replay_training(rows, trace, changed_state)


def test_nonfinite_generation_is_invalid_instead_of_becoming_an_argmax():
    assert run.require_finite_logits(None, None, torch.tensor([1., 2.])) is None
    with pytest.raises(ValueError, match="nonfinite"):
        run.require_finite_logits(None, None, torch.tensor([float("nan")]))
