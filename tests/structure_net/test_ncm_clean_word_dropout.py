from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.structure_net import ncm_clean_word_dropout as run
from experiments.neural_architecture_lab import finalize_ncm_clean_word_dropout as audit


class Attention(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.dropout = probability
        self.residual = nn.Dropout(probability)
        self.projection = nn.Linear(8, 8)

    def forward(self, x):
        x = self.projection(x)
        return self.residual(F.scaled_dot_product_attention(x, x, x,
            dropout_p=self.dropout if self.training else 0.))


def stub():
    branch = nn.Module()
    branch.generator = nn.Module()
    branch.generator.transformer = Attention(.3)
    branch.generator.prefix_len = 8
    branch.word_decoder = Attention(.1)
    return branch


def test_zero_dropout_covers_functional_sdpa_and_modules_without_changing_eval_or_weights():
    branch = stub().eval()
    source = torch.randn(2, 4, 8)
    weights = deepcopy(branch.state_dict())
    expected = branch.word_decoder(branch.generator.transformer(source))
    maps = run.apply_dropout(branch, "zero_dropout")
    assert set(maps["original"].values()) == {.1, .3}
    assert set(maps["applied"].values()) == {0.}
    assert len(maps["original"]) == 4
    assert torch.equal(expected, branch.word_decoder(branch.generator.transformer(source)))
    assert all(torch.equal(value, branch.state_dict()[key]) for key, value in weights.items())
    branch.train()
    first = branch.word_decoder(branch.generator.transformer(source))
    assert torch.equal(first, branch.word_decoder(branch.generator.transformer(source)))


def test_original_dropout_remains_stochastic_and_bad_original_metadata_is_rejected():
    branch = stub().train()
    maps = run.apply_dropout(branch, "original_dropout")
    assert maps["applied"] == maps["original"]
    source = torch.randn(2, 4, 8)
    assert not torch.equal(branch.generator.transformer(source), branch.generator.transformer(source))
    before = run.dropout_map(branch)
    with pytest.raises(ValueError):
        run.apply_dropout(branch, "zero_dropout", {"invalid": .1})
    assert run.dropout_map(branch) == before


def tiny_model():
    spec = importlib.util.spec_from_file_location("dropout_ncm_test_model",
        Path("tests/structure_net/test_ncm_generated_history_core.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model()


def test_real_ncm_zero_dropout_retains_word_gradients_and_frozen_modules():
    fixed, codec, features, mask, source_ids = tiny_model()
    branch = run.clean.recovery.study.core.independent_word_branch(fixed).train()
    original_forecaster = deepcopy(fixed.state_dict())
    original_codec = deepcopy(codec.state_dict())
    run.apply_dropout(branch, "zero_dropout")
    widths = branch.generator.levels
    batch = {"features": features, "mask": mask, "source_ids": source_ids,
        "codes": torch.randint(0, 16, (2, sum(widths))), "words": torch.randint(0, 30, (2, widths[-1]))}
    optimizer = torch.optim.AdamW([v for v in branch.parameters() if v.requires_grad], lr=1e-3)
    assert run.clean.clean_backward(branch, batch) > 0
    assert features.grad is None
    assert all(v.grad is None for v in fixed.parameters())
    assert all(v.grad is None for v in codec.parameters())
    assert all(v.grad is None for v in branch.parameters() if not v.requires_grad)
    assert branch.source_head.source_projection.weight.grad.abs().sum() > 0
    assert branch.word_projection.weight.grad.abs().sum() > 0
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in branch.word_decoder.parameters())
    optimizer.step()
    assert all(torch.equal(value, fixed.state_dict()[key]) for key, value in original_forecaster.items())
    assert all(torch.equal(value, codec.state_dict()[key]) for key, value in original_codec.items())


def test_checkpoint_roundtrip_requires_exact_arm_and_nonpersistent_dropout_metadata(tmp_path, monkeypatch):
    branch = stub()
    maps = run.apply_dropout(branch, "zero_dropout")
    optimizer = torch.optim.AdamW(branch.parameters())
    sampler = torch.Generator().manual_seed(run.Config().sampler_seed)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    producer = {"fingerprint": "producer", "config": {"initial_word_checkpoint_sha256": "initializer"}}
    path = tmp_path / "model.pt"
    run.save(path, "zero_dropout", branch, optimizer, sampler, 2, maps, producer)
    state = torch.load(path, weights_only=True)
    assert set(state["parameter_names"]) == set(dict(branch.named_parameters()))
    run.validate_checkpoint(state, "zero_dropout", maps, 2, "initializer")
    fresh = stub()
    restored_maps = run.apply_dropout(fresh, state["arm"], maps["original"])
    fresh.load_state_dict(state["model"], strict=True)
    assert restored_maps == maps and run.dropout_map(fresh) == maps["applied"]
    for key, changed in (("arm", "original_dropout"), ("step", 1),
            ("initial_word_checkpoint_sha256", "wrong"), ("dropout_maps", {"original": {}, "applied": {}})):
        broken = {**state, key: changed}
        with pytest.raises(ValueError):
            run.validate_checkpoint(broken, "zero_dropout", maps, 2, "initializer")
    with pytest.raises(ValueError):
        run.save(path, "zero_dropout", branch, optimizer, sampler, 2, maps, producer)


def test_upstream_gate_distinguishes_pending_positive_negative_and_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(run.forecaster, "WORKFLOW", tmp_path)
    monkeypatch.setattr(run.forecaster, "admission", lambda: None)
    assert run.admission() is None
    monkeypatch.setattr(run.forecaster, "admission", lambda: True)
    assert run.admission() is False
    monkeypatch.setattr(run.forecaster, "admission", lambda: False)
    assert run.admission() is None
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"status": "not_admitted_clean_word_fit_gate_failed",
        "stages": [], "model_training_executed": False}))
    assert run.admission() is True
    path.write_text(json.dumps({"status": "not_admitted_clean_word_fit_gate_failed",
        "stages": [{"exit_code": 0}], "model_training_executed": True}))
    with pytest.raises(ValueError):
        run.admission()


def test_fixed_schedule_classification_and_independent_destinations():
    config = run.Config()
    assert run.rate(1) == config.max_lr / config.warmup
    assert run.rate(config.warmup) == config.max_lr
    assert run.rate(config.updates) == config.max_lr
    with pytest.raises(ValueError):
        run.rate(0)
    with pytest.raises(FrozenInstanceError):
        config.updates = 1
    assert audit.classify(dict(zip(run.ARMS, (False, True)))) == "zero_dropout_fit_passed_control_failed"
    assert audit.classify(dict(zip(run.ARMS, (True, True)))) == "both_continuations_fit_passed"
    assert audit.classify(dict(zip(run.ARMS, (True, False)))) == "original_dropout_only_fit_passed"
    assert audit.classify(dict(zip(run.ARMS, (False, False)))) == "both_dropout_fit_gates_failed"
    with pytest.raises(ValueError):
        audit.classify({run.ARMS[0]: False})
    locations = [run.root(arm, stage) for arm in run.ARMS for stage in ("pilot", "train", "fit")]
    assert len(set(locations)) == 6 and not set(locations) & {run.clean.TRAIN, run.forecaster.TRAIN}


def replay_fixture(monkeypatch):
    config = run.Config(updates=4, warmup=1, snapshot_interval=2)
    monkeypatch.setattr(run, "Config", lambda: config)
    row = {"id": "a", "source_frame": 256, "concept_ids": list(range(800)),
        "word_ids": [7, 50256] + [50257] * 190}
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    digest = run.clean.recovery.study.base.tensor_digest
    trace = []
    for step in range(1, 5):
        torch.randint(1, (1,), generator=sampler)
        trace.append({"step": step, "id": "a", "source_frame": 256, "lr": run.rate(step, config),
            "correct_history_sha256": digest({"codes": torch.tensor([row["concept_ids"][:608]])}),
            "target_word_sha256": digest({"words": torch.tensor([row["word_ids"]])}),
            "clean_word_CE": 1., "gradient_norm": 1.})
    state = {"step": 4, "arm": run.ARMS[1], "config": asdict(config),
        "initial_word_checkpoint_sha256": "initial", "forecaster_checkpoint_sha256": run.clean.INITIAL_FORECASTER,
        "dropout_maps": {"original": {"t": .3, "d": .1}, "applied": {"t": 0., "d": 0.}},
        "sampler_rng": sampler.get_state(), "optimizer": {
            "state": {0: {"step": torch.tensor(4.)}},
            "param_groups": [{"lr": config.max_lr, "betas": (config.beta1, config.beta2),
                "eps": config.optimizer_eps, "weight_decay": config.weight_decay}]}}
    return [row], trace, state


def test_raw_trace_replay_rejects_target_sampler_optimizer_and_dropout_changes(monkeypatch):
    rows, trace, state = replay_fixture(monkeypatch)
    assert audit.replay_trace(rows, trace, state, run.ARMS[1], "initial")["every_TRAIN_case_sampled"]
    for key in ("correct_history_sha256", "target_word_sha256", "id"):
        broken = deepcopy(trace)
        broken[1][key] = "bad"
        with pytest.raises(ValueError):
            audit.replay_trace(rows, broken, state, run.ARMS[1], "initial")
    for key in ("lr", "weight_decay", "eps"):
        broken = deepcopy(state)
        broken["optimizer"]["param_groups"][0][key] = 99.
        with pytest.raises(ValueError):
            audit.replay_trace(rows, trace, broken, run.ARMS[1], "initial")
    broken = deepcopy(state)
    broken["dropout_maps"]["applied"]["t"] = .3
    with pytest.raises(ValueError):
        audit.replay_trace(rows, trace, broken, run.ARMS[1], "initial")
    broken = deepcopy(state)
    broken["sampler_rng"] = torch.Generator().manual_seed(1).get_state()
    with pytest.raises(ValueError):
        audit.replay_trace(rows, trace, broken, run.ARMS[1], "initial")


def test_pilot_audit_requires_actual_maximum_frame_and_unchanged_sampler(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "root", lambda arm, stage: tmp_path)
    rows = [{"id": str(i), "source_ids": [1] * (i + 1), "source_frame": 256,
        "concept_ids": [0] * 800, "word_ids": [7, 50256] + [50257] * 190} for i in range(2)]
    digest = run.clean.recovery.study.base.tensor_digest
    trace = [{"step": i + 1, "id": row["id"], "source_frame": 256 if i == 0 else 3328,
        "lr": run.Config().max_lr, "clean_word_CE": 1., "gradient_norm": 1.,
        "correct_history_sha256": digest({"codes": torch.tensor([row["concept_ids"][:608]])}),
        "target_word_sha256": digest({"words": torch.tensor([row["word_ids"]])})}
        for i, row in enumerate(rows)]
    maps = {"original": {"t": .3, "d": .1}, "applied": {"t": 0., "d": 0.}}
    state = {"step": 2, "arm": run.ARMS[1], "config": asdict(run.Config()), "dropout_maps": maps,
        "initial_word_checkpoint_sha256": "initial", "forecaster_checkpoint_sha256": run.clean.INITIAL_FORECASTER,
        "producer_fingerprint": "producer", "optimizer": {"state": {0: {"step": torch.tensor(2.)}}},
        "sampler_rng": torch.Generator().manual_seed(run.Config().sampler_seed).get_state()}
    torch.save(state, tmp_path / "model.pt")
    (tmp_path / "result.json").write_text(json.dumps({"dropout_maps": maps, "producer_fingerprint": "producer",
        "checkpoint_sha256": run.file_hash(tmp_path / "model.pt")}))
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in trace))
    assert audit.replay_pilot(rows, run.ARMS[1], "initial")["dropout_maps"] == maps
    trace[1]["source_frame"] = 512
    path.write_text("".join(json.dumps(item) + "\n" for item in trace))
    with pytest.raises(ValueError, match="maximum-frame"):
        audit.replay_pilot(rows, run.ARMS[1], "initial")


def test_raw_fit_replay_includes_eos_excludes_pad_and_rejects_false_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "root", lambda arm, stage: tmp_path)
    selected, measured, histories = [], [], {}
    for i in range(32):
        row = {"id": str(i), "source_sha256": f"sha{i}", "target_text": "w7",
            "word_ids": [7, 50256] + [50257] * 190, "concept_ids": [0] * 800,
            "diagnostic_group": "original_16" if i < 16 else "additional_16"}
        selected.append(row)
        values = {"raw_word_ids": [7, 50256] + [8] * 190, "text": "w7",
            "global_EOS_position": 1, "correct": 2, "total": 2, "CE": .1, "exact_target_text": True}
        measured.append({"id": row["id"], "source_sha256": row["source_sha256"],
            "target_word_ids": row["word_ids"], "target_text": "w7", "diagnostic_group": row["diagnostic_group"],
            "actual_coarse_codes": [0] * 608,
            "arms": {name: deepcopy(values) for name in run.clean.diagnostic.ARMS},
            "own_gold_agreement": {"correct": 2, "total": 2},
            "concepts": [{"level": level, "active_correct": width // 4, "active_total": width // 4}
                         for level, width in enumerate(run.forecaster.WIDTHS)]})
        histories[row["id"]] = {"coarse_codes": [0] * 608}
    path = tmp_path / "case_measurements.json"
    path.write_text(json.dumps(measured))
    result = {"arm": run.ARMS[1], "components": run.clean.diagnostic.aggregate(measured),
        "clean_fit_gate": True, "measurements_sha256": run.file_hash(path)}
    (tmp_path / "result.json").write_text(json.dumps(result))
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " ".join(f"w{i}" for i in ids))
    assert audit.replay_fit(run.ARMS[1], selected, histories, tokenizer)["clean_fit_gate"]
    measured[0]["arms"]["correct_history"]["raw_word_ids"][1] = 7
    path.write_text(json.dumps(measured))
    with pytest.raises(ValueError, match="raw counts"):
        audit.replay_fit(run.ARMS[1], selected, histories, tokenizer)
