from copy import deepcopy
from dataclasses import asdict, replace, FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import ncm_dense_word_fit as run
from experiments.neural_architecture_lab import finalize_ncm_dense_word_fit as audit


def test_config_and_schedule_are_fixed_and_reach_declared_endpoints():
    c = run.Config()
    assert run.rate(1) == c.max_lr / c.warmup
    assert run.rate(c.warmup) == c.max_lr
    assert run.rate(c.updates) == c.min_lr
    assert run.rate(c.warmup + 1) < c.max_lr
    with pytest.raises(FrozenInstanceError):
        c.updates = 2
    for step in (0, c.updates + 1):
        with pytest.raises(ValueError, match="outside"):
            run.rate(step)


def test_admission_requires_both_valid_dropout_fitting_misses():
    for original, zero, classification in ((False, False, "both_dropout_fit_gates_failed"),
            (False, True, "zero_dropout_fit_passed_control_failed"),
            (True, False, "original_dropout_only_fit_passed"),
            (True, True, "both_continuations_fit_passed")):
        assert run.admission_result({"fit_gates": dict(zip(run.dropout.ARMS, (original, zero))),
            "classification": classification}) is (not original and not zero)
    with pytest.raises(ValueError, match="disagree"):
        run.admission_result({"fit_gates": dict(zip(run.dropout.ARMS, (False, False))), "classification": "both_continuations_fit_passed"})
    with pytest.raises(ValueError, match="both fixed"):
        run.admission_result({"fit_gates": dict(zip(run.dropout.ARMS, (0, False))), "classification": "both_dropout_fit_gates_failed"})


def components(first=199, second=199):
    return {name: {"cases": count, "correct": correct, "total": total}
        for name, count, correct, total in (("original_16", 16, first, 200),
            ("additional_16", 16, second, 200), ("all_32", 32, first + second, 400))}


def test_actual_history_gate_is_inclusive_and_applies_to_each_group():
    assert run.fitting_gate(components())
    assert not run.fitting_gate(components(198, 200))
    broken = components()
    broken["all_32"]["total"] -= 1
    with pytest.raises(ValueError, match="compose"):
        run.fitting_gate(broken)


def trace_fixture(monkeypatch, pilot=False):
    config = replace(run.Config(), updates=8, warmup=2, snapshot_interval=4)
    real_rate = run.rate
    monkeypatch.setattr(run, "Config", lambda: config)
    monkeypatch.setattr(run, "rate", lambda step: real_rate(step, config))
    rows = [{"id": str(i), "source_frame": 512 + 256 * i, "source_ids": [3] * (20 + i),
        "word_ids": [i, 50256] + [50257] * 190} for i in range(2)]
    histories = {r["id"]: {"coarse_codes": [i] * 608} for i, r in enumerate(rows)}
    labels = {r["id"]: {"refined": torch.ones(1, 192, 384) * i} for i, r in enumerate(rows)}
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    steps = 2 if pilot else config.updates
    trace = []
    digest = run.clean.recovery.study.base.tensor_digest
    for step in range(1, steps + 1):
        row = rows[step - 1] if pilot else rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        trace.append({"step": step, "id": row["id"],
            "source_frame": config.max_source_frame if pilot and step == 2 else row["source_frame"],
            "actual_history_sha256": digest({"codes": torch.tensor([histories[row["id"]]["coarse_codes"]])}),
            "target_word_sha256": digest({"words": torch.tensor([row["word_ids"]])}),
            "teacher_latent_sha256": digest({"latent": labels[row["id"]]["refined"]}),
            "lr": config.max_lr if pilot else real_rate(step, config), "latent_MSE": .1, "word_CE": .2, "gradient_norm": 1.})
    name = "source_head.source_projection.weight"
    state = {"config": asdict(config), "step": steps, "sampler_rng": sampler.get_state(),
        "parameter_names": [name], "trainable_parameter_names": [name], "model": {name: torch.ones(1)},
        "optimizer": {"param_groups": [{"params": [0], "betas": (config.beta1, config.beta2),
            "eps": config.optimizer_eps, "weight_decay": config.weight_decay,
            "lr": config.max_lr if pilot else real_rate(config.updates, config)}],
            "state": {0: {"step": torch.tensor(float(steps)), "exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1)}}}}
    return rows, histories, labels, trace, state


def test_trace_replay_binds_actual_history_dense_labels_and_sampler(monkeypatch):
    rows, histories, labels, trace, state = trace_fixture(monkeypatch)
    result = audit.replay_trace(rows, histories, labels, trace, state)
    assert result["updates"] == 8 and len(result["sample_counts"]) == 2
    changed = deepcopy(trace)
    changed[0]["teacher_latent_sha256"] = "changed"
    with pytest.raises(ValueError, match="actual source"):
        audit.replay_trace(rows, histories, labels, changed, state)
    state["optimizer"]["state"][0]["step"] -= 1
    with pytest.raises(ValueError, match="optimizer step"):
        audit.replay_trace(rows, histories, labels, trace, state)


def test_maximum_frame_maximum_rate_pilot_replay(monkeypatch):
    rows, histories, labels, trace, state = trace_fixture(monkeypatch, True)
    assert audit.replay_trace(rows, histories, labels, trace, state, pilot=True)["updates"] == 2
    trace[1]["source_frame"] = rows[1]["source_frame"]
    with pytest.raises(ValueError, match="actual source"):
        audit.replay_trace(rows, histories, labels, trace, state, pilot=True)


def test_raw_source_only_fit_preserves_complete_frames_counts_and_eos():
    rows, values, histories = [], [], {}
    for i in range(32):
        row = {"id": str(i), "source_sha256": "source" + str(i),
            "diagnostic_group": "original_16" if i < 16 else "additional_16",
            "word_ids": [7, 50256] + [50257] * 190, "target_text": "w7"}
        rows.append(row)
        histories[str(i)] = {"coarse_codes": [0] * 608}
        values.append({"id": row["id"], "source_sha256": row["source_sha256"],
            "diagnostic_group": row["diagnostic_group"], "target_word_ids": row["word_ids"],
            "target_text": "w7", "raw_word_ids": [7, 50256] + [8] * 190,
            "actual_coarse_codes": [0] * 608, "latent_sha256": "a" * 64,
            "correct": 2, "total": 2, "text": "w7", "global_EOS_position": 1, "exact_target_text": True})
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " ".join("w" + str(x) for x in ids))
    components = audit.replay_fit(rows, histories, values, tokenizer)
    assert components["all_32"]["correct"] == 64
    assert run.fitting_gate(components)
    values[0]["raw_word_ids"][0] = 9
    with pytest.raises(ValueError, match="raw full decoding"):
        audit.replay_fit(rows, histories, values, tokenizer)
