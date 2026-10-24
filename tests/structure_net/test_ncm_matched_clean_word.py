from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import ncm_matched_clean_word as run
from experiments.neural_architecture_lab import finalize_ncm_matched_clean_word as audit


def components(old_correct=199, new_correct=199):
    return {
        "original_16": {"cases": 16, "arms": {"correct_history": {"correct": old_correct, "total": 200}}},
        "additional_16": {"cases": 16, "arms": {"correct_history": {"correct": new_correct, "total": 200}}},
        "all_32": {"cases": 32, "arms": {"correct_history": {"correct": old_correct + new_correct, "total": 400}}},
    }


def test_fitting_gate_is_inclusive_and_each_group_must_pass():
    assert run.fitting_gate(components())
    assert not run.fitting_gate(components(198, 200))
    assert not run.fitting_gate(components(200, 198))


def test_fitting_gate_rejects_incomplete_or_inconsistent_counts():
    values = components()
    values["all_32"]["arms"]["correct_history"]["total"] = 399
    with pytest.raises(ValueError):
        run.fitting_gate(values)
    values = components()
    values["additional_16"]["cases"] = 15
    with pytest.raises(ValueError):
        run.fitting_gate(values)


def test_fixed_schedule_and_frozen_config():
    config = run.Config()
    assert run.rate(1, config) == config.max_lr / config.warmup
    assert run.rate(config.warmup, config) == config.max_lr
    assert run.rate(config.updates, config) == pytest.approx(config.min_lr)
    with pytest.raises(ValueError):
        run.rate(config.updates + 1, config)
    with pytest.raises(FrozenInstanceError):
        config.updates = 1
    assert run.recovery.study.q.digest(asdict(config)) != run.recovery.study.q.digest(
        asdict(replace(config, fit_numerator=198)))


def tiny_model():
    specification = importlib.util.spec_from_file_location(
        "matched_clean_test_model", Path("tests/structure_net/test_ncm_generated_history_core.py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.model()


def test_clean_objective_has_real_word_gradients_and_no_finest_label_leakage():
    forecaster, codec, features, mask, source_ids = tiny_model()
    branch = run.recovery.study.core.independent_word_branch(forecaster).eval()
    widths = branch.generator.levels
    codes = torch.randint(0, 16, (2, sum(widths)))
    batch = {"features": features, "mask": mask, "source_ids": source_ids,
             "codes": codes, "words": torch.randint(0, 30, (2, widths[-1]))}
    loss = run.clean_backward(branch, batch)
    assert loss > 0 and features.grad is None
    assert all(parameter.grad is None for parameter in forecaster.parameters())
    assert all(parameter.grad is None for parameter in codec.parameters())
    assert all(parameter.grad is None for parameter in branch.generator.lm_heads.parameters())
    assert branch.source_head.source_projection.weight.grad.abs().sum() > 0
    assert branch.word_projection.weight.grad.abs().sum() > 0
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in branch.word_decoder.parameters())
    gradient = branch.word_projection.weight.grad.detach().clone()
    branch.zero_grad(set_to_none=True)
    changed = codes.clone()
    changed[:, -widths[-1]:] = (changed[:, -widths[-1]:] + 1) % 16
    assert run.clean_backward(branch, {**batch, "codes": changed}) == loss
    assert torch.equal(branch.word_projection.weight.grad, gradient)


def replay_fixture(monkeypatch):
    config = run.Config(updates=4, warmup=1)
    monkeypatch.setattr(run, "Config", lambda: config)
    rows = [{"id": "a", "source_frame": 256, "concept_ids": list(range(800)),
             "word_ids": [7, 50256] + [50257] * 190}]
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    expected_history = run.recovery.study.base.tensor_digest({
        "codes": torch.tensor([rows[0]["concept_ids"][:608]], dtype=torch.long)})
    expected_words = run.recovery.study.base.tensor_digest({
        "words": torch.tensor([rows[0]["word_ids"]], dtype=torch.long)})
    trace = []
    for step in range(1, 5):
        torch.randint(1, (1,), generator=sampler)
        trace.append({"step": step, "id": "a", "source_frame": 256,
            "correct_history_sha256": expected_history, "lr": run.rate(step, config),
            "target_word_sha256": expected_words,
            "clean_word_CE": 1., "gradient_norm": 2.})
    state = {"step": 4, "config": asdict(config),
        "initial_word_checkpoint_sha256": run.INITIAL_WORD,
        "forecaster_checkpoint_sha256": run.INITIAL_FORECASTER,
        "sampler_rng": sampler.get_state(), "optimizer": {"state": {0: {"step": torch.tensor(4.)}}}}
    return rows, trace, state


def test_trace_replay_checks_sampler_optimizer_and_correct_histories(monkeypatch):
    rows, trace, state = replay_fixture(monkeypatch)
    assert audit.replay_training(rows, trace, state)["every_TRAIN_case_sampled"]
    broken = deepcopy(trace)
    broken[2]["correct_history_sha256"] = "wrong"
    with pytest.raises(ValueError):
        audit.replay_training(rows, broken, state)
    broken = deepcopy(trace)
    broken[1]["target_word_sha256"] = "wrong"
    with pytest.raises(ValueError):
        audit.replay_training(rows, broken, state)
    broken_state = deepcopy(state)
    broken_state["optimizer"]["state"][0]["step"] = torch.tensor(3.)
    with pytest.raises(ValueError):
        audit.replay_training(rows, trace, broken_state)


def test_artifact_roots_are_distinct_and_stage_is_not_semantic_success():
    roots = (run.PILOT, run.FIT_PILOT, run.TRAIN, run.FIT, run.WORKFLOW, run.RESULTS)
    assert len(set(roots)) == len(roots)
    assert all(root != run.recovery.study.TRAIN for root in roots)
    assert run.HYPOTHESIS != run.recovery.study.HYPOTHESIS


def test_fit_replay_rejects_a_raw_output_that_disagrees_with_reported_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "FIT", tmp_path)
    selected, measurements, histories = [], [], {}
    widths = (16, 24, 40, 56, 80, 104, 128, 160)
    for i in range(32):
        row = {"id": str(i), "source_sha256": f"sha{i}", "target_text": "w7",
            "word_ids": [7, 50256] + [50257] * 190, "concept_ids": [0] * 800,
            "diagnostic_group": "original_16" if i < 16 else "additional_16"}
        selected.append(row)
        arm = {"raw_word_ids": [7, 50256] + [7] * 190, "text": "w7",
            "global_EOS_position": 1, "correct": 2, "total": 2, "CE": .1,
            "exact_target_text": True}
        measurements.append({"id": row["id"], "source_sha256": row["source_sha256"],
            "target_text": "w7", "target_word_ids": row["word_ids"],
            "diagnostic_group": row["diagnostic_group"], "actual_coarse_codes": [0] * 608,
            "arms": {key: deepcopy(arm) for key in run.diagnostic.ARMS},
            "own_gold_agreement": {"correct": 2, "total": 2},
            "concepts": [{"level": level, "active_correct": width // 4, "active_total": width // 4}
                         for level, width in enumerate(widths)]})
        histories[row["id"]] = {"coarse_codes": [0] * 608}
    output = tmp_path / "case_measurements.json"
    output.write_text(json.dumps(measurements))
    result = {"components": run.diagnostic.aggregate(measurements), "clean_fit_gate": True,
              "measurements_sha256": audit.file_hash(output)}
    (tmp_path / "result.json").write_text(json.dumps(result))
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return " ".join(f"w{value}" for value in ids)
    assert audit.replay_fit(selected, histories, {}, Tokenizer())["clean_fit_gate"]
    measurements[0]["arms"]["actual_history"]["raw_word_ids"][0] = 8
    output.write_text(json.dumps(measurements))
    with pytest.raises(ValueError, match="raw word fitting"):
        audit.replay_fit(selected, histories, {}, Tokenizer())
