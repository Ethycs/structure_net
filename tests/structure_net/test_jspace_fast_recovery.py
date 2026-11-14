"""CPU protocol tests; the actual GPU validation is always reservation-controlled."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net import jspace_fast_recovery as study


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_pretest_recovery_screen_uses_choices_and_relative_endpoint_loss():
    result = {"baseline_development": {"loss": 1.}, "development": {"loss": 1.001},
              "development_comparison": {"agreement": 1.}}
    assert not study.recovery_required(result)
    result["development"]["loss"] = 1.00101
    assert study.recovery_required(result)
    result["development"]["loss"] = .9
    result["development_comparison"]["agreement"] = 63 / 64
    assert study.recovery_required(result)
    result["development"]["loss"] = float("nan")
    with pytest.raises(ValueError):
        study.recovery_required(result)


def test_fresh_split_excludes_both_previous_experiments_and_duplicates():
    parent = {"splits": {"train": [{"identity": "train"}], "evaluation": [{"identity": "parent-test"}],
                         "previous_evaluation": [{"identity": "old-test"}], "development": [{"identity": "dev"}]}}
    layer = {"splits": {"evaluation": [{"identity": "layer-test"}]}}
    excluded = study.merged_exclusions(parent, layer)
    fresh = [{"identity": f + str(i), "family": f} for f in study.base.DATASETS for i in range(128)]
    study.validate_fresh(fresh, excluded)
    for identity in ("train", "dev", "parent-test", "old-test", "layer-test", fresh[1]["identity"]):
        changed = [dict(x) for x in fresh]; changed[0]["identity"] = identity
        with pytest.raises(ValueError, match="unseen"):
            study.validate_fresh(changed, excluded)
    changed = [dict(x) for x in fresh]; changed[0]["family"] = list(study.base.DATASETS)[1]
    with pytest.raises(ValueError, match="balance"):
        study.validate_fresh(changed, excluded)


def test_selection_binds_candidate_benchmark_and_unseen_test(tmp_path):
    benchmark = tmp_path / "result.json"; benchmark.write_text("frozen benchmark")
    selection = {"source_arm": "r1536_h64_gb", "candidate": "combined", "validation_mode": "direct",
                 "bench_result_sha256": study.sha(benchmark), "heldout_inspected": False}
    study.validate_selection(selection, "r1536_h64_gb", "combined", benchmark, False)
    for key, value in (("source_arm", "other"), ("candidate", "other"), ("validation_mode", "recovery"),
                       ("heldout_inspected", True), ("bench_result_sha256", "changed")):
        with pytest.raises(ValueError, match="pre-test selection"):
            study.validate_selection({**selection, key: value}, "r1536_h64_gb", "combined", benchmark, False)


@pytest.mark.parametrize("recover", [False, True])
def test_export_hash_must_match_before_selected_checkpoint_loading(tmp_path, recover):
    path = tmp_path / "fast.pt"; path.write_bytes(b"actual exported tensors")
    report = {"checkpoint_sha256": study.sha(path)}
    dump(tmp_path / ("training_fast.json" if recover else "deployment.json"), report if recover else {"checkpoints": {"fast": report}})
    assert study.selected_checkpoint(tmp_path, "fast", recover) == path
    path.write_bytes(b"changed export")
    with pytest.raises(ValueError, match="Changed selected"):
        study.selected_checkpoint(tmp_path, "fast", recover)


def test_exact_tensor_identity_includes_dtype_and_all_buffers():
    state = {"weight": torch.ones(2), "basis": torch.eye(2)}
    assert study.equal_state(state, {k: v.clone() for k, v in state.items()})
    assert not study.equal_state(state, {**state, "extra": torch.ones(1)})
    assert not study.equal_state(state, {**state, "basis": state["basis"].bfloat16()})
    assert not study.equal_state(state, {**state, "weight": torch.zeros(2)})


def test_direct_decision_prevents_unrequested_training(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "verify", lambda *a: {"recover": False})
    monkeypatch.setattr(study, "Stage", lambda *a: pytest.fail("Direct validation started training"))
    with pytest.raises(ValueError, match="frozen recovery decision"):
        study.train(tmp_path, "fast")


def test_recovery_evaluation_requires_both_completed_arms(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "verify", lambda *a: {"recover": True})
    names = []
    def checked(root, name):
        names.append(name)
        if name == "train_fast":
            raise ValueError("Incomplete")
    monkeypatch.setattr(study, "checked", checked)
    monkeypatch.setattr(study, "Stage", lambda *a: pytest.fail("Evaluation exposed test before both arms finished"))
    with pytest.raises(ValueError, match="Incomplete"):
        study.evaluate(tmp_path)
    assert names == ["train_control", "train_fast"]


def test_reload_consumes_exported_checkpoint_even_in_direct_mode(tmp_path, monkeypatch):
    calls = []
    rows = [{"identity": "heldout", "prediction": 0, "answer": 0, "family": "arc", "logits": [1., 0.]}]
    class Stage:
        def __init__(self, root, name):
            self.root, self.path = root, root / name; self.path.mkdir()
            self.plan, self.tasks = {"recover": False}, {"evaluation": rows}
        def install(self, arm, **kwargs):
            calls.append((arm, kwargs))
        def finish(self):
            return {"complete": True}
    monkeypatch.setattr(study, "Stage", Stage)
    monkeypatch.setattr(study, "checked", lambda *a: None)
    monkeypatch.setattr(study, "run_cases", lambda *a: rows)
    for arm in study.ARMS:
        dump(tmp_path / "evaluation" / (arm + "_fresh.json"), rows)
    study.reload(tmp_path)
    assert calls == [(arm, {"packed": True, "selected": True}) for arm in study.ARMS]
    value = json.loads((tmp_path / "reload/verification.json").read_text())
    assert all(x["records_exact"] for x in value.values())


def test_matched_continuation_uses_all_four_epochs_and_common_order(tmp_path, monkeypatch):
    seen = {}
    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.weight = nn.Parameter(torch.tensor(.5))
        def pack(self, dtype=None, buffer_dtype=None):
            self.to(dtype=dtype).requires_grad_(False); return self
    class Stage:
        def __init__(self, root, name):
            self.root, self.path, self.name = root, root / name, name; self.path.mkdir()
            self.items = [{"index": i} for i in range(32)]; self.training = list(range(32))
            self.model = nn.Module(); self.model.model = nn.Module()
            self.model.model.layers = nn.ModuleList([Block()]); self.model.model.norm = nn.Linear(1, 1, bias=False)
            with torch.no_grad(): self.model.model.norm.weight.fill_(1)
            seen[name] = []
        def install(self, arm, trainable=False):
            assert trainable; return self.model.model.layers[-1]
        def event(self, *a, **k): pass
        def finish(self, *a): return {"complete": True}
    def logits(stage, case):
        seen[stage.name].append(case["index"])
        return stage.model.model.layers[-1].weight + .1 * stage.model.model.norm.weight.squeeze()
    def objective(logits, teacher, case):
        loss = (logits - 1).square(); return loss, loss, loss
    def development(stage, tag):
        rows = [{"weight": float(stage.model.model.layers[-1].weight.detach())}]
        dump(stage.path / (tag + "_development.json"), rows)
        return {"loss": 1.}, rows
    monkeypatch.setattr(study, "Stage", Stage)
    monkeypatch.setattr(study, "verify", lambda *a: {"recover": True, "source_arm": "r1536_h64_gb", "initializer_sha256": "same"})
    monkeypatch.setattr(study.torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(study.base, "cached_logits", logits); monkeypatch.setattr(study.base, "objective", objective)
    monkeypatch.setattr(study.base, "packed_development", development); monkeypatch.setattr(study.base, "development", development)
    monkeypatch.setattr(study, "read_bits", lambda *a: None)
    Stage.targets = None
    for arm in study.ARMS:
        study.train(tmp_path, arm)
        report = json.loads((tmp_path / ("training_" + arm + ".json")).read_text())
        assert report["completed_epochs"] == 4 and report["updates"] == 32
        assert report["selected_epoch"] == 0 and report["initializer_sha256"] == "same"
        assert report["checkpoint_sha256"] == study.sha(tmp_path / (arm + ".pt"))
    assert len(seen["train_control"]) == 128 and seen["train_control"] == seen["train_fast"]
