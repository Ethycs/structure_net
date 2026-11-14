"""Regression for norm tensors restored by inference-mode deployment validation."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net import jspace_fast_recovery_v2 as study


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    parent, performance = tmp_path / "parent", tmp_path / "performance"
    parent.mkdir(); (performance / "bench").mkdir(parents=True)
    native = nn.Linear(2, 2, bias=False).bfloat16().eval().requires_grad_(False)
    norm = nn.Linear(2, 2, bias=False).bfloat16().eval().requires_grad_(False)
    torch.save({"block": native.state_dict(), "norm": norm.state_dict()}, parent / "r1536_h64_gb.pt")
    rows = [{"identity": "heldout", "family": "arc", "answer": 0, "prediction": 0, "logits": [1., 0.]}]
    for tag in ("r1536_h64_gb_baseline", "r1536_h64_gb_combined"):
        (performance / "bench" / (tag + "_development.json")).write_text(json.dumps(rows))
    plan = {"recover": False, "source_arm": "r1536_h64_gb", "candidate": "combined",
            "fast_options": study.perf.CANDIDATES["combined"], "performance_root": str(performance),
            "parent_root": str(parent), "initializer_sha256": "shared", "fingerprint": "test"}
    resets, stages, scoring = [], [], []

    class CPUStage:
        def __init__(self, root, name):
            self.root, self.path, self.name, self.parent = root, root / name, name, parent
            self.path.mkdir(); self.plan = plan; self.tasks = {"evaluation": rows}
            self.native = copy.deepcopy(native)
            self.native_norm = {k: v.clone() for k, v in norm.state_dict().items()}
            self.model = nn.Module(); self.model.model = nn.Module()
            self.model.model.layers = nn.ModuleList([copy.deepcopy(native)])
            self.model.model.norm = copy.deepcopy(norm)
            stages.append(self)

        def install(self, arm, packed=False, selected=False):
            resets.append((self.name, arm, torch.is_inference_mode_enabled(), self.model.model.norm.weight.is_inference(), selected))
            # Exercise the real lifecycle operation that failed in v1.
            study.base.Stage.reset(self)
            if selected:
                path = study.selected_checkpoint(self.root, arm, False)
                state = torch.load(path, weights_only=True)
                self.model.model.layers[-1].load_state_dict(state["block"], strict=True)
                self.model.model.norm.load_state_dict(state["norm"], strict=True)
            return self.model.model.layers[-1]

        def event(self, *a, **k):
            pass

        def finish(self, *a):
            return {"status": "completed"}

    def score(stage, cases, tag):
        scoring.append((stage.name, tag, torch.is_inference_mode_enabled()))
        (stage.path / (tag + ".json")).write_text(json.dumps(rows))
        return rows

    monkeypatch.setattr(nn.Module, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(study.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(study, "Stage", CPUStage)
    monkeypatch.setattr(study, "verify", lambda *a, **k: plan)
    monkeypatch.setattr(study, "checked", lambda *a: {})
    monkeypatch.setattr(study, "validate_fast", lambda stage: None)  # Kernel tests are separate.
    monkeypatch.setattr(study.perf, "candidate_factory", lambda block, name: copy.deepcopy(block))
    monkeypatch.setattr(study.base, "development", lambda *a: ({"loss": 1.}, rows))
    monkeypatch.setattr(study.base, "runtime", lambda stage, arm: {"inference_mode": torch.is_inference_mode_enabled()})
    monkeypatch.setattr(study, "run_cases", score)
    return SimpleNamespace(root=tmp_path, stage=CPUStage, resets=resets, stages=stages, scoring=scoring)


def test_unprotected_restored_norm_reproduces_v1_failure(lifecycle):
    stage = lifecycle.stage(lifecycle.root, "unprotected")
    assert not torch.is_inference_mode_enabled()
    study.validate_direct_tensors(stage, "control")
    assert stage.model.model.norm.weight.is_inference()
    with pytest.raises(RuntimeError, match="inference tensor outside InferenceMode"):
        stage.install("native")


def test_actual_evaluate_export_reload_and_runtime_keep_reset_inside_inference(lifecycle):
    root = lifecycle.root
    study.evaluate(root)
    assert (root / "control.pt").exists() and (root / "fast.pt").exists()
    deployment = json.loads((root / "deployment.json").read_text())
    assert all(x["timing_tensors_match_parent_candidate"] for x in deployment["checkpoints"].values())
    study.reload(root)
    study.runtime(root)
    assert all(enabled for _, _, enabled, _, _ in lifecycle.resets)
    assert any(inference_tensor for _, _, _, inference_tensor, _ in lifecycle.resets)
    assert all(enabled for _, _, enabled in lifecycle.scoring)
    reloads = [(arm, selected) for name, arm, _, _, selected in lifecycle.resets if name == "reload"]
    assert reloads == [("control", True), ("fast", True)]
    verification = json.loads((root / "reload/verification.json").read_text())
    assert all(x["records_exact"] for x in verification.values())
    timing = json.loads((root / "runtime/result.json").read_text())
    assert all(x["inference_mode"] for x in timing.values())


def test_campaign_launches_only_v2_producer(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(study, "verify", lambda *a: {"recover": False})
    monkeypatch.setattr(study, "wait_ready", lambda *a: {})
    monkeypatch.setattr(study.subprocess, "run", lambda args, **kwargs: calls.append(args))
    study.campaign(tmp_path)
    assert [args[3] for args in calls] == ["evaluate", "reload"]
    assert all(args[2] == "experiments.structure_net.jspace_fast_recovery_v2" for args in calls)
    assert study.DEFAULT_ROOT.name == "20261113_v2"
