"""CPU lifecycle checks for matched boundary-distillation experiments."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net import jspace_qwen_layerkd as study


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


def test_calibration_is_balanced_and_training_only():
    items = []
    for family in study.DATASETS:
        items += [{"split": "development", "family": family}] * 3
        items += [{"split": "train", "family": family}] * 20
        items += [{"split": "evaluation", "family": family}] * 2
    selected = study.calibration_indices(items)
    assert len(selected) == 32 and len(set(selected)) == 32
    assert all(items[i]["split"] == "train" for i in selected)
    assert all(sum(items[i]["family"] == f for i in selected) == 8 for f in study.DATASETS)


def test_order_is_complete_repeatable_and_arm_independent():
    original = list(range(512))
    for epoch in (1, 6, 12):
        orders = [study.training_order(original, epoch) for _ in study.ARMS]
        assert all(order == orders[0] for order in orders)
        assert sorted(orders[0]) == original
    assert original == list(range(512))
    assert study.training_order(original, 1) != study.training_order(original, 2)


def test_development_cannot_read_auxiliary_training_targets(monkeypatch):
    stage = SimpleNamespace(
        items=[{"split": "development", "loss_positions": [0]}],
        x=lambda case: None, targets=None,
        model=SimpleNamespace(model=SimpleNamespace(norm=nn.Identity()),
                              lm_head=SimpleNamespace(weight=torch.eye(2))),
        teacher_boundary=lambda case: pytest.fail("Development read auxiliary teacher boundary"),
        metric=lambda: pytest.fail("Development read auxiliary output metric"),
    )
    monkeypatch.setattr(study.torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(study.base, "forward_block", lambda *a: torch.zeros(1, 1, 2))
    monkeypatch.setattr(study, "read_bits", lambda *a: None)
    monkeypatch.setattr(study.base, "objective", lambda *a: (0., 0., 0.))
    with pytest.raises(ValueError, match="only consume training"):
        study.losses(stage, 0, auxiliary="jacobian")
    assert study.losses(stage, 0, auxiliary=False) == (0., 0., 0., {})


def test_auxiliary_display_normalization_preserves_raw_objective_and_gradient():
    x = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    endpoint, local = (x - 1).square(), (2 * x + .3).square()
    calibration = {"coefficients": {arm: {"raw_coefficient": .17, "normalizer": 3.7,
                    "normalized_coefficient": .17 * 3.7} for arm in study.ARMS[1:]}}
    for arm in study.ARMS[1:]:
        actual = study.total_loss(endpoint, {arm: local}, arm, calibration)
        expected = endpoint + .17 * local
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        ga = torch.autograd.grad(actual, x, retain_graph=True)[0]
        ge = torch.autograd.grad(expected, x, retain_graph=True)[0]
        torch.testing.assert_close(ga, ge, rtol=1e-14, atol=1e-14)
    assert study.total_loss(endpoint, {}, "endpoint", None) is endpoint


def test_evaluation_requires_all_three_completed_arms_before_stage(monkeypatch, tmp_path):
    calls = []

    def checked(root, name):
        calls.append(name)
        if name == "train_jacobian":
            raise ValueError("Incomplete stage")

    monkeypatch.setattr(study, "checked", checked)
    monkeypatch.setattr(study, "Stage", lambda *a: pytest.fail("Evaluation started before all arms completed"))
    with pytest.raises(ValueError, match="Incomplete stage"):
        study.evaluate(tmp_path)
    assert calls == ["train_" + arm for arm in study.ARMS]


def test_systems_root_cannot_train(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "verify", lambda *a, **k: {"systems_only": True})
    monkeypatch.setattr(study, "Stage", lambda *a: pytest.fail("Systems root constructed a training stage"))
    with pytest.raises(ValueError, match="Systems roots"):
        study.train(tmp_path, "endpoint")


def test_checkpoint_tampering_is_rejected_before_model_install(tmp_path):
    path = tmp_path / "endpoint.pt"
    path.write_bytes(b"original checkpoint")
    dump(tmp_path / "training_endpoint.json", {"checkpoint_sha256": study.sha(path)})
    path.write_bytes(b"changed checkpoint")
    stage = SimpleNamespace(root=tmp_path, install=lambda **kw: pytest.fail("Installed an unverified checkpoint"))
    with pytest.raises(ValueError, match="Changed selected checkpoint"):
        study.load_selected(stage, "endpoint")


def fixture_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    monkeypatch.setattr(study, "PARENT", parent)
    root = tmp_path / "run"
    (root / "producer").mkdir(parents=True)
    source = tmp_path / "producer_source.py"
    source.write_text("frozen_source = True\n")
    (root / "producer" / source.name).write_bytes(source.read_bytes())
    inputs = {}
    for name in ("fit.pt", "relation.pt"):
        (parent / name).write_bytes(name.encode())
        (root / name).write_bytes((parent / name).read_bytes())
        inputs[str(parent / name)] = study.sha(parent / name)
    tasks = {"splits": {"train": [], "development": [], "evaluation": []}}
    dump(root / "tasks.json", tasks)
    plan = {"study_id": study.STUDY_ID, "config": study.CONFIG, "systems_only": False,
            "source_sha256": {str(source): study.sha(source)}, "input_sha256": inputs,
            "task_digest": study.base.digest(tasks)}
    plan["fingerprint"] = study.base.digest(plan)
    dump(root / "plan.json", plan)
    return root, parent, source, plan


@pytest.mark.parametrize("changed", ["source", "snapshot", "tasks", "fit", "relation", "parent"])
def test_provenance_rejects_changed_sources_tasks_or_compression(tmp_path, monkeypatch, changed):
    root, parent, source, plan = fixture_plan(tmp_path, monkeypatch)
    assert study.verify(root, inputs=True)["fingerprint"] == plan["fingerprint"]
    targets = {"source": source, "snapshot": root / "producer" / source.name,
               "tasks": root / "tasks.json", "fit": root / "fit.pt",
               "relation": root / "relation.pt", "parent": parent / "fit.pt"}
    if changed == "tasks":
        dump(targets[changed], {"splits": {"evaluation": ["changed"]}})
    else:
        targets[changed].write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed"):
        study.verify(root, inputs=True)


def test_stage_manifest_hashes_are_enforced(tmp_path, monkeypatch):
    root, _, _, plan = fixture_plan(tmp_path, monkeypatch)
    artifact = root / "setup" / "bank_metadata.json"
    dump(artifact, {"training_only": True})
    dump(root / "setup_manifest.json", {"status": "completed", "fingerprint": plan["fingerprint"],
         "artifact_sha256": {"setup/bank_metadata.json": study.sha(artifact)}})
    assert study.checked(root, "setup")["status"] == "completed"
    dump(artifact, {"training_only": False})
    with pytest.raises(ValueError, match="Changed stage artifact"):
        study.checked(root, "setup")


def test_all_arms_complete_same_fixed_epochs_and_order_when_development_plateaus(tmp_path, monkeypatch):
    """Exercise the actual training loop with a tiny CPU model and constant dev loss."""
    seen, stages = {}, {}

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(.5))

        def pack(self, dtype=None, buffer_dtype=None):
            self.to(dtype=dtype).requires_grad_(False)
            return self

    class FakeStage:
        def __init__(self, root, name):
            self.root, self.name, self.path = root, name, root / name
            self.path.mkdir()
            self.items = [{"identity": f"{f}-{i}", "family": f, "split": "train"}
                          for f in study.DATASETS for i in range(8)]
            self.training = list(range(len(self.items)))
            self.model = nn.Module()
            self.model.model = nn.Module()
            self.model.model.layers = nn.ModuleList([Block()])
            self.model.model.norm = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.model.model.norm.weight.fill_(1)
            seen[name] = []
            stages[name] = self

        def install(self, trainable=False, initialized=False):
            assert trainable and initialized
            return self.model.model.layers[-1]

        def event(self, *args, **kwargs):
            pass

        def finish(self, names):
            return {"completed": True}

    def losses(stage, index, auxiliary=True):
        seen[stage.name].append(index)
        w = stage.model.model.layers[-1].weight
        norm = stage.model.model.norm.weight.squeeze()
        endpoint = (w + .1 * norm - 1).square()
        aux = {"euclidean": (w - .2).square(), "jacobian": (2 * w - .4).square()} if auxiliary else {}
        return endpoint, endpoint * .2, endpoint * .3, aux

    def development(stage, tag):
        row = {"identity": "heldout", "logits": [float(stage.model.model.layers[-1].weight.detach())]}
        dump(stage.path / (tag + "_development.json"), [row])
        return {"loss": 1., "accuracy": {"correct": 1, "count": 1, "overall": 1.}}, [row]

    monkeypatch.setattr(study, "Stage", FakeStage)
    monkeypatch.setattr(study, "verify", lambda *a, **k: {"systems_only": False})
    monkeypatch.setattr(study, "checked", lambda *a: None)
    monkeypatch.setattr(study, "losses", losses)
    monkeypatch.setattr(study.base, "packed_development", development)
    monkeypatch.setattr(study.base, "development", development)
    dump(tmp_path / "calibration.json", {"initialized_checkpoint_sha256": "shared_initializer",
         "coefficients": {arm: {"raw_coefficient": .1, "normalizer": 2., "normalized_coefficient": .2}
                          for arm in study.ARMS[1:]}})
    assert study.CONFIG["epochs"] == 12 and study.CONFIG["early_stopping"] is False
    for arm in study.ARMS:
        study.train(tmp_path, arm)
        result = json.loads((tmp_path / f"training_{arm}.json").read_text())
        assert result["completed_epochs"] == 12
        assert result["selected_epoch"] == 0  # No arm may stop when development plateaus.
        assert result["updates"] == 12 * 32 // study.CONFIG["accumulation"]
        assert result["initialized_checkpoint_sha256"] == "shared_initializer"
        assert result["checkpoint_sha256"] == study.sha(tmp_path / f"{arm}.pt")
        assert len(seen["train_" + arm]) == 12 * 32
    assert seen["train_endpoint"] == seen["train_euclidean"] == seen["train_jacobian"]
