"""Amendment provenance, aggregate quality policy, and isolated orchestration."""
import json
from pathlib import Path

import pytest

from experiments.structure_net import jspace_allblocks_v2 as v2


def test_scoped_protocol_restores_bindings_and_reservation_module_after_failure():
    old = {name: getattr(v2.base, name) for name in ("CONFIG", "SCHEMA", "MODULE", "sources")}
    old_gate = v2.evaluator.acceptance_gates
    with pytest.raises(RuntimeError):
        with v2.protocol_context():
            command = v2.base.leased_command(Path("/tmp/never-launched"), "relation", index=0, layer=30)
            assert command[command.index("-m") + 1] == v2.MODULE
            assert command[:4] == ["/usr/bin/python3", "/usr/local/bin/inference-ctl", "lease", "run"]
            assert v2.base.CONFIG["maximum_accuracy_drop"] == .05
            assert v2.evaluator.acceptance_gates is v2.acceptance_gates
            with v2.protocol_context():
                assert v2.base.MODULE == v2.MODULE
            raise RuntimeError("stage failure")
    for name, value in old.items():
        assert getattr(v2.base, name) is value
    assert v2.evaluator.acceptance_gates is old_gate


def test_user_aggregate_allowance_has_no_hidden_family_veto():
    quality = {"overall": {"accuracy_drop": .04}, "by_family": {"one": {"accuracy_drop": .16}}}
    values = (quality, 100, 60, 10, 14, 10)
    result = v2.acceptance_gates(*values, reload_exact=True, graph_static_exact=True, config=v2.CONFIG)
    assert result["passed"]
    assert result["diagnostics"] == {"every_family_quality": False, "original_one_percentage_point_quality": False}
    assert "every_family_quality" not in result["gates"]
    assert result["thresholds"]["maximum_accuracy_drop"] == .05
    assert result["thresholds"]["family_quality_is_gate"] is False
    old = v2._V1_ACCEPTANCE(*values, reload_exact=True, graph_static_exact=True, config=v2._V1_CONFIG)
    assert not old["passed"] and not old["gates"]["overall_quality"]
    quality["overall"]["accuracy_drop"] = .051
    assert not v2.acceptance_gates(*values, reload_exact=True, graph_static_exact=True, config=v2.CONFIG)["passed"]
    quality["overall"]["accuracy_drop"] = 0
    assert not v2.acceptance_gates(quality, 100, 70, 10, 14, 10, reload_exact=True,
                                    graph_static_exact=True, config=v2.CONFIG)["passed"]
    assert not v2.acceptance_gates(quality, 100, 60, 10, 13, 10, reload_exact=True,
                                    graph_static_exact=True, config=v2.CONFIG)["passed"]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def previous(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = repo / "v1"
    root.mkdir(parents=True)
    monkeypatch.setattr(v2, "ROOT", repo)
    monkeypatch.setattr(v2.base, "ROOT", repo)
    source = repo / "source.py"
    source.write_text("# frozen source\n")
    archived = root / "producer/source.py"
    archived.parent.mkdir()
    archived.write_bytes(source.read_bytes())
    input_file = repo / "training-input.dat"
    input_file.write_bytes(b"original data")
    model = repo / "model"
    model.mkdir()
    (model / "weight").write_bytes(b"native weights")
    tasks = {"splits": {"train": [{"identity": "t"}], "development": [{"identity": "d"}],
                        "evaluation": [{"identity": "e"}]}}
    _write(root / "tasks.json", tasks)
    plan = {"schema": v2._V1_SCHEMA, "study_id": v2.base.STUDY_ID, "config": v2._V1_CONFIG,
        "systems_only": False, "source_sha256": {str(source): v2.base.sha(source)},
        "input_sha256": {str(input_file): v2.base.sha(input_file)},
        "model": {"path": str(model), "files": {"weight": v2.base.sha(model / "weight")}},
        "task_digest": v2.base.digest(tasks), "label_ids": [0, 1], "groups": [v2.base.group_layers(i) for i in range(6)],
        "original_cache": str(repo / "original-cache"), "budget": {"free_disk_bytes": 1000}}
    plan["fingerprint"] = v2.base.digest(plan)
    _write(root / "plan.json", plan)
    capture, basis = {}, {}
    for name in ("metadata.json", "items.json", "accepted_teacher_logits.npy"):
        name = "groups/group_00/capture/" + name
        _write(root / name, {})
        capture[name] = v2.base.sha(root / name)
    for layer in range(30, 36):
        for filename in ("mlp_calibration.npy", "boundary.npy", "mlp_jacobian_bf16.npy", "statistics.pt", "mlp_development.npy"):
            name = f"groups/group_00/capture/block_{layer:02d}/{filename}"
            _write(root / name, {})
            capture[name] = v2.base.sha(root / name)
        for filename in ("compiled_dev.npy", "output_weight.pt", "fit.json", "interface.json", "fit.pt"):
            name = f"groups/group_00/block_{layer:02d}/{filename}"
            _write(root / name, {"input_sha256": {str(input_file): v2.base.sha(input_file)}})
            basis[name] = v2.base.sha(root / name)
        if layer in v2.REUSED_LAYERS:
            artifacts = {}
            for filename in ("relation.pt", "relations.json"):
                name = f"groups/group_00/block_{layer:02d}/{filename}"
                _write(root / name, {"input_sha256": {str(input_file): v2.base.sha(input_file)}})
                artifacts[name] = v2.base.sha(root / name)
            _write(root / f"groups/group_00/block_{layer:02d}/fit_result/manifest.json",
                   {"status": "completed", "fingerprint": plan["fingerprint"], "artifacts": artifacts})
    for phase, artifacts in (("capture", capture), ("basis", basis)):
        _write(root / f"groups/group_00/{phase}/manifest.json",
               {"status": "completed", "fingerprint": plan["fingerprint"], "artifacts": artifacts})
    log = root / "logs/relation_g0_l30.log"
    log.parent.mkdir()
    log.write_text("ValueError: Insufficient independent task-Jacobian directions for the anchor bank\n")
    amendment = repo / "amendment.md"
    amendment.write_text("User authorized aggregate 5pp allowance; diagnosed rank rule only.\n")
    monkeypatch.setattr(v2, "AMENDMENT", amendment)
    monkeypatch.setattr(v2, "sources", lambda: [source, amendment])
    diagnostic = repo / "diagnostic"
    (diagnostic / "producer").mkdir(parents=True)
    (diagnostic / "producer/source.py").write_bytes(source.read_bytes())
    (diagnostic / "spectral_state.pt").write_bytes(b"diagnosed spectrum")
    interface = root / "groups/group_00/block_30/interface.json"
    dplan = {"layer": 30, "interface": str(interface), "teacher_fingerprint": v2.base.digest(plan["model"]),
             "settings": dict(svd_rank=64, oversample=16, power_iterations=1, chunk_rows=256,
                              anchors=24, old_rows=8192, seed=v2.CONFIG["seed"] + 30),
             "source_sha256": plan["source_sha256"], "input_sha256": {str(interface): v2.base.sha(interface)}}
    dplan["fingerprint"] = v2.base.digest(dplan)
    _write(diagnostic / "plan.json", dplan)
    _write(diagnostic / "result.json", {"status": "completed", "fingerprint": dplan["fingerprint"],
        "layer": 30, "seed": v2.CONFIG["seed"] + 30, "requested_anchors": 24, "spectral_state_sha256": v2.base.sha(diagnostic / "spectral_state.pt"),
        "rank_cutoffs": {"frozen_original_shape": {"rank": 3}, "actual_svd_shape": {"rank": 64}}})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return root, diagnostic / "result.json", source


def test_prepare_reuses_bytes_without_relabeling_production_or_selecting_tasks(previous):
    old, diagnostic, source = previous
    root = old.parent / "v2"
    result = v2.prepare(root, previous_root=old, diagnostic_path=diagnostic)
    assert result["no_model_or_evaluation_executed"]
    assert (root / "tasks.json").read_bytes() == (old / "tasks.json").read_bytes()
    relative = "groups/group_00/block_31/relation.pt"
    assert (root / relative).samefile(old / relative)
    wrapper = json.loads((root / "groups/group_00/block_31/fit_result/manifest.json").read_text())
    old_plan = json.loads((old / "plan.json").read_text())
    assert wrapper["fingerprint"] == result["fingerprint"] != old_plan["fingerprint"]
    assert wrapper["reused_from"]["source_fingerprint"] == old_plan["fingerprint"]
    assert wrapper["original_stage_metadata"]["fingerprint"] == old_plan["fingerprint"]
    assert not (root / "groups/group_00/block_30/relation.pt").exists()
    assert not (root / "logs").exists()
    with v2.protocol_context():
        assert v2.base.checked(root, root / "groups/group_00/block_31/fit_result")["provenance_kind"] == "verified_immutable_reuse"
    with pytest.raises(FileExistsError):
        v2.prepare(root, previous_root=old, diagnostic_path=diagnostic)


@pytest.mark.parametrize("violation", ["training", "heldout", "source", "archive", "artifact", "spectrum", "incomplete"])
def test_prepare_rejects_invalid_reuse_before_creating_destination(previous, violation):
    old, diagnostic, source = previous
    if violation == "training":
        _write(old / "groups/group_00/recovery/progress.json", {})
    elif violation == "heldout":
        (old / "coordinator.jsonl").write_text(json.dumps({"kind": "phase_start", "stage": "evaluate"}) + "\n")
    elif violation == "source":
        source.write_text("changed live source")
    elif violation == "archive":
        (old / "producer/source.py").write_text("changed archive")
    elif violation == "artifact":
        (old / "groups/group_00/block_31/relation.pt").write_bytes(b"changed")
    elif violation == "spectrum":
        value = json.loads(diagnostic.read_text())
        value["rank_cutoffs"]["actual_svd_shape"]["rank"] = 5
        _write(diagnostic, value)
    else:
        (old / "groups/group_00/block_31/fit_result/manifest.json").unlink()
    root = old.parent / "v2"
    with pytest.raises(ValueError):
        v2.prepare(root, previous_root=old, diagnostic_path=diagnostic)
    assert not root.exists()


def test_hash_audit_hashes_shared_input_once_and_detects_mutation(tmp_path, monkeypatch):
    path = tmp_path / "input"
    path.write_bytes(b"data")
    original = v2.base.sha
    calls = []
    monkeypatch.setattr(v2.base, "sha", lambda p: (calls.append(p), original(p))[1])
    audit = v2.HashAudit()
    value = audit.check(path)
    assert audit.check(path, value) == value
    assert len(calls) == 1
    path.write_bytes(b"mutated")
    with pytest.raises(ValueError):
        audit.check(path, value)
