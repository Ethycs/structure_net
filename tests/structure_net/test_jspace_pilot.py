"""Scientific gates and artifact invariants for the bounded J-space pilot."""
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import jspace_pilot as pilot
from experiments.structure_net.jspace_tasks import TwoHopCase


def cases():
    return [TwoHopCase("one", "x", "p", "France", "Paris", "Italy", "Rome"),
            TwoHopCase("two", "x", "p", "spider", "eight", "ant", "six")]


def gates():
    return {"answer_retention": .95, "accuracy_loss_max": .05,
            "next_concept_relative_l2_p95_max": .05,
            "retained_coordinate_relative_error_max": .005}


def test_retained_patch_preserves_scores_but_changes_complement_and_only_final_token():
    torch.manual_seed(23)
    bank = torch.linalg.qr(torch.randn(12, 4)).Q.T
    hidden = torch.randn(1, 5, 12)
    donor = torch.randn(12)
    patch, stats = pilot.retained_patch(bank, donor)
    changed = patch(hidden)
    assert torch.equal(changed[:, :-1], hidden[:, :-1])
    torch.testing.assert_close(changed[:, -1] @ bank.T, hidden[:, -1] @ bank.T)
    projector = bank.T @ bank
    torch.testing.assert_close(changed[0, -1] @ (torch.eye(12) - projector),
                               donor @ (torch.eye(12) - projector), atol=1e-6, rtol=1e-5)
    assert stats["retained_coordinate_relative_error"] < 1e-5
    assert stats["perturbation_relative_l2"] > .1
    # An affine mean reconstruction is the same algebra with a fixed donor.
    zero_strength, _ = pilot.retained_patch(bank, donor, alpha=0)
    torch.testing.assert_close(zero_strength(hidden), hidden)


def test_factor_bank_records_exact_semantics_and_truncation_is_not_exact():
    torch.manual_seed(4)
    raw = torch.randn(9, 12)
    normalized, basis, singular, coefficients = pilot.factor_bank(raw)
    torch.testing.assert_close(coefficients @ basis, normalized)
    h = torch.randn(12)
    torch.testing.assert_close(normalized @ h, coefficients @ (basis @ h))
    assert (normalized - (normalized @ basis[:2].T) @ basis[:2]).norm() > .1
    assert len(singular) == 9


def test_control_changes_match_jacobian_norm_without_identical_direction():
    torch.manual_seed(99)
    h = torch.randn(1, 6, 16)
    pair, control = torch.randn(2, 16), torch.randn(2, 16)
    patches = {arm: pilot.functional_patch(pair, control, arm, 4)
               for arm in ("jacobian", "unembedding", "random")}
    deltas = {name: patch(h) - h for name, patch in patches.items()}
    for name in ("unembedding", "random"):
        torch.testing.assert_close(deltas[name].norm(dim=-1), deltas["jacobian"].norm(dim=-1))
        assert not torch.allclose(deltas[name], deltas["jacobian"])
    torch.testing.assert_close(patches["random"](h), patches["random"](h))


def test_actual_downstream_change_fails_even_when_retained_scores_are_exact():
    c = cases()
    baseline = {"one": {"answer": "Paris"}, "two": {"answer": "8"}}
    diagnostics = {case.case_id: {"next_concept_relative_l2": .2,
                                 "retained_coordinate_relative_error": 0} for case in c}
    summary = pilot.metric_summary(c, baseline, baseline, diagnostics)
    assert summary["answer_retention"] == 1
    assert not pilot.passes_sufficiency(summary, gates())
    for row in diagnostics.values():
        row["next_concept_relative_l2"] = .001
    summary = pilot.metric_summary(c, baseline, baseline, diagnostics)
    assert pilot.passes_sufficiency(summary, gates())
    diagnostics["one"]["retained_coordinate_relative_error"] = .01
    assert not pilot.passes_sufficiency(pilot.metric_summary(c, baseline, baseline, diagnostics), gates())


def test_missing_answers_cannot_count_as_retained_or_correct():
    c = cases()
    baseline = {"one": {"answer": None}, "two": {"answer": "eight"}}
    diagnostics = {case.case_id: {"next_concept_relative_l2": 0,
                                 "retained_coordinate_relative_error": 0} for case in c}
    summary = pilot.metric_summary(c, baseline, baseline, diagnostics)
    assert summary["answer_retention"] == .5
    assert summary["score"]["baseline_accuracy"]["successes"] == 1


def test_strict_results_are_exclusive_and_reject_nonfinite(tmp_path):
    path = tmp_path / "result.json"
    pilot.write_json(path, {"ok": True})
    with pytest.raises(FileExistsError):
        pilot.write_json(path, {"ok": False})
    assert json.loads(path.read_text()) == {"ok": True}
    with pytest.raises(ValueError):
        pilot.write_json(tmp_path / "invalid.json", {"metric": float("nan")})
    assert not (tmp_path / "invalid.json").exists()


def test_producer_change_cannot_reuse_frozen_plan(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    artifact = tmp_path / "artifact"
    (artifact / "producer").mkdir(parents=True)
    (repo / "code.py").write_text("first")
    (artifact / "producer" / "code.py").write_text("first")
    for name in ("tasks.json", "calibration.json"):
        (artifact / name).write_text("{}")
    plan = {"producer_sha256": {"code.py": pilot.file_hash(repo / "code.py")},
            "data_sha256": pilot.file_hash(artifact / "tasks.json"),
            "calibration_sha256": pilot.file_hash(artifact / "calibration.json"), "models": []}
    plan["scientific_fingerprint"] = pilot.digest(plan)
    pilot.write_json(artifact / "plan.json", plan)
    monkeypatch.setattr(pilot, "ROOT", repo)
    assert pilot.verify_plan(artifact) == plan
    (repo / "code.py").write_text("second")
    with pytest.raises(ValueError, match="Producer changed"):
        pilot.verify_plan(artifact)


def test_gpu_execution_refuses_unleased_before_reading_plan_or_initializing_cuda(monkeypatch):
    from structure_net_resources.client import ReservationError
    monkeypatch.delenv("IC_LEASE", raising=False)
    monkeypatch.delenv("STRUCTURENET_GPU_LEASE", raising=False)
    monkeypatch.setattr(pilot, "verify_plan", lambda _: pytest.fail("unleased execution read the plan"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("unleased CUDA initialization"))
    with pytest.raises(ReservationError, match="no IC_LEASE"):
        pilot.run("does-not-exist")
