"""CPU runner contract tests. Model, CUDA, lease and analysis backends are mocked."""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.structure_net import jspace_relation_study as study
from experiments.structure_net.jspace_relation_tasks import build_dataset
from structure_net_resources.client import ReservationError


@pytest.fixture(scope="module")
def dataset():
    return build_dataset()


def rewrite(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def test_intervention_selection_is_frozen_balanced_and_independent_of_input_order(dataset):
    cases = dataset["splits"]["evaluation"]
    selected = study.select_intervention_cases(cases)
    assert len(selected) == 64
    assert study.select_intervention_cases(list(reversed(cases))) == selected
    cells = Counter((row["intermediate"], row["answer"]) for row in selected)
    assert len(cells) == 16 and set(cells.values()) == {4}
    for cell in cells:
        expected = sorted((r["case_id"] for r in cases if (r["intermediate"], r["answer"]) == cell))[:4]
        assert [r["case_id"] for r in selected if (r["intermediate"], r["answer"]) == cell] == expected
    with pytest.raises(ValueError, match="16"):
        study.select_intervention_cases(cases[:10])


@pytest.fixture
def frozen_plan(tmp_path, monkeypatch):
    repository, root = tmp_path / "repo", tmp_path / "run"
    repository.mkdir()
    root.mkdir()
    source = "source.py"
    (repository / source).write_text("# frozen\n")
    (root / "producer").mkdir()
    (root / "producer" / source).write_text("# frozen\n")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"synthetic bytes, not a model")
    study.write_json(root / "tasks.json", {"synthetic": True})
    study.write_json(root / "calibration.json", {"rows": []})
    plan = {"study_id": study.STUDY_ID, "config": asdict(study.CONFIG),
            "producer_sha256": {source: study.file_hash(repository / source)},
            "dataset_digest": study.digest({"synthetic": True}),
            "calibration_sha256": study.file_hash(root / "calibration.json"),
            "model": {"path": str(checkpoint), "files": {"weights.bin": study.file_hash(checkpoint / "weights.bin")}}}
    plan["scientific_fingerprint"] = study.digest(plan)
    study.write_json(root / "plan.json", plan)
    monkeypatch.setattr(study, "ROOT", repository)
    return root, repository, checkpoint, plan


@pytest.mark.parametrize("mutation", [None, "plan", "live_source", "archive", "data", "calibration", "weights"])
def test_frozen_plan_detects_artifact_mutation(frozen_plan, mutation):
    root, repository, checkpoint, plan = frozen_plan
    if mutation is None:
        assert study.verify_plan(root)["scientific_fingerprint"] == plan["scientific_fingerprint"]
        return
    paths = {"plan": root / "plan.json", "live_source": repository / "source.py",
             "archive": root / "producer/source.py", "data": root / "tasks.json",
             "calibration": root / "calibration.json", "weights": checkpoint / "weights.bin"}
    if mutation == "plan":
        plan["config"]["seed"] += 1
        rewrite(paths[mutation], plan)
    elif mutation == "data":
        rewrite(paths[mutation], {"synthetic": "changed"})
    else:
        paths[mutation].write_text("changed")
    with pytest.raises(ValueError):
        study.verify_plan(root)


def test_json_and_tensor_serialization_is_exclusive_and_reloadable(tmp_path):
    tensor = torch.arange(16).reshape(2, 8).float()
    study.save_tensor(tmp_path / "tensor.pt", tensor)
    assert torch.equal(torch.load(tmp_path / "tensor.pt", weights_only=True), tensor)
    study.write_json(tmp_path / "record.json", {"good": True})
    with pytest.raises(FileExistsError):
        study.save_tensor(tmp_path / "tensor.pt", tensor + 1)
    with pytest.raises(FileExistsError):
        study.write_json(tmp_path / "record.json", {"good": False})
    with pytest.raises(ValueError):
        study.write_json(tmp_path / "nonfinite.json", {"bad": float("nan")})
    assert not (tmp_path / "nonfinite.json").exists()
    assert json.loads((tmp_path / "record.json").read_text()) == {"good": True}


def test_lease_guard_runs_before_cuda_or_model_calls(tmp_path, monkeypatch):
    calls = []

    def refuse(*args, **kwargs):
        calls.append("lease")
        raise ReservationError("no test lease")

    def forbidden(*args, **kwargs):
        pytest.fail("CUDA/model/provenance was touched before lease validation")

    monkeypatch.setattr("structure_net_resources.client.require_gpu_lease", refuse)
    monkeypatch.setattr(study, "verify_plan", forbidden)
    monkeypatch.setattr(study, "load_small_qwen", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", forbidden)
    with pytest.raises(ReservationError, match="no test lease"):
        study.capture(tmp_path / "absent")
    assert calls == ["lease"] and not (tmp_path / "absent").exists()


def test_cpu_prepare_refuses_visible_cuda_before_loading_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        study.prepare(tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_projection_preserves_continuous_values_and_checks_shape_finiteness():
    hidden = torch.arange(48).reshape(3, 2, 8).float() / 11 - 2.5
    banks = {arm: torch.eye(8)[:4].repeat(2, 1, 1) for arm in study.ARMS}
    result = study.project_scores(hidden, banks)
    for scores in result.values():
        assert scores.shape == (3, 2, 4) and scores.dtype == np.float64
        np.testing.assert_array_equal(scores, hidden[..., :4].double().numpy())
        assert np.any(scores < 0) and np.any((scores % 1) != 0)
    with pytest.raises(ValueError):
        study.project_scores(hidden.reshape(6, 8), banks)
    bad = hidden.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        study.project_scores(bad, banks)
    banks["pca"] = torch.ones(2, 3, 8)
    with pytest.raises(ValueError):
        study.project_scores(hidden, banks)


def capture_data(cases, *, arm="natural", site=None):
    records = [{"case": case, "case_id": case["case_id"], "hidden_row_index": index,
                "observation_layers": list(study.CONFIG.layers), "arm": arm, "source_site": site,
                "original_correct": True, "counterfactual_correct": False, "answer_status": "complete"}
               for index, case in enumerate(cases)]
    hidden = torch.randn(len(cases), len(study.CONFIG.layers), study.CONFIG.hidden_width,
                         generator=torch.Generator().manual_seed(123))
    return {"records": records, "hidden": hidden}


@pytest.mark.parametrize("mutation", [None, "row", "layer", "case_id", "case", "dtype", "nonfinite"])
def test_loaded_capture_validates_tensor_and_record_alignment(tmp_path, dataset, monkeypatch, mutation):
    monkeypatch.setattr(study, "CONFIG", replace(study.CONFIG, hidden_width=8))
    cases = dataset["splits"]["train"][:2]
    data = capture_data(cases)
    if mutation == "row":
        data["records"][0]["hidden_row_index"] = 1
    elif mutation == "layer":
        data["records"][0]["observation_layers"] = list(reversed(study.CONFIG.layers))
    elif mutation == "case_id":
        data["records"][0]["case_id"] = "other"
    elif mutation == "case":
        data["records"][0]["case"] = cases[1]
    elif mutation == "dtype":
        data["hidden"] = data["hidden"].half()
    elif mutation == "nonfinite":
        data["hidden"][0, 0, 0] = float("inf")
    study._save_capture(tmp_path / "capture/train", data)
    if mutation is None:
        hidden, records = study._load_split(tmp_path, "train", cases)
        assert torch.equal(hidden, data["hidden"]) and records == data["records"]
    else:
        with pytest.raises(ValueError):
            study._load_split(tmp_path, "train", cases)


def report_with_counts(jacobian=(0, 0, 0), pca=(0, 0, 0), random=(0, 0, 0)):
    return {"banks": {arm: {"summary": dict(zip(("layer_relations_passing_test",
                          "joint_relations_passing_test", "propagation_passing_dev_and_test"), counts))}
                      for arm, counts in zip(study.ARMS, (jacobian, pca, random))}}


@pytest.mark.parametrize("counts,classification", [
    ((0, 0, 0), "no_heldout_relations_detected_in_bounded_search"),
    ((1, 0, 0), "heldout_relations_without_validated_propagation"),
    ((0, 1, 1), "validated_approximate_propagation"),
])
def test_summary_classifies_primary_j_evidence_separately_from_controls(counts, classification):
    summary = study.study_summary(report_with_counts(jacobian=counts, pca=(4, 4, 4)))
    assert summary["primary_classification"] == classification
    assert summary["jacobian_more_validated_consequences_than_both_controls"] is False


@pytest.fixture
def simulated_capture(tmp_path, monkeypatch, dataset):
    """Exercise disk orchestration, replacing every device/model/lease operation."""
    monkeypatch.setattr(study, "CONFIG", replace(study.CONFIG, hidden_width=8))
    root = tmp_path / "run"
    root.mkdir()
    plan = {"scientific_fingerprint": "f" * 64, "config": asdict(study.CONFIG),
            "model": {"path": "unused-mocked-model"}, "concept_token_ids": [0, 1, 2, 3],
            "final_layer_dictionary": "identity", "intervention_case_ids": [
                r["case_id"] for r in study.select_intervention_cases(dataset["splits"]["evaluation"])]}
    study.write_json(root / "tasks.json", dataset)
    study.write_json(root / "calibration.json", {"rows": [{"text": "CPU fixture"}] * 16})
    study.write_json(root / "plan.json", plan)
    plan = json.loads((root / "plan.json").read_text())
    lease = {"id": "synthetic", "holder": "structure_net", "state": "active", "ready": True,
             "allocation": {"mock-card": 128}, "pid": 1, "pid_start": 1, "end": 9999999999}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "mock-card")
    monkeypatch.setattr("structure_net_resources.client.require_gpu_lease", lambda *_: lease)
    monkeypatch.setattr(study, "verify_plan", lambda *args, **kwargs: plan)
    for name in ("is_available", "is_bf16_supported"):
        monkeypatch.setattr(torch.cuda, name, lambda: True)
    for name in ("manual_seed_all", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch, "manual_seed", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "CPU test double")
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 1024)
    mock_model = SimpleNamespace(get_output_embeddings=lambda: SimpleNamespace(weight=torch.eye(8)))
    monkeypatch.setattr(study, "load_small_qwen", lambda *args, **kwargs: (mock_model, object()))

    def natural(model, tokenizer, cases, *, layers, progress):
        data = capture_data(cases)
        for row in data["records"]:
            progress(row)
        return data

    def fit(model, tokenizer, prompts, *, source_layers, progress_callback, **kwargs):
        assert json.loads((root / "shakedown/result.json").read_text())["save_reload_equal"] is True
        for index in range(len(prompts)):
            progress_callback({"completed_prompts": index + 1})
        return SimpleNamespace(token_rows={layer: torch.eye(8)[:4] for layer in source_layers}, metadata={})

    def interventions(model, tokenizer, cases, *, layers, banks, concept_names, sites, progress):
        records = []
        for case in cases:
            for site in sites:
                for arm in study.ARMS:
                    row = capture_data([case], arm=arm, site=site)["records"][0]
                    row["hidden_row_index"] = len(records)
                    records.append(row)
                    progress(row)
        return {"records": records, "hidden": torch.randn(len(records), len(layers), 8,
                  generator=torch.Generator().manual_seed(7))}

    monkeypatch.setattr(study, "capture_natural", natural)
    monkeypatch.setattr(study, "fit_token_rows", fit)
    monkeypatch.setattr(study, "capture_interventions", interventions)
    original_threads = torch.get_num_threads()
    manifest = study.capture(root)
    yield root, plan, manifest
    torch.set_num_threads(original_threads)


def test_simulated_capture_has_exact_schedule_and_verified_roundtrip(simulated_capture):
    root, plan, manifest = simulated_capture
    assert manifest["counts"] == {"shakedown": 16, "lens_fit": 16, "train": 512,
                                  "development": 128, "evaluation": 128, "interventions": 384}
    assert study.verify_capture(root, plan) == manifest
    smoke = json.loads((root / "shakedown/result.json").read_text())
    assert smoke["save_reload_equal"] is True
    assert smoke["classification"] == "systems_lifecycle_only_not_quality_evidence"
    rows = json.loads((root / "capture/interventions/records.json").read_text())
    assert [(r["case_id"], r["source_site"], r["arm"]) for r in rows] == [
        (case_id, site, arm) for case_id in plan["intervention_case_ids"]
        for site in study.CONFIG.sites for arm in study.ARMS]
    with pytest.raises(FileExistsError):
        study.capture(root)


@pytest.mark.parametrize("mutation", ["counts", "missing_shakedown", "bytes"])
def test_manifest_rejects_incomplete_or_changed_capture(simulated_capture, mutation):
    root, plan, manifest = simulated_capture
    if mutation == "counts":
        manifest["counts"]["interventions"] -= 1
    elif mutation == "missing_shakedown":
        del manifest["artifacts"]["shakedown/hidden.pt"]
    else:
        (root / "capture/memory_pilot.json").write_text("changed")
    rewrite(root / "capture_manifest.json", manifest)
    with pytest.raises(ValueError):
        study.verify_capture(root, plan)


def test_analysis_freezes_before_evaluation_and_writes_one_campaign(simulated_capture, monkeypatch):
    from experiments.structure_net import jspace_relation_analysis as analysis

    root, plan, manifest = simulated_capture
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    real_load = study._load_split
    loaded = []

    def load(output, split, cases=None):
        if split in ("evaluation", "interventions"):
            assert (root / "analysis/frozen_models.json").is_file()
        loaded.append(split)
        return real_load(output, split, cases)

    def fit(scores, **kwargs):
        assert loaded == ["train", "development"]
        assert all(set(splits) == {"train", "development"} for splits in scores.values())
        return {"status": "frozen_before_evaluation", "marker": "CPU fixture"}

    def evaluate(frozen, evaluation, interventions, records):
        assert loaded == ["train", "development", "evaluation", "interventions"]
        assert frozen == json.loads((root / "analysis/frozen_models.json").read_text())
        assert len(records) == 384 and evaluation["jacobian"].shape == (128, 12, 4)
        return report_with_counts()

    monkeypatch.setattr(study, "_load_split", load)
    monkeypatch.setattr(analysis, "fit_models", fit)
    monkeypatch.setattr(analysis, "evaluate_models", evaluate)
    campaign = study.analyze(root)
    assert campaign["completed"] == campaign["requested"] == 1 and campaign["failed"] == 0
    assert campaign["exact_neural_claim"] is False and campaign["compression_tested"] is False
    assert campaign["result_sha256"] == study.file_hash(root / "analysis/result.json")
    assert campaign == json.loads((root / "campaign_results.json").read_text())
