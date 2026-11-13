"""Synthetic artifact audits and mocked registration; never open a real database."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from neural_architecture_lab import jspace_relation_meta_hypothesis as meta


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def metric(arm, site, rows, original, counterfactual):
    return {"arm": arm, "source_site": site, "rows": rows,
            "original_correct": original, "original_accuracy": original / rows,
            "counterfactual_correct": counterfactual, "counterfactual_accuracy": counterfactual / rows,
            "answer_status_counts": {"complete": rows}}


def counts(n):
    return {**{key: n for key in meta.COUNT_KEYS}, "symbolic_status_counts": {"ok": 1}}


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / "study"
    source = root / "producer/experiments/structure_net/relation_study.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic frozen producer; no model or database is accessed\n")
    split_counts = {"train": 8, "development": 4, "evaluation": 4, "shakedown": 2}
    tasks = {"manifest": {"counts": split_counts}, "splits": {
        name: [{"case_id": f"{name}-{i}"} for i in range(n)] for name, n in split_counts.items()}}
    write(root / "tasks.json", tasks)
    write(root / "calibration.json", {"rows": [{"text": "first fixture"}, {"text": "second fixture"}]})
    config = {**{name + "_count": count for name, count in split_counts.items()},
              "sites": [1], "hidden_width": 8, "model_depth": 3, "layers": [0, 1], "edges": [[0, 1]]}
    plan = {
        "schema_version": 1, "study_id": meta.HYPOTHESIS_ID, "conformance": "EXPLORATORY",
        "prepared_at": "2026-11-13T08:00:00+00:00", "config": config,
        "producer_sha256": {"experiments/structure_net/relation_study.py": meta._hash(source)},
        "dataset_digest": meta._digest(tasks), "dataset_manifest": tasks["manifest"],
        "calibration_sha256": meta._hash(root / "calibration.json"),
        "intervention_case_ids": ["evaluation-0", "evaluation-1"],
        "model": {"path": "/cache/models--Qwen--Qwen2.5-3B-Instruct/snapshots/revision",
                  "revision": "revision", "files": {"model.safetensors": "a" * 64}},
        "not_claimed": ["exact_neural_ideal", "compression", "full_J_space", "independent_model_replication"],
    }
    plan["scientific_fingerprint"] = meta._digest(plan)
    write(root / "plan.json", plan)
    capture_artifacts = {}
    for split in ("train", "development", "evaluation", "interventions"):
        folder = root / "capture" / split
        folder.mkdir(parents=True)
        (folder / "hidden.pt").write_bytes(b"synthetic tensor placeholder; do not deserialize")
        write(folder / "records.json", {"fixture": split})
        for name in ("hidden.pt", "records.json"):
            capture_artifacts[f"capture/{split}/{name}"] = meta._hash(folder / name)
    for name in ("capture/banks.pt", "capture/lens.pt", "capture/lease.json", "shakedown/result.json"):
        write(root / name, {"fixture": name})
        capture_artifacts[name] = meta._hash(root / name)
    capture = {"schema_version": 1, "study_id": meta.HYPOTHESIS_ID, "status": "completed",
               "scientific_fingerprint": plan["scientific_fingerprint"],
               "counts": {**split_counts, "lens_fit": 2, "interventions": 6}, "artifacts": capture_artifacts}
    write(root / "capture_manifest.json", capture)
    frozen = {"status": "frozen_before_evaluation", "exact_neural_claim": False, "banks": {}, "seed": 7}
    frozen["content_sha256"] = meta._frozen_digest(frozen)
    write(root / "analysis/frozen_models.json", frozen)
    by_bank = {"jacobian": counts(2), "pca": counts(1), "random": counts(0)}
    summary = {"primary_classification": "validated_approximate_propagation", "by_bank": by_bank,
               "jacobian_more_validated_consequences_than_both_controls": True,
               "comparison_interpretation": "descriptive counts only"}
    metrics = {name: [metric("natural", None, n, n - 1, 0)]
               for name, n in split_counts.items() if name != "shakedown"}
    metrics["interventions"] = [metric(arm, 1, 2, 1, int(arm == "jacobian")) for arm in meta.ARMS]
    metrics["interventions_originally_correct_only"] = []
    result = {
        "schema_version": 1, "study_id": meta.HYPOTHESIS_ID, "status": "completed",
        "conformance": "EXPLORATORY", "scientific_fingerprint": plan["scientific_fingerprint"],
        "config": config, "task_metrics": metrics, "summary": summary,
        "relation_results": {"banks": {arm: {"summary": value} for arm, value in by_bank.items()},
                             "frozen_content_sha256": frozen["content_sha256"], "exact_neural_claim": False},
        "frozen_models_sha256": meta._hash(root / "analysis/frozen_models.json"),
        "capture_manifest_sha256": meta._hash(root / "capture_manifest.json"),
        "elapsed_seconds": 12.5, "completed_at": "2026-11-13T08:10:00+00:00",
        "exact_neural_claim": False, "compression_tested": False,
    }
    write(root / "analysis/result.json", result)
    campaign = {
        "schema_version": 1, "study_id": meta.HYPOTHESIS_ID, "status": "completed", "conformance": "EXPLORATORY",
        "scientific_fingerprint": plan["scientific_fingerprint"], "config": config,
        "capture_counts": capture["counts"], "requested": 1, "completed": 1, "failed": 0, "retried": 0, "reused": 0,
        "task_metrics": metrics, "summary": summary, "result_path": "analysis/result.json",
        "result_sha256": meta._hash(root / "analysis/result.json"),
        "frozen_models_sha256": result["frozen_models_sha256"],
        "exact_neural_claim": False, "compression_tested": False,
        "limitations": ["single_checkpoint", "bounded_polynomial_search"],
    }
    results = root / "campaign_results.json"
    write(results, campaign)
    report = tmp_path / "report.md"
    report.write_text("One exploratory approximate-surrogate study; no compression or exact neural proof.\n")
    return {"root": root, "results": results, "report": report, "campaign": campaign,
            "plan": plan, "result": result, "capture": capture, "frozen": frozen}


def refresh_direct(a):
    write(a["root"] / "analysis/result.json", a["result"])
    a["campaign"]["result_sha256"] = meta._hash(a["root"] / "analysis/result.json")
    write(a["results"], a["campaign"])


def build(a):
    return meta.build_jspace_relation_meta_hypothesis(a["results"], a["report"])


def test_positive_relation_counts_never_confirm_or_become_accuracy(artifacts):
    record = build(artifacts)
    assert record["hypothesis"]["id"] == meta.HYPOTHESIS_ID
    assert record["hypothesis"]["evidence_count"] == record["hypothesis"]["direct_experiment_count"] == 1
    assert record["hypothesis"]["confirmed"] is False
    assert record["result"]["confirmed"] is False
    assert record["result"]["exact_neural_claim"] is False
    assert record["result"]["compression_tested"] is False
    assert record["result"]["model"]["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert record["result"]["surrogate_fitting_performed"] is True
    experiment, = meta.build_jspace_relation_experiment_results(record)
    assert experiment.primary_metric == 2
    assert experiment.metrics["accuracy"] == .75
    assert experiment.metrics["evaluation_task_total"] == 4
    assert experiment.metrics["intervention_jacobian_site_1_counterfactual_accuracy"] == .5
    assert experiment.training_time == 0
    assert experiment.timestamp.isoformat() == artifacts["result"]["completed_at"]
    assert len(record["evidence"]["direct_tests"]) == 1
    assert str((artifacts["root"] / "capture/train/hidden.pt").resolve()) in record["source_sha256"]


def test_zero_relation_result_is_still_a_completed_direct_test(artifacts):
    a = artifacts
    for arm in meta.ARMS:
        a["result"]["summary"]["by_bank"][arm].update(counts(0))
    a["result"]["summary"]["primary_classification"] = "no_heldout_relations_detected_in_bounded_search"
    a["result"]["summary"]["jacobian_more_validated_consequences_than_both_controls"] = False
    refresh_direct(a)
    record = build(a)
    experiment, = meta.build_jspace_relation_experiment_results(record)
    assert record["hypothesis"]["tested"] is True
    assert record["hypothesis"]["confirmed"] is False
    assert experiment.primary_metric == 0
    assert experiment.metrics["accuracy"] == .75


@pytest.mark.parametrize("key,value", [
    ("status", "running"), ("completed", 0), ("failed", 1), ("retried", 1), ("reused", 1),
    ("schema_version", 2), ("schema_version", True), ("study_id", "wrong-study"),
    ("exact_neural_claim", True), ("compression_tested", True), ("confirmed", True),
    ("conformance", "CONFIRMATORY"), ("result_sha256", "0" * 64),
    ("scientific_fingerprint", "0" * 64), ("result_path", "../outside.json"),
])
def test_changed_or_uncompleted_campaign_is_rejected(artifacts, key, value):
    artifacts["campaign"][key] = value
    write(artifacts["results"], artifacts["campaign"])
    with pytest.raises(ValueError):
        build(artifacts)


@pytest.mark.parametrize("relative", [
    "analysis/result.json", "analysis/frozen_models.json", "capture_manifest.json",
    "capture/train/hidden.pt", "capture/interventions/records.json", "capture/banks.pt",
    "producer/experiments/structure_net/relation_study.py", "calibration.json",
])
def test_any_changed_hashed_artifact_is_rejected(artifacts, relative):
    path = artifacts["root"] / relative
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        build(artifacts)


def test_task_data_uses_canonical_content_digest_not_whitespace_hash(artifacts):
    # Unlike binary artifacts, tasks.json is frozen by canonical JSON digest.
    path = artifacts["root"] / "tasks.json"
    tasks = json.loads(path.read_text())
    path.write_text(json.dumps(tasks, indent=4))
    assert build(artifacts)["hypothesis"]["confirmed"] is False
    tasks["splits"]["evaluation"][0]["case_id"] = "changed"
    write(path, tasks)
    with pytest.raises(ValueError, match="dataset"):
        build(artifacts)


def test_builder_requires_real_report_and_rejects_nonfinite_json(artifacts):
    artifacts["report"].unlink()
    with pytest.raises(ValueError, match="report"):
        build(artifacts)
    artifacts["report"].write_text("report")
    artifacts["campaign"]["extra"] = float("nan")
    write(artifacts["results"], artifacts["campaign"])
    with pytest.raises(ValueError):
        build(artifacts)


def test_summary_must_agree_with_direct_counts_and_classification(artifacts):
    artifacts["result"]["summary"]["by_bank"]["jacobian"]["propagation_passing_dev_and_test"] = 9
    refresh_direct(artifacts)
    with pytest.raises(ValueError, match="Inconsistent relation"):
        build(artifacts)


def test_task_accuracy_must_match_the_measured_counts(artifacts):
    artifacts["result"]["task_metrics"]["evaluation"][0]["original_accuracy"] = .95
    refresh_direct(artifacts)
    with pytest.raises(ValueError, match="numerator/denominator"):
        build(artifacts)


def test_campaign_summary_cannot_replace_the_hashed_direct_result(artifacts):
    artifacts["campaign"]["summary"] = deepcopy(artifacts["campaign"]["summary"])
    artifacts["campaign"]["summary"]["primary_classification"] = "fake_success"
    write(artifacts["results"], artifacts["campaign"])
    with pytest.raises(ValueError, match="differs from direct"):
        build(artifacts)


def test_producer_path_cannot_escape_its_frozen_archive(artifacts):
    a = artifacts
    a["plan"]["producer_sha256"] = {"../tasks.json": meta._hash(a["root"] / "tasks.json")}
    fp = meta._digest({k: v for k, v in a["plan"].items() if k != "scientific_fingerprint"})
    for value in (a["plan"], a["campaign"], a["result"]):
        value["scientific_fingerprint"] = fp
    write(a["root"] / "plan.json", a["plan"])
    refresh_direct(a)
    with pytest.raises(ValueError, match="escaping artifact"):
        build(a)


class Collection:
    def __init__(self):
        self.rows = {}
        self.corrupt = None

    def get(self, ids, include):
        result = {"ids": [key for key in ids if key in self.rows],
                  "metadatas": [dict(self.rows[key]) for key in ids if key in self.rows]}
        if self.corrupt:
            self.corrupt(result)
        return result


class Logger:
    def __init__(self):
        self.hypotheses_collection = Collection()
        self.experiments_collection = Collection()

    def log_hypothesis(self, record):
        self.hypotheses_collection.rows[record["id"]] = record
        return record["id"]

    def log_experiment_result(self, result):
        key = hashlib.sha256(json.dumps(asdict(result), default=str, sort_keys=True).encode()).hexdigest()[:16]
        self.experiments_collection.rows[key] = {
            "experiment_id": result.experiment_id, "hypothesis_id": result.hypothesis_id,
            "accuracy": result.metrics["accuracy"], "primary_metric": result.primary_metric,
        }
        return key


def test_mocked_storage_is_idempotent_receipts_append_only_and_readback_verified(artifacts, tmp_path, monkeypatch):
    logger = Logger()
    monkeypatch.setattr(meta, "_make_logger", lambda _: logger)
    kwargs = dict(results_path=artifacts["results"], report_path=artifacts["report"])
    first = meta.store_jspace_relation_meta_hypothesis(**kwargs, output_path=tmp_path / "first.json")
    second = meta.store_jspace_relation_meta_hypothesis(**kwargs, output_path=tmp_path / "second.json")
    assert first["storage"]["result_hashes"] == second["storage"]["result_hashes"]
    assert len(logger.hypotheses_collection.rows) == len(logger.experiments_collection.rows) == 1
    assert first["storage"]["readback"]["verified"] is True
    assert first["storage"]["readback"]["accuracy"] == .75
    assert first["storage"]["readback"]["primary_metric"] == 2
    with pytest.raises(FileExistsError):
        meta.store_jspace_relation_meta_hypothesis(**kwargs, output_path=tmp_path / "first.json")


@pytest.mark.parametrize("collection,field,value", [
    ("hypotheses_collection", "confirmed", True), ("hypotheses_collection", "evidence_count", 12),
    ("hypotheses_collection", "direct_experiment_count", 3),
    ("hypotheses_collection", "evidence_count", True),
    ("experiments_collection", "accuracy", .95), ("experiments_collection", "primary_metric", .75),
    ("experiments_collection", "hypothesis_id", "wrong"),
])
def test_bad_mocked_readback_cannot_leave_success_receipt(artifacts, tmp_path, monkeypatch, collection, field, value):
    logger = Logger()
    getattr(logger, collection).corrupt = lambda response: response["metadatas"][0].update({field: value})
    monkeypatch.setattr(meta, "_make_logger", lambda _: logger)
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="readback"):
        meta.store_jspace_relation_meta_hypothesis(artifacts["results"], artifacts["report"], output)
    assert not output.exists()


def test_incomplete_campaign_never_opens_registry(artifacts, tmp_path, monkeypatch):
    artifacts["campaign"]["status"] = "capturing"
    write(artifacts["results"], artifacts["campaign"])
    monkeypatch.setattr(meta, "_make_logger", lambda _: pytest.fail("must not open a database"))
    with pytest.raises(ValueError):
        meta.store_jspace_relation_meta_hypothesis(artifacts["results"], artifacts["report"], tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_cli_validate_only_never_opens_registry_or_writes_receipt(artifacts, tmp_path, monkeypatch, capsys):
    from experiments.neural_architecture_lab.store_jspace_relation_meta_hypothesis import main

    monkeypatch.setattr(meta, "_make_logger", lambda _: pytest.fail("must not open a database"))
    output = tmp_path / "unused.json"
    main(["--results", str(artifacts["results"]), "--report", str(artifacts["report"]),
          "--output", str(output), "--validate-only"])
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "validated_without_storage"
    assert response["confirmed"] is False
    assert not output.exists()
