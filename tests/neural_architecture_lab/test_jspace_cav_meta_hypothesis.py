"""Synthetic, CPU-only provenance and registration checks for one pilot."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from neural_architecture_lab import jspace_cav_meta_hypothesis as meta


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def artifacts(tmp_path):
    root = tmp_path / "pilot"
    source = root / "producer/experiments/structure_net/pilot.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen producer\n")
    write(root / "tasks.json", {"tasks": []})
    write(root / "calibration.json", {"rows": []})
    model = {"path": "/cache/models--Qwen--Qwen2.5-0.5B/snapshots/revision",
             "revision": "revision", "files": {"config.json": "f" * 64}, "width": 896}
    config = {"schema_version": 1, "hypothesis_id": meta.HYPOTHESIS_ID,
              "conformance": "EXPLORATORY", "prepared_at": "2026-11-13T07:28:00+00:00",
              "producer_sha256": {"experiments/structure_net/pilot.py": meta._hash(source)},
              "data_sha256": meta._hash(root / "tasks.json"),
              "calibration_sha256": meta._hash(root / "calibration.json"),
              "models": [model], "not_claimed": ["global_closure", "compression_speedup"]}
    config["scientific_fingerprint"] = meta._digest(config)
    score = {"raw_accuracy": {"accuracy": .25}, "baseline_accuracy": {"accuracy": .75},
             "parent_correct_accuracy": {"accuracy": 1 / 3}}
    result = {"schema_version": 1, "hypothesis_id": meta.HYPOTHESIS_ID,
              "scientific_fingerprint": config["scientific_fingerprint"],
              "status": "completed", "conformance": "EXPLORATORY", "confirmed": False,
              "classification": "joint_gate_not_met", "selection": {"model": model["path"]},
              "competence_screens": [{"model": model["path"], "correct": 9, "total": 12}],
              "functional": {"jacobian": score}, "elapsed_seconds": 20,
              "gates": {"enough_parent_correct": True, "causal": False,
                        "local_sufficiency": False, "j_specific_retention": False},
              "autonomous_reduced_transition_established": False,
              "not_claimed": config["not_claimed"]}
    run = root / "runs/pilot/seed_7"
    write(run / "result.json", result)
    write(root / "plan.json", config)
    campaign = {"schema_version": 1, "hypothesis_id": meta.HYPOTHESIS_ID,
                "scientific_fingerprint": config["scientific_fingerprint"],
                "config": config, "result": result, "artifacts": {"run": str(run)},
                "completion_counts": {"requested": 1, "completed": 1, "failed": 0, "reused": 0}}
    results = root / "campaign_results.json"
    write(results, campaign)
    report = tmp_path / "report.md"
    report.write_text("An exploratory local test, not a replication.\n")
    return results, report, campaign


def test_builder_preserves_one_direct_experiment_and_raw_accuracy(artifacts):
    results, report, _ = artifacts
    record = meta.build_jspace_cav_meta_hypothesis(results, report)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["evidence_count"] == 1
    assert record["result"]["classification"] == "joint_gate_not_met"
    assert record["result"]["model"]["model_id"] == "Qwen/Qwen2.5-0.5B"
    assert "other token states and context remain intact" in record["result"]["local_context_boundary"]
    experiments = meta.build_jspace_cav_experiment_results(record)
    assert len(experiments) == 1
    assert experiments[0].metrics["accuracy"] == .25
    assert experiments[0].metrics["parent_accuracy"] == .75
    assert experiments[0].training_time == 0


def test_passing_gates_are_still_exploratory_not_confirmed(artifacts):
    results, report, campaign = artifacts
    campaign["result"]["gates"] = {k: True for k in campaign["result"]["gates"]}
    campaign["result"]["classification"] = "promising_local_causal_and_sufficiency_result"
    write(Path(campaign["artifacts"]["run"]) / "result.json", campaign["result"])
    write(results, campaign)
    record = meta.build_jspace_cav_meta_hypothesis(results, report)
    assert record["result"]["confirmed"] is False
    assert record["hypothesis"]["confirmed"] is False
    assert meta.build_jspace_cav_experiment_results(record)[0].primary_metric == 1


def test_competence_stop_does_not_invent_evaluation_accuracy(artifacts):
    results, report, campaign = artifacts
    for key in ("functional", "selection", "gates"):
        campaign["result"].pop(key)
    campaign["result"]["classification"] = "parent_competence_insufficient"
    write(Path(campaign["artifacts"]["run"]) / "result.json", campaign["result"])
    write(results, campaign)
    record = meta.build_jspace_cav_meta_hypothesis(results, report)
    experiment = meta.build_jspace_cav_experiment_results(record)[0]
    assert record["result"]["model"] is None
    assert record["result"]["accuracy_scope"].endswith("evaluation_not_run")
    assert experiment.metrics["accuracy"] == .75
    assert experiment.metrics["evaluation_accuracy_measured"] == 0
    assert experiment.primary_metric == 0


@pytest.mark.parametrize("mutation", ["counts", "schema", "hypothesis", "fingerprint", "direct", "producer", "confirmed", "report"])
def test_builder_rejects_incomplete_or_changed_sources(artifacts, mutation):
    results, report, campaign = artifacts
    if mutation == "counts":
        campaign["completion_counts"]["failed"] = 1
    elif mutation == "schema":
        campaign["schema_version"] = 2
    elif mutation == "hypothesis":
        campaign["hypothesis_id"] = "unrelated"
    elif mutation == "fingerprint":
        campaign["config"]["scientific_fingerprint"] = "0" * 64
    elif mutation == "direct":
        write(Path(campaign["artifacts"]["run"]) / "result.json", {})
    elif mutation == "producer":
        (results.parent / "producer/experiments/structure_net/pilot.py").write_text("changed")
    elif mutation == "confirmed":
        campaign["result"]["confirmed"] = True
    else:
        report.unlink()
    write(results, campaign)
    with pytest.raises(ValueError):
        meta.build_jspace_cav_meta_hypothesis(results, report)


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
        self.experiments_collection.rows[key] = {"experiment_id": result.experiment_id,
                                               "hypothesis_id": result.hypothesis_id}
        return key


def test_storage_is_idempotent_but_output_is_append_only(artifacts, tmp_path, monkeypatch):
    results, report, _ = artifacts
    logger = Logger()
    monkeypatch.setattr(meta, "_make_logger", lambda _: logger)
    first = meta.store_jspace_cav_meta_hypothesis(results, report, tmp_path / "first.json")
    second = meta.store_jspace_cav_meta_hypothesis(results, report, tmp_path / "second.json")
    assert first["storage"]["result_hashes"] == second["storage"]["result_hashes"]
    assert len(logger.experiments_collection.rows) == len(logger.hypotheses_collection.rows) == 1
    assert first["storage"]["readback"]["verified"] is True
    with pytest.raises(FileExistsError):
        meta.store_jspace_cav_meta_hypothesis(results, report, tmp_path / "first.json")


@pytest.mark.parametrize("field,value", [("confirmed", True), ("evidence_count", 48), ("id", "wrong")])
def test_bad_readback_never_leaves_success_receipt(artifacts, tmp_path, monkeypatch, field, value):
    results, report, _ = artifacts
    logger = Logger()
    logger.hypotheses_collection.corrupt = lambda row: row["metadatas"][0].update({field: value})
    monkeypatch.setattr(meta, "_make_logger", lambda _: logger)
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="readback"):
        meta.store_jspace_cav_meta_hypothesis(results, report, output)
    assert not output.exists()


def test_real_standardized_logger_roundtrip_in_temporary_chroma(artifacts, tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    results, report, _ = artifacts
    original = meta._make_logger

    class LocalEmbedding:
        def __call__(self, input):
            return [[1.0, 0.0, 0.0] for _ in input]

    def logger_factory(path):
        logger = original(path)
        # Metadata readback is under test, not semantic retrieval or an embedding download.
        for collection in (logger.hypotheses_collection, logger.experiments_collection):
            assert collection is not None
            object.__setattr__(collection, "_embedding_function", LocalEmbedding())
        return logger

    monkeypatch.setattr(meta, "_make_logger", logger_factory)
    record = meta.store_jspace_cav_meta_hypothesis(
        results, report, tmp_path / "stored.json", chromadb_path=tmp_path / "chroma")
    assert record["storage"]["readback"]["confirmed"] is False
    assert record["storage"]["readback"]["evidence_count"] == 1
