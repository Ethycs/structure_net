"""Conservative NAL registration of a completed J-space relation study.

Builders are read-only and use the archived producer, frozen inputs, and capture
hashes. Chroma is imported only by the explicit store function, after validation.
Relation counts are descriptive results from one experiment, not independent
replications, exact neural identities, or evidence of compression.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re


HYPOTHESIS_ID = "JSPACE-RELATION-PROPAGATION-01"
ARMS = ("jacobian", "pca", "random")
COUNT_KEYS = (
    "layer_candidates_accepted_on_dev", "layer_relations_passing_test",
    "joint_candidates_accepted_on_dev", "joint_relations_passing_test",
    "eliminated_polynomials", "propagation_passing_dev", "propagation_passing_dev_and_test",
)
INTERPRETATION_BOUNDARY = (
    "Exploratory approximate relations measured through an intact frozen model. "
    "Exact algebra applies only to the specified rational surrogate, not the neural network. "
    "Relation/control counts are not independent replications or statistical J-specificity. "
    "Parent task accuracy is a separate endpoint; no compression, speedup, or autonomous closure was tested."
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _frozen_digest(value):
    body = {k: v for k, v in value.items() if k != "content_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    json.dumps(value, allow_nan=False)
    return value


def _inside(root, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("Artifact paths must be relative to their archive")
    resolved = (root / name).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Missing or escaping artifact: {name}")
    return resolved


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _check_artifact(root, name, expected, ledger):
    path = _inside(root, name)
    if not _sha(expected) or _hash(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {name}")
    ledger[str(path)] = expected
    return path


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def _no_positive_claims(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"confirmed", "exact_neural_claim", "compression_tested"} and child is not False:
                raise ValueError(f"Registration cannot assert {key}")
            _no_positive_claims(child)
    elif isinstance(value, list):
        for child in value:
            _no_positive_claims(child)


def _validate_task_metrics(metrics, config, intervention_cases):
    required = {"train", "development", "evaluation", "interventions"}
    if not isinstance(metrics, dict) or not required <= set(metrics):
        raise ValueError("Measured natural and intervention task metrics are required")
    for split, groups in metrics.items():
        if not isinstance(groups, list):
            raise ValueError("Task metrics must be grouped rows")
        seen = set()
        for group in groups:
            key = (group.get("arm"), group.get("source_site"))
            if key in seen:
                raise ValueError("Duplicate task metric group")
            seen.add(key)
            n = group.get("rows")
            if not _nonnegative_int(n) or n == 0:
                raise ValueError("Task metric denominators must be positive integers")
            for prefix in ("original", "counterfactual"):
                count, accuracy = group.get(prefix + "_correct"), group.get(prefix + "_accuracy")
                if (not _nonnegative_int(count) or count > n or type(accuracy) not in (int, float)
                        or not math.isfinite(accuracy) or not math.isclose(accuracy, count / n, abs_tol=1e-12)):
                    raise ValueError("Task accuracy does not match measured numerator/denominator")
            statuses = group.get("answer_status_counts")
            if (not isinstance(statuses, dict) or not all(_nonnegative_int(v) for v in statuses.values())
                    or sum(statuses.values()) != n):
                raise ValueError("Answer status counts do not match task denominator")
        if split in ("train", "development", "evaluation"):
            if (len(groups) != 1 or groups[0]["arm"] != "natural" or groups[0]["source_site"] is not None
                    or groups[0]["rows"] != config[split + "_count"]):
                raise ValueError("Natural task metrics differ from frozen split counts")
        elif split == "interventions":
            expected = {(arm, site) for arm in ARMS for site in config["sites"]}
            if seen != expected or any(g["rows"] != intervention_cases for g in groups):
                raise ValueError("Intervention task metrics differ from frozen schedule")


def _validate_summary(summary, relation_results):
    if not isinstance(summary, dict) or set(summary.get("by_bank", {})) != set(ARMS):
        raise ValueError("Summary must include J, PCA, and random banks")
    if set(relation_results.get("banks", {})) != set(ARMS):
        raise ValueError("Direct relation analysis must include all control banks")
    for arm, counts in summary["by_bank"].items():
        if any(not _nonnegative_int(counts.get(key)) for key in COUNT_KEYS):
            raise ValueError("Relation counts must be nonnegative integers")
        statuses = counts.get("symbolic_status_counts")
        if not isinstance(statuses, dict) or not all(_nonnegative_int(v) for v in statuses.values()):
            raise ValueError("Invalid symbolic status counts")
        if (counts["layer_relations_passing_test"] > counts["layer_candidates_accepted_on_dev"]
                or counts["joint_relations_passing_test"] > counts["joint_candidates_accepted_on_dev"]
                or not counts["propagation_passing_dev_and_test"] <= counts["propagation_passing_dev"] <= counts["eliminated_polynomials"]):
            raise ValueError("Inconsistent relation discovery/validation counts")
        if counts != relation_results["banks"][arm].get("summary"):
            raise ValueError("Summary differs from direct relation results")
    primary = summary["by_bank"]["jacobian"]
    classification = (
        "validated_approximate_propagation" if primary["propagation_passing_dev_and_test"] else
        "heldout_relations_without_validated_propagation"
        if primary["layer_relations_passing_test"] or primary["joint_relations_passing_test"] else
        "no_heldout_relations_detected_in_bounded_search"
    )
    if summary.get("primary_classification") != classification:
        raise ValueError("Classification does not follow the recorded J counts")
    more = primary["propagation_passing_dev_and_test"] > max(
        summary["by_bank"][arm]["propagation_passing_dev_and_test"] for arm in ("pca", "random"))
    if summary.get("jacobian_more_validated_consequences_than_both_controls") is not more:
        raise ValueError("Descriptive bank comparison disagrees with counts")


def _validated(results_path, report_path):
    results_path, report_path = Path(results_path).resolve(), Path(report_path).resolve()
    if not report_path.is_file() or not report_path.stat().st_size:
        raise ValueError("An existing, nonempty final report is required")
    root = results_path.parent
    campaign = _read(results_path)
    plan_path = _inside(root, "plan.json")
    plan = _read(plan_path)
    if campaign.get("result_path") != "analysis/result.json":
        raise ValueError("Expected the frozen analysis/result.json artifact")
    ledger = {str(p): _hash(p) for p in (results_path, report_path, plan_path)}
    direct_path = _check_artifact(root, campaign["result_path"], campaign.get("result_sha256"), ledger)
    result = _read(direct_path)
    for value in (campaign, plan, result):
        if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
                or value.get("study_id") != HYPOTHESIS_ID or value.get("conformance") != "EXPLORATORY"):
            raise ValueError("Unsupported schema/study or non-exploratory campaign")
        _no_positive_claims(value)
    for value in (campaign, result):
        if (value.get("status") != "completed" or value.get("exact_neural_claim") is not False
                or value.get("compression_tested") is not False):
            raise ValueError("Only completed studies with explicit claim limits can be registered")
    expected_completion = {"requested": 1, "completed": 1, "failed": 0, "retried": 0, "reused": 0}
    if any(type(campaign.get(k)) is not int or campaign[k] != v for k, v in expected_completion.items()):
        raise ValueError("Exactly one completed direct study and zero failures/retries/reuse are required")
    fingerprint = _digest({k: v for k, v in plan.items() if k != "scientific_fingerprint"})
    if any(value.get("scientific_fingerprint") != fingerprint for value in (campaign, plan, result)):
        raise ValueError("Campaign/result/plan scientific fingerprints disagree")
    if campaign.get("config") != plan.get("config") or result.get("config") != plan.get("config"):
        raise ValueError("Campaign/result configuration differs from frozen plan")
    config = plan["config"]
    producers = plan.get("producer_sha256")
    if not isinstance(producers, dict) or not producers:
        raise ValueError("Frozen producer hashes are required")
    for name, expected in producers.items():
        _check_artifact(root / "producer", name, expected, ledger)
    tasks_path = _inside(root, "tasks.json")
    tasks = _read(tasks_path)
    if _digest(tasks) != plan.get("dataset_digest") or tasks.get("manifest") != plan.get("dataset_manifest"):
        raise ValueError("Frozen dataset/manifest fingerprint mismatch")
    ledger[str(tasks_path)] = _hash(tasks_path)
    calibration_path = _check_artifact(root, "calibration.json", plan.get("calibration_sha256"), ledger)
    calibration = _read(calibration_path)
    if not isinstance(calibration.get("rows"), list) or not calibration["rows"]:
        raise ValueError("Nonempty frozen lens calibration rows are required")
    for split in ("train", "development", "evaluation", "shakedown"):
        count = config.get(split + "_count")
        if (not _nonnegative_int(count) or count == 0
                or len(tasks.get("splits", {}).get(split, [])) != count
                or plan["dataset_manifest"].get("counts", {}).get(split) != count):
            raise ValueError("Frozen task split counts disagree")
    sites, case_ids = config.get("sites"), plan.get("intervention_case_ids")
    if (not isinstance(sites, list) or not sites or len(set(sites)) != len(sites)
            or any(not _nonnegative_int(site) for site in sites)
            or not isinstance(case_ids, list) or not case_ids or len(set(case_ids)) != len(case_ids)
            or not set(case_ids) <= {row["case_id"] for row in tasks["splits"]["evaluation"]}):
        raise ValueError("Invalid frozen intervention schedule")
    capture_path = _check_artifact(root, "capture_manifest.json", result.get("capture_manifest_sha256"), ledger)
    capture = _read(capture_path)
    if (capture.get("status") != "completed" or capture.get("study_id") != HYPOTHESIS_ID
            or type(capture.get("schema_version")) is not int or capture["schema_version"] != 1
            or capture.get("scientific_fingerprint") != fingerprint):
        raise ValueError("Capture is incomplete or from another study")
    counts = {split: config[split + "_count"] for split in ("train", "development", "evaluation", "shakedown")}
    counts.update(lens_fit=len(calibration["rows"]), interventions=len(case_ids) * len(sites) * len(ARMS))
    if (capture.get("counts") != counts or campaign.get("capture_counts") != counts
            or any(type(v) is not int for v in capture["counts"].values())
            or any(type(v) is not int for v in campaign["capture_counts"].values())):
        raise ValueError("Capture counts differ from frozen schedule")
    required = {f"capture/{split}/{name}" for split in ("train", "development", "evaluation", "interventions")
                for name in ("hidden.pt", "records.json")}
    required |= {"capture/banks.pt", "capture/lens.pt", "capture/lease.json", "shakedown/result.json"}
    artifacts = capture.get("artifacts")
    if not isinstance(artifacts, dict) or not required <= set(artifacts):
        raise ValueError("Capture artifact manifest is incomplete")
    for name, expected in artifacts.items():
        _check_artifact(root, name, expected, ledger)
    frozen_path = _check_artifact(root, "analysis/frozen_models.json", result.get("frozen_models_sha256"), ledger)
    if campaign.get("frozen_models_sha256") != result["frozen_models_sha256"]:
        raise ValueError("Campaign and direct frozen-model hashes disagree")
    frozen = _read(frozen_path)
    _no_positive_claims(frozen)
    if (frozen.get("status") != "frozen_before_evaluation" or frozen.get("content_sha256") != _frozen_digest(frozen)
            or frozen.get("exact_neural_claim") is not False):
        raise ValueError("Relation fitting bundle is not valid frozen exploratory evidence")
    if result.get("relation_results", {}).get("frozen_content_sha256") != frozen["content_sha256"]:
        raise ValueError("Direct relation report used a different frozen fitting bundle")
    for key in ("task_metrics", "summary"):
        if campaign.get(key) != result.get(key):
            raise ValueError(f"Campaign differs from direct analysis: {key}")
    _validate_task_metrics(result.get("task_metrics"), config, len(case_ids))
    _validate_summary(result.get("summary"), result.get("relation_results", {}))
    not_claimed = plan.get("not_claimed")
    if (not isinstance(not_claimed, list) or not all(isinstance(v, str) for v in not_claimed)
            or not {"exact_neural_ideal", "compression", "full_J_space", "independent_model_replication"} <= set(not_claimed)):
        raise ValueError("Frozen method boundaries are missing")
    model = plan.get("model", {})
    if (not isinstance(model.get("path"), str) or not model["path"]
            or not isinstance(model.get("revision"), str) or not model["revision"]
            or not isinstance(model.get("files"), dict) or not model["files"]
            or not all(_sha(value) for value in model["files"].values())
            or not _nonnegative_int(config.get("hidden_width")) or not config["hidden_width"]):
        raise ValueError("Frozen model identity and weight-file fingerprints are required")
    for stamp in (plan.get("prepared_at"), result.get("completed_at")):
        if not isinstance(stamp, str):
            raise ValueError("Preparation and analysis completion timestamps are required")
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    elapsed = result.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("A measured nonnegative analysis duration is required")
    return campaign, plan, result, ledger


def build_jspace_relation_meta_hypothesis(results_path, report_path):
    """Build the registration record without importing/opening any database."""
    campaign, plan, result, ledger = _validated(results_path, report_path)
    model = plan["model"]
    model_id = next((part[len("models--"):].replace("--", "/")
                     for part in Path(model["path"]).parts if part.startswith("models--")), model["path"])
    identity = {"model_id": model_id, "checkpoint": model["path"], "revision": model["revision"],
                "file_sha256": model["files"], "hidden_width": plan["config"]["hidden_width"],
                "depth": plan["config"].get("model_depth")}
    fingerprint = plan["scientific_fingerprint"]
    experiment_id = f"jspace-relation-{fingerprint[:16]}-{ledger[str(Path(results_path).resolve())][:16]}"
    classification = result["summary"]["primary_classification"]
    direct_path = str((Path(results_path).resolve().parent / "analysis/result.json"))
    return {
        "schema_version": "nal.meta-hypothesis.v1", "conformance": "EXPLORATORY",
        "hypothesis": {
            "id": HYPOTHESIS_ID, "name": "J-derived concept relation propagation through an intact model",
            "category": "representation", "tested": True, "confirmed": False,
            "description": "One frozen-model study of continuous concept-score relations across layers, with held-out templates and control banks.",
            "question": "Which approximate polynomial relations among J-derived observables persist or transform across neural layers?",
            "prediction": "Some development-selected surrogate consequences survive held-out evaluation; J/control differences remain descriptive.",
            "created_at": plan["prepared_at"], "confirmation_status": "exploratory_" + classification,
            "evidence_count": 1, "direct_experiment_count": 1,
            "tested_scope": "One checkpoint, one task/data seed; layers, banks, polynomials and cases are not independent model replications.",
            "explicitly_not_tested": plan["not_claimed"],
        },
        "result": {
            "confirmed": False, "confirmation_reason": "Exploratory single-study evidence; no confirmatory replication or exact neural proof.",
            "classification": classification, "num_direct_experiments": 1, "model": identity,
            "summary": result["summary"], "task_metrics": result["task_metrics"],
            "descriptive_metrics": {"analysis_elapsed_seconds": result["elapsed_seconds"],
                                    "capture_counts": campaign["capture_counts"]},
            "not_claimed": plan["not_claimed"], "limitations": campaign.get("limitations", []),
            "exact_neural_claim": False, "compression_tested": False, "timing_is_benchmark": False,
            "model_training_performed": False, "surrogate_fitting_performed": True,
            "primary_metric_scope": "jacobian propagation_passing_dev_and_test count; not accuracy or independent evidence count",
            "accuracy_scope": "natural evaluation task original_correct / rows; separate from relation/intervention outcomes",
            "interpretation_boundary": INTERPRETATION_BOUNDARY,
            "completed_at": result["completed_at"], "timestamp_source": "analysis/result.json.completed_at",
        },
        "evidence": {"direct_tests": [{"experiment_id": experiment_id,
            "scientific_fingerprint": fingerprint, "model": identity,
            "result_path": direct_path, "result_sha256": ledger[direct_path],
            "classification": classification, "summary": result["summary"]}]},
        "source_artifacts": sorted(ledger), "source_sha256": dict(sorted(ledger.items())),
    }


def build_jspace_relation_experiment_results(record):
    from neural_architecture_lab.core import ExperimentResult

    result = record["result"]
    metrics = {f"{arm}_{key}": float(values[key]) for arm, values in result["summary"]["by_bank"].items()
               for key in COUNT_KEYS}
    natural = result["task_metrics"]["evaluation"][0]
    metrics.update(accuracy=float(natural["original_accuracy"]),
                   evaluation_task_correct=float(natural["original_correct"]),
                   evaluation_task_total=float(natural["rows"]),
                   analysis_elapsed_seconds=float(result["descriptive_metrics"]["analysis_elapsed_seconds"]))
    for group in result["task_metrics"]["interventions"]:
        prefix = f"intervention_{group['arm']}_site_{group['source_site']}"
        metrics[prefix + "_counterfactual_accuracy"] = float(group["counterfactual_accuracy"])
        metrics[prefix + "_original_accuracy"] = float(group["original_accuracy"])
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite experiment metrics")
    model = result["model"]
    return [ExperimentResult(
        experiment_id=record["evidence"]["direct_tests"][0]["experiment_id"], hypothesis_id=HYPOTHESIS_ID,
        metrics=metrics, primary_metric=metrics["jacobian_propagation_passing_dev_and_test"],
        model_architecture=[model["hidden_width"]], model_parameters=0, training_time=0.0,
        model_checkpoint=model["checkpoint"],
        timestamp=datetime.fromisoformat(result["completed_at"].replace("Z", "+00:00")),
        observations=["EXPLORATORY; confirmed=false; one direct experiment.",
                      "model_parameters=0 is an unmeasured sentinel; the neural model was not trained; polynomial surrogates were fitted.",
                      "accuracy scope: " + result["accuracy_scope"],
                      "primary metric: " + result["primary_metric_scope"], INTERPRETATION_BOUNDARY,
                      json.dumps({"classification": result["classification"], "model": model,
                                  "not_claimed": result["not_claimed"],
                                  "source_artifacts": record["source_artifacts"]}, sort_keys=True)],
    )]


def _make_logger(chromadb_path):
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger

    base = Path(chromadb_path).parent
    base.mkdir(parents=True, exist_ok=True)
    return StandardizedLogger(LoggingConfig(
        project_name="structure_net_meta_hypotheses", enable_wandb=False, auto_upload=False,
        enable_chromadb=True, chromadb_path=str(chromadb_path),
        queue_dir=str(base / "experiment_queue"), sent_dir=str(base / "experiment_sent"),
        rejected_dir=str(base / "experiment_rejected"),
    ))


def store_jspace_relation_meta_hypothesis(results_path, report_path, output_path, *, chromadb_path=Path("data/chroma_db")):
    """Explicit registration; a success receipt requires database metadata readback."""
    record = build_jspace_relation_meta_hypothesis(results_path, report_path)
    experiments = build_jspace_relation_experiment_results(record)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        try:
            logger = _make_logger(chromadb_path)
            if logger.hypotheses_collection is None or logger.experiments_collection is None:
                raise RuntimeError("ChromaDB collections unavailable")
            logger.log_hypothesis(record["hypothesis"])
            hashes = [logger.log_experiment_result(item) for item in experiments]
            hyp = logger.hypotheses_collection.get(ids=[HYPOTHESIS_ID], include=["metadatas"])
            exp = logger.experiments_collection.get(ids=hashes, include=["metadatas"])
            hm, em = hyp.get("metadatas", []), exp.get("metadatas", [])
            experiment = experiments[0]
            if (hyp.get("ids") != [HYPOTHESIS_ID] or len(hm) != 1
                    or hm[0].get("id") != HYPOTHESIS_ID or hm[0].get("confirmed") is not False
                    or type(hm[0].get("evidence_count")) is not int or hm[0]["evidence_count"] != 1
                    or type(hm[0].get("direct_experiment_count")) is not int or hm[0]["direct_experiment_count"] != 1
                    or hm[0].get("confirmation_status") != record["hypothesis"]["confirmation_status"]
                    or exp.get("ids") != hashes or len(em) != 1
                    or em[0].get("experiment_id") != experiment.experiment_id
                    or em[0].get("hypothesis_id") != HYPOTHESIS_ID
                    or em[0].get("accuracy") != experiment.metrics["accuracy"]
                    or em[0].get("primary_metric") != experiment.primary_metric):
                raise RuntimeError("J-space relation ChromaDB readback failed")
            record["storage"] = {
                "aggregate_path": str(output_path), "chromadb_path": str(chromadb_path),
                "experiment_ids": [experiment.experiment_id], "result_hashes": hashes,
                "readback": {"verified": True, "hypothesis_id": HYPOTHESIS_ID, "confirmed": False,
                             "evidence_count": 1, "experiment_id": experiment.experiment_id,
                             "accuracy": experiment.metrics["accuracy"], "primary_metric": experiment.primary_metric},
            }
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            output_path.unlink()
            raise
    return record
