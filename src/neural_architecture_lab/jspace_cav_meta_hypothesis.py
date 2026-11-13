"""Register one exploratory J-space pilot, with verified artifact and DB readback."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

HYPOTHESIS_ID = "JSPACE-CAV-PILOT-01"
LOCAL_CONTEXT_BOUNDARY = (
    "Sufficiency interventions alter only the final prompt position; other token "
    "states and context remain intact. This is not autonomous reduced-state "
    "closure, whole-attention compression, or a speed benchmark."
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    json.dumps(value, allow_nan=False)
    return value


def _validated(results_path, report_path):
    results_path, report_path = Path(results_path), Path(report_path)
    if not report_path.is_file():
        raise ValueError("An existing final report is required")
    campaign = _read(results_path)
    config, result = campaign.get("config", {}), campaign.get("result", {})
    for item in (campaign, config, result):
        if type(item.get("schema_version")) is not int or item["schema_version"] != 1:
            raise ValueError("Unsupported J-space schema")
        if item.get("hypothesis_id") != HYPOTHESIS_ID:
            raise ValueError("Unexpected J-space hypothesis")
    counts = campaign.get("completion_counts", {})
    if (counts != {"requested": 1, "completed": 1, "failed": 0, "reused": 0}
            or any(type(v) is not int for v in counts.values())):
        raise ValueError("Exactly one completed direct pilot and zero failures are required")
    if (result.get("status") != "completed" or result.get("confirmed") is not False
            or result.get("conformance") != "EXPLORATORY" or config.get("conformance") != "EXPLORATORY"):
        raise ValueError("Only completed, explicitly exploratory/unconfirmed pilots can be registered")
    fingerprint = _digest({k: v for k, v in config.items() if k != "scientific_fingerprint"})
    if any(item.get("scientific_fingerprint") != fingerprint for item in (campaign, config, result)):
        raise ValueError("Campaign/config/result scientific fingerprints disagree")
    root = results_path.parent
    plan_path = root / "plan.json"
    if _read(plan_path) != config:
        raise ValueError("Frozen plan differs from campaign config")
    run_path = Path(campaign["artifacts"]["run"])
    if not run_path.is_absolute() and not run_path.is_dir():
        run_path = root / run_path
    direct_path = run_path / "result.json"
    if _read(direct_path) != result:
        raise ValueError("Direct run result differs from the campaign result")
    sources = config.get("producer_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Frozen producer hashes are required")
    archive = (root / "producer").resolve()
    for name, expected in sources.items():
        path = (archive / name).resolve()
        if not path.is_relative_to(archive) or not path.is_file() or _hash(path) != expected:
            raise ValueError(f"Frozen producer fingerprint mismatch: {name}")
    for name, key in (("tasks.json", "data_sha256"), ("calibration.json", "calibration_sha256")):
        if _hash(root / name) != config.get(key):
            raise ValueError(f"Frozen input fingerprint mismatch: {name}")
    if result.get("not_claimed") != config.get("not_claimed") or not isinstance(result.get("not_claimed"), list):
        raise ValueError("Method boundaries differ from the frozen plan")
    if not isinstance(result.get("classification"), str) or not result["classification"]:
        raise ValueError("A completed classification is required")
    if any(type(v) is not bool for v in result.get("gates", {}).values()):
        raise ValueError("Gate flags must be booleans")
    return campaign, direct_path, plan_path


def _model_identity(spec):
    if spec is None:
        return None
    model_id = next((part[len("models--"):].replace("--", "/")
                     for part in Path(spec["path"]).parts if part.startswith("models--")), spec["path"])
    return {"model_id": model_id, "checkpoint": spec["path"],
            "revision": spec["revision"], "file_sha256": spec["files"], "hidden_width": spec["width"]}


def build_jspace_cav_meta_hypothesis(results_path, report_path):
    results_path, report_path = Path(results_path), Path(report_path)
    campaign, direct_path, plan_path = _validated(results_path, report_path)
    config, result = campaign["config"], campaign["result"]
    chosen = result.get("selection", {}).get("model")
    matches = [spec for spec in config["models"] if spec["path"] == chosen]
    if chosen is not None and len(matches) != 1:
        raise ValueError("Selected model is absent or ambiguous in the frozen plan")
    model = _model_identity(matches[0] if matches else None)
    experiment_id = f"jspace-cav-pilot-{config['scientific_fingerprint'][:16]}-{_hash(results_path)[:16]}"
    descriptive = {key: result[key] for key in (
        "gates", "selection", "competence_screens", "evaluation_parent_correct", "functional",
        "paired_differences", "sufficiency", "retention_differences", "bank_metadata",
        "causal_test", "sufficiency_test", "next_experiment", "elapsed_seconds",
        "peak_allocated_mib", "peak_reserved_mib", "autonomous_reduced_transition_established",
    ) if key in result}
    artifacts = [results_path, report_path, plan_path, direct_path]
    return {
        "schema_version": "nal.meta-hypothesis.v1", "conformance": "EXPLORATORY",
        "hypothesis": {
            "id": HYPOTHESIS_ID, "name": "J-derived CAV causal and local sufficiency pilot",
            "category": "representation", "tested": True, "confirmed": False,
            "description": "One frozen-model pilot of J-derived concept redirection and retained-coordinate interventions.",
            "question": "Do J directions outperform controls causally and retain task behavior under local complement replacement?",
            "prediction": "The recorded parent competence, causal, local sufficiency, and specificity gates all pass.",
            "created_at": config["prepared_at"],
            "confirmation_status": "exploratory_" + result["classification"],
            "evidence_count": 1, "direct_experiment_count": 1,
            "tested_scope": "One selected frozen model and split; cases and control arms are not independent models.",
            "explicitly_not_tested": result["not_claimed"],
        },
        "result": {
            "confirmed": False, "confirmation_reason": "Exploratory single-pilot evidence; no confirmatory replication.",
            "classification": result["classification"], "num_direct_experiments": 1,
            "model": model, "screened_models": [_model_identity(s) for s in config["models"]
                                                   if any(c["model"] == s["path"] for c in result["competence_screens"])],
            "descriptive_metrics": descriptive, "not_claimed": result["not_claimed"],
            "accuracy_scope": ("evaluation_counterfactual_answer" if "functional" in result
                               else "last_development_competence_screen; evaluation_not_run"),
            "local_context_boundary": LOCAL_CONTEXT_BOUNDARY,
            "timing_is_benchmark": False, "training_performed": False,
            "timestamp_source": "config.prepared_at; completion wall-clock time was not recorded",
        },
        "evidence": {"direct_tests": [{"experiment_id": experiment_id,
            "scientific_fingerprint": config["scientific_fingerprint"], "model": model,
            "result_path": str(direct_path), "result_sha256": _hash(direct_path),
            "gates": result.get("gates", {}), "classification": result["classification"]}]},
        "source_artifacts": [str(p) for p in artifacts],
        "source_sha256": {str(p): _hash(p) for p in artifacts},
    }


def build_jspace_cav_experiment_results(record):
    from neural_architecture_lab.core import ExperimentResult

    result = record["result"]
    descriptive = result["descriptive_metrics"]
    gates = descriptive.get("gates", {})
    metrics = {"gate_" + name: float(value) for name, value in gates.items()}
    metrics["elapsed_seconds"] = float(descriptive.get("elapsed_seconds", 0))
    if "functional" in descriptive:
        jacobian = descriptive["functional"]["jacobian"]
        metrics.update(accuracy=float(jacobian["raw_accuracy"]["accuracy"]),
                       parent_accuracy=float(jacobian["baseline_accuracy"]["accuracy"]))
        conditional = jacobian["parent_correct_accuracy"]["accuracy"]
        if conditional is not None:
            metrics["parent_conditioned_causal_accuracy"] = float(conditional)
    else:
        # StandardizedLogger defaults an absent accuracy to zero. Supply the actual
        # development screen with an explicit scope instead of inventing evaluation.
        screens = descriptive.get("competence_screens", [])
        if not screens or not 0 <= screens[-1]["correct"] <= screens[-1]["total"] or screens[-1]["total"] <= 0:
            raise ValueError("A measured competence screen is required when evaluation did not run")
        metrics["accuracy"] = screens[-1]["correct"] / screens[-1]["total"]
        metrics["development_screen_accuracy"] = metrics["accuracy"]
    metrics["evaluation_accuracy_measured"] = float("functional" in descriptive)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite experiment metrics")
    model = result["model"]
    return [ExperimentResult(
        experiment_id=record["evidence"]["direct_tests"][0]["experiment_id"],
        hypothesis_id=HYPOTHESIS_ID, metrics=metrics,
        primary_metric=float(bool(gates) and all(gates.values())),
        model_architecture=[model["hidden_width"]] if model else [],
        model_parameters=0, training_time=0.0,
        model_checkpoint=model["checkpoint"] if model else None,
        timestamp=datetime.fromisoformat(record["hypothesis"]["created_at"].replace("Z", "+00:00")),
        observations=["EXPLORATORY; confirmed=false; one direct experiment.",
                      "model_parameters=0 is an unmeasured sentinel; no model training occurred.",
                      "accuracy scope: " + result["accuracy_scope"],
                      result["timestamp_source"], LOCAL_CONTEXT_BOUNDARY,
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


def store_jspace_cav_meta_hypothesis(results_path, report_path, output_path, *, chromadb_path=Path("data/chroma_db")):
    record = build_jspace_cav_meta_hypothesis(results_path, report_path)
    experiments = build_jspace_cav_experiment_results(record)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve exclusively before any DB mutation; failed readback leaves no success receipt.
    with output_path.open("x") as stream:
        try:
            logger = _make_logger(chromadb_path)
            if logger.hypotheses_collection is None or logger.experiments_collection is None:
                raise RuntimeError("ChromaDB collections unavailable")
            logger.log_hypothesis(record["hypothesis"])
            hashes = [logger.log_experiment_result(item) for item in experiments]
            hyp = logger.hypotheses_collection.get(ids=[HYPOTHESIS_ID], include=["metadatas"])
            exp = logger.experiments_collection.get(ids=hashes, include=["metadatas"])
            hmeta = hyp.get("metadatas", [])
            emeta = exp.get("metadatas", [])
            experiment_id = experiments[0].experiment_id
            if (hyp.get("ids") != [HYPOTHESIS_ID] or len(hmeta) != 1
                    or hmeta[0].get("id") != HYPOTHESIS_ID or hmeta[0].get("confirmed") is not False
                    or hmeta[0].get("evidence_count") != 1 or hmeta[0].get("direct_experiment_count") != 1
                    or exp.get("ids") != hashes or len(emeta) != 1
                    or emeta[0].get("experiment_id") != experiment_id
                    or emeta[0].get("hypothesis_id") != HYPOTHESIS_ID):
                raise RuntimeError("J-space ChromaDB readback failed")
            record["storage"] = {
                "aggregate_path": str(output_path), "chromadb_path": str(chromadb_path),
                "experiment_ids": [experiment_id], "result_hashes": hashes,
                "readback": {"verified": True, "hypothesis_id": HYPOTHESIS_ID,
                             "confirmed": False, "evidence_count": 1, "experiment_id": experiment_id},
            }
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            output_path.unlink()
            raise
    return record
