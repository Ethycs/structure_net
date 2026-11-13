"""Register task-head accuracy and causal evidence with separate endpoint scopes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from neural_architecture_lab.core import ExperimentResult
from neural_architecture_lab.jspace_relation_meta_hypothesis import _make_logger
from experiments.structure_net.jspace_active_study import DEFAULT_ROOT, STUDY_ID, verify, checked_manifest
from experiments.structure_net.jspace_relation_study import ROOT, file_hash, write_json


SCOPE = "fresh_four_colour_lookup_accuracy_of_development_selected_head_not_general_LM_accuracy"
PRIMARY_SCOPE = "block29_vanishing_feature_head_minus_MLP_accuracy"
FOLLOWUPS = (ROOT/"data/experiments/jspace_active_closure_audit/20261113_v1",
             ROOT/"data/experiments/jspace_active_latency_audit/20261113_v1")


def checked_result(root):
    plan = verify(root)
    training = checked_manifest(root, "training")
    checked_manifest(root, "evaluation")
    result = json.loads((root/"amended_result.json").read_text())
    aggregate = json.loads((root/"campaign_results_amended.json").read_text())
    if (result["status"] != "completed" or result["study_id"] != STUDY_ID or result["confirmed"] is not False
            or result["base_scientific_fingerprint"] != plan["scientific_fingerprint"]
            or result["base_result_sha256"] != file_hash(root/"result.json")
            or result["fit_sha256"] != file_hash(root/"fit.pt")
            or result["fit_selection_sha256"] != file_hash(root/"fit.json")
            or result["amendment_plan_sha256"] != file_hash(root/"amendment_01_plan.json")
            or aggregate["result_sha256"] != file_hash(root/"amended_result.json")
            or {k: v for k, v in aggregate.items() if k not in ("result_path", "result_sha256")} != result):
        raise ValueError("Amended aggregate/provenance mismatch")
    if result["candidate_quality_gate"]:
        checked_manifest(root, "benchmark")
    candidate = result["benchmark_candidate"]
    accuracy = result["heads"][str(candidate["layer"])][candidate["head"]]["accuracy"]
    primary = result["primary_head_comparison_29_vanishing_minus_mlp"]["difference"]
    return plan, training, result, accuracy, primary


def readback(root, output, chromadb, result_hash=None):
    _, _, result, accuracy, primary = checked_result(root)
    logger = _make_logger(chromadb)
    if result_hash is None:
        result_hash = json.loads(output.read_text())["storage"]["result_hash"]
    back = logger.experiments_collection.get(ids=[result_hash], include=["metadatas", "documents"])
    if back["ids"] != [result_hash]:
        raise RuntimeError("Experiment not found by exact ID")
    m = back["metadatas"][0]
    experiment_id = "jspace-active-cav-heads-"+result["scientific_fingerprint"][:16]
    if (m["hypothesis_id"] != STUDY_ID or m["accuracy"] != accuracy or m["primary_metric"] != primary
            or m["accuracy_scope"] != SCOPE or m["primary_scope"] != PRIMARY_SCOPE
            or m["confirmed"] is not False or m["confirmation_status"] != result["classification"]
            or m["experiment_id"] != experiment_id):
        raise RuntimeError("Experiment scope/readback mismatch")
    h = logger.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas"])
    metadata = h["metadatas"][0] if h["ids"] == [STUDY_ID] else {}
    if (metadata.get("confirmed") is not False or metadata.get("confirmation_status") != result["classification"]
            or metadata.get("evidence_count") != 1 or metadata.get("direct_experiment_count") != 1
            or json.loads(metadata.get("direct_experiment_ids_json", "[]")) != [experiment_id]
            or json.loads(metadata.get("direct_result_hashes_json", "[]")) != [result_hash]):
        raise RuntimeError("Hypothesis readback mismatch")
    return {"readback_verified": True, "hypothesis_id": STUDY_ID, "result_hash": result_hash,
            "accuracy": accuracy, "accuracy_scope": SCOPE, "primary_metric": primary,
            "primary_scope": PRIMARY_SCOPE, "confirmed": False, "classification": result["classification"],
            "experiment_id": experiment_id, "evidence_count": 1, "direct_experiment_count": 1}


def store(root, report, output, chromadb):
    if output.exists():
        raise FileExistsError(output)
    plan, training, result, accuracy, primary = checked_result(root)
    followups = []
    for path in FOLLOWUPS:
        audit = json.loads((path/"result.json").read_text())
        aggregate = json.loads((path/"campaign_results.json").read_text())
        if (audit["status"] != "completed" or audit["confirmed"] is not False
                or aggregate["result_sha256"] != file_hash(path/"result.json")
                or {k: v for k, v in aggregate.items() if k not in ("result_path", "result_sha256")} != audit):
            raise ValueError("Exploratory follow-up aggregate changed")
        for name, expected in audit["artifacts"].items():
            if file_hash(path/name) != expected:
                raise ValueError("Exploratory follow-up artifact changed")
        followups.append({"study_id": audit["study_id"], "scope": "posthoc_reused_evaluation_not_independent_accuracy_replication",
                          "aggregate": str(path/"campaign_results.json"), "result": audit})
    hypothesis = {"id": STUDY_ID, "name": "Active-position country CAVs and polynomial task heads",
                  "category": "representation", "tested": True, "confirmed": False, "evidence_count": 1,
                  "direct_experiment_count": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                  "confirmation_status": result["classification"],
                  "tested_scope": "single_checkpoint_fresh_worlds_causal_components_and_four_colour_task_heads",
                  "description": "Positional CAV and 94-token Jacobian interventions at blocks 24/29; separate five-family task-head comparison. Prospective stricter specificity control. One checkpoint, 64 evaluation worlds; no general LM or exact Gröbner claim."}
    fit = json.loads((root/"fit.json").read_text())
    candidate = result["benchmark_candidate"]
    experiment = ExperimentResult(
        experiment_id="jspace-active-cav-heads-"+result["scientific_fingerprint"][:16], hypothesis_id=STUDY_ID,
        metrics={"accuracy": accuracy, "teacher_accuracy": result["teacher"]["accuracy"], "vanishing_minus_MLP": primary},
        primary_metric=primary, model_architecture=[2048]*36, model_parameters=training["model_parameters"],
        model_checkpoint=plan["model"]["path"], training_time=fit["elapsed_seconds"],
        observations=[SCOPE, PRIMARY_SCOPE, result["classification"], str(report),
                      f"Development-selected candidate: {candidate}.",
                      "Architecture/parameter fields describe the frozen teacher. Duration is CPU head/CAV fitting wall time.",
                      "Country intervention, polynomial-head quality and any task latency benchmark are separate endpoints; confirmed=false."])
    logger = _make_logger(chromadb)
    logger.log_hypothesis(hypothesis)
    result_hash = logger.log_experiment_result(experiment)
    logger.experiments_collection.update(ids=[result_hash], metadatas=[{
        "accuracy_scope": SCOPE, "primary_scope": PRIMARY_SCOPE, "confirmed": False,
        "confirmation_status": result["classification"], "prospective_amendment": 1,
        "parameter_scope": "frozen_teacher_checkpoint", "compression_benchmarked": result["compression_benchmarked"]}],
        documents=[f"{STUDY_ID}: selected specialized lookup head accuracy {accuracy:.8f}; {SCOPE}. "
                   f"Primary head contrast {primary:.8f}; {PRIMARY_SCOPE}. Causal classification {result['classification']}. "
                   "Prospective amendment 01; no general language-model accuracy or exact neural algebra proof."])
    logger.hypotheses_collection.update(ids=[STUDY_ID], metadatas=[{
        "direct_experiment_ids_json": json.dumps([experiment.experiment_id]),
        "direct_result_hashes_json": json.dumps([result_hash]),
        "posthoc_audit_ids_json": json.dumps([value["study_id"] for value in followups]),
        "report_path": str(report), "aggregate_sha256": file_hash(root/"campaign_results_amended.json")}])
    storage = {**readback(root, output, chromadb, result_hash=result_hash), "experiment_id": experiment.experiment_id}
    paths = [root/"amended_result.json", root/"campaign_results_amended.json", root/"amendment_01_plan.json", report, Path(__file__)]
    paths.extend(path/name for path in FOLLOWUPS for name in ("plan.json", "result.json", "campaign_results.json"))
    write_json(output, {"schema_version": "nal.meta-hypothesis.v1", "hypothesis": hypothesis, "result": result,
                        "exploratory_followups": followups,
                        "storage": storage, "source_sha256": {str(p): file_hash(p) for p in paths}})
    return storage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("store", "readback"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=ROOT/"docs/08 - Analysis/jspace/2026-11-13_jspace-active-cav-heads.md")
    parser.add_argument("--output", type=Path, default=ROOT/"data/hypothesis_registry/jspace_active_cav_heads_20261113.json")
    parser.add_argument("--chromadb", type=Path, default=ROOT/"data/chroma_db")
    args = parser.parse_args()
    value = store(args.root, args.report, args.output, args.chromadb) if args.stage == "store" else readback(args.root, args.output, args.chromadb)
    print(json.dumps(value, allow_nan=False))
