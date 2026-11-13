"""Register the bounded donor-localization pilot with explicit endpoint scopes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from neural_architecture_lab.core import ExperimentResult
from neural_architecture_lab.jspace_relation_meta_hypothesis import _make_logger
from experiments.structure_net.jspace_donor_localization import STUDY_ID, verify
from experiments.structure_net.jspace_relation_study import ROOT, file_hash, write_json


def store(root, report, output, chromadb):
    if output.exists():
        raise FileExistsError(output)
    plan = verify(root)
    result = json.loads((root/"result.json").read_text())
    campaign = json.loads((root/"campaign_results.json").read_text())
    if (result["study_id"] != STUDY_ID or result["status"] != "completed" or result["confirmed"] is not False
            or result["scientific_fingerprint"] != plan["scientific_fingerprint"]
            or campaign["result_sha256"] != file_hash(root/"result.json")
            or {k:v for k,v in campaign.items() if k not in ("result_path", "result_sha256")} != result
            or file_hash(root/"capture_manifest.json") != result["capture_manifest_sha256"]):
        raise ValueError("Result/aggregate integrity mismatch")
    for path, expected in result["capture"]["artifacts"].items():
        if file_hash(root/path) != expected:
            raise ValueError("Capture artifact changed")
    measured = result["primary_metric_status"] == "measured"
    gate = result["capture"].get("competence")
    scope = ("block24_query_except_final_counterfactual_answer_accuracy_on_clean_correct_pairs" if measured else
             "clean_source_answer_accuracy_localization_endpoint_not_measured" if gate else "systems_checks_only_no_answer_accuracy")
    hypothesis = {"id": STUDY_ID, "name": "Dense donor causal localization across token positions and layers",
                  "category": "representation", "tested": True, "confirmed": False, "evidence_count": 1,
                  "direct_experiment_count": int(gate is not None), "created_at": datetime.now(timezone.utc).isoformat(),
                  "confirmation_status": result["classification"], "tested_scope": scope,
                  "description": "Fresh matched-length Qwen3B donor/source pairs; exact terminal logits, prefix and copy controls; seven-layer/three-mask sweep. One checkpoint, no J-space sufficiency or compression claim."}
    logger = _make_logger(chromadb)
    if logger.hypotheses_collection is None or logger.experiments_collection is None:
        raise RuntimeError("Registry unavailable")
    logger.log_hypothesis(hypothesis)
    storage = {"readback_verified": False, "accuracy_scope": scope, "localization_endpoint_measured": measured}
    if gate is not None:
        accuracy = result["metrics"]["24"]["query_except_final"]["target_success"] if measured else gate["source_accuracy"]
        primary = result["primary_metric"] if measured else accuracy
        experiment = ExperimentResult(
            experiment_id="jspace-donor-localization-"+plan["scientific_fingerprint"][:16], hypothesis_id=STUDY_ID,
            metrics={"accuracy": accuracy, "clean_source_accuracy": gate["source_accuracy"], "clean_donor_accuracy": gate["donor_accuracy"]},
            primary_metric=primary, model_architecture=[2048]*36, model_parameters=0,
            model_checkpoint=plan["model"]["path"], training_time=result["capture"]["elapsed_seconds"],
            observations=[scope, result["primary_metric_scope"], result["classification"], str(report),
                          "Frozen transformer; model_parameters=0 is an unmeasured sentinel, not a parameter count.",
                          "GPU-stage time is not training or a throughput benchmark. Single-checkpoint pilot; no compression claim."])
        result_hash = logger.log_experiment_result(experiment)
        logger.experiments_collection.update(ids=[result_hash], metadatas=[{
            "accuracy_scope": scope, "localization_endpoint_measured": measured, "confirmed": False,
            "confirmation_status": result["classification"]}],
            documents=[f"{STUDY_ID}: {result['classification']}; accuracy {accuracy:.8f}, scope {scope}; "
                       f"primary comparison {primary:.8f}. Frozen-model donor interventions; no J-space or compression proof."])
        back = logger.experiments_collection.get(ids=[result_hash], include=["metadatas"])
        metadata = back["metadatas"][0]
        if (back["ids"] != [result_hash] or metadata["hypothesis_id"] != STUDY_ID or metadata["accuracy"] != accuracy
                or metadata["primary_metric"] != primary or metadata["accuracy_scope"] != scope
                or metadata["localization_endpoint_measured"] is not measured):
            raise RuntimeError("Experiment readback failed")
        storage.update({"experiment_id": experiment.experiment_id, "result_hash": result_hash})
    back = logger.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas"])
    metadata = back["metadatas"][0]
    if back["ids"] != [STUDY_ID] or metadata["confirmed"] is not False or metadata["confirmation_status"] != result["classification"]:
        raise RuntimeError("Hypothesis readback failed")
    storage["readback_verified"] = True
    paths = [root/"plan.json", root/"result.json", root/"campaign_results.json", report, Path(__file__)]
    write_json(output, {"schema_version": "nal.meta-hypothesis.v1", "hypothesis": hypothesis, "result": result,
                        "storage": storage, "source_sha256": {str(p): file_hash(p) for p in paths}})
    return storage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT/"data/experiments/jspace_donor_localization/20261113_v1")
    parser.add_argument("--report", type=Path, default=ROOT/"docs/08 - Analysis/jspace/2026-11-13_jspace-donor-localization.md")
    parser.add_argument("--output", type=Path, default=ROOT/"data/hypothesis_registry/jspace_donor_localization_20261113.json")
    parser.add_argument("--chromadb", type=Path, default=ROOT/"data/chroma_db")
    args = parser.parse_args()
    print(json.dumps(store(args.root, args.report, args.output, args.chromadb)))
