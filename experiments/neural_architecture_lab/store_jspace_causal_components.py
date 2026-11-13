"""Hash-check and register a completed causal-component pilot, including stops."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from neural_architecture_lab.core import ExperimentResult
from neural_architecture_lab.jspace_relation_meta_hypothesis import _make_logger
from experiments.structure_net.jspace_causal_components import STUDY_ID, verify, verify_manifest
from experiments.structure_net.jspace_relation_study import ROOT, file_hash, write_json


def store(root, report, output, chromadb):
    if output.exists():
        raise FileExistsError(output)
    plan = verify(root)
    aggregate = json.loads((root/"campaign_results.json").read_text())
    result = json.loads((root/"result.json").read_text())
    if (aggregate["study_id"] != STUDY_ID or aggregate["status"] != "completed"
            or aggregate["confirmed"] is not False or aggregate["scientific_fingerprint"] != plan["scientific_fingerprint"]
            or aggregate["result_sha256"] != file_hash(root/"result.json")
            or {k: v for k, v in aggregate.items() if k not in ("result_sha256", "result_path")} != result):
        raise ValueError("Aggregate/direct result mismatch")
    for path, expected in result["source_sha256"].items():
        if file_hash(root/path) != expected:
            raise ValueError("Changed result input")
    verify_manifest(root, root/"training/manifest.json")
    if "interventions" in result:
        verify_manifest(root, root/"interventions/manifest.json")
    measured = result["primary_metric_status"] == "measured"
    stage = result.get("interventions", {})
    competence = stage.get("evaluation_competence", result["training"]["development_competence"])
    accuracy = result["primary_metric"] if measured else competence["source_accuracy"]
    scope = result["accuracy_scope"] if measured else (
        "evaluation_clean_source_accuracy_causal_endpoint_not_measured" if "evaluation_competence" in stage
        else "development_clean_source_accuracy_causal_endpoint_not_measured")
    hypothesis = {"id": STUDY_ID, "name": "Causal effects of learned country vectors and sparse J components",
                  "category": "representation", "tested": True, "confirmed": False,
                  "created_at": datetime.now(timezone.utc).isoformat(), "evidence_count": 1, "direct_experiment_count": 1,
                  "confirmation_status": result["classification"], "tested_scope": scope,
                  "description": "Prospective single-checkpoint Qwen3B pilot on new named worlds. Tests full, J-component and remainder edits with matched controls, competence and dense-donor gates. No compression claim."}
    metrics = {"accuracy": accuracy, "clean_source_accuracy": competence["source_accuracy"],
               "clean_donor_accuracy": competence["donor_accuracy"], "clean_joint_fraction": competence["joint_fraction"]}
    if measured:
        metrics.update({arm+"_counterfactual_success": row["target_success"] for arm, row in result["primary_metrics"].items()})
    experiment = ExperimentResult(
        experiment_id="jspace-causal-components-"+plan["scientific_fingerprint"][:16], hypothesis_id=STUDY_ID,
        metrics=metrics, primary_metric=accuracy, model_architecture=[2048]*36, model_parameters=0,
        training_time=result["training"]["elapsed_seconds"]+stage.get("elapsed_seconds", 0),
        model_checkpoint=plan["model"]["path"],
        observations=["Accuracy scope: "+scope, result["classification"], str(report),
                      "model_parameters=0 is an unmeasured sentinel. The transformer was frozen; only concept means were fitted.",
                      "Time is GPU-stage wall time, not model training or a throughput benchmark.",
                      "A stopped study retains null for its causal endpoint in the aggregate; its searchable stage result reports measured clean competence.",
                      "Single-checkpoint pilot, unconfirmed; no lossless compression or Gröbner propagation claim."],
    )
    logger = _make_logger(chromadb)
    if logger.hypotheses_collection is None or logger.experiments_collection is None:
        raise RuntimeError("Registry unavailable")
    logger.log_hypothesis(hypothesis)
    result_hash = logger.log_experiment_result(experiment)
    logger.experiments_collection.update(ids=[result_hash], metadatas=[{
        "accuracy_scope": scope, "causal_endpoint_measured": measured, "confirmed": False,
        "confirmation_status": result["classification"]}],
        documents=[f"{STUDY_ID}: {result['classification']}. Measured accuracy {accuracy:.8f}; scope {scope}. "
                   f"Causal endpoint measured: {measured}. Single-checkpoint pilot; no compression claim."])
    hypothesis_readback = logger.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas"])
    experiment_readback = logger.experiments_collection.get(ids=[result_hash], include=["metadatas"])
    hm, em = hypothesis_readback["metadatas"][0], experiment_readback["metadatas"][0]
    if (hypothesis_readback["ids"] != [STUDY_ID] or hm["tested"] is not True or hm["confirmed"] is not False
            or hm["confirmation_status"] != result["classification"] or experiment_readback["ids"] != [result_hash]
            or em["hypothesis_id"] != STUDY_ID or em["accuracy"] != accuracy or em["primary_metric"] != accuracy
            or em["accuracy_scope"] != scope or em["causal_endpoint_measured"] is not measured):
        raise RuntimeError("Registry readback mismatch")
    sources = [root/"plan.json", root/"result.json", root/"campaign_results.json", report, Path(__file__)]
    record = {"schema_version": "nal.meta-hypothesis.v1", "hypothesis": hypothesis, "result": result,
              "source_sha256": {str(path): file_hash(path) for path in sources},
              "storage": {"experiment_id": experiment.experiment_id, "result_hash": result_hash,
                          "readback_verified": True, "accuracy_scope": scope, "causal_endpoint_measured": measured}}
    write_json(output, record)
    return record["storage"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT/"data/experiments/jspace_causal_components/20261113_v1")
    parser.add_argument("--report", type=Path, default=ROOT/"docs/08 - Analysis/jspace/2026-11-13_jspace-causal-components.md")
    parser.add_argument("--output", type=Path, default=ROOT/"data/hypothesis_registry/jspace_causal_components_20261113.json")
    parser.add_argument("--chromadb", type=Path, default=ROOT/"data/chroma_db")
    args = parser.parse_args()
    print(json.dumps(store(args.root, args.report, args.output, args.chromadb)))
