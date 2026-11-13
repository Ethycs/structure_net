"""Register the completed exploratory country probe with explicit accuracy scope."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from neural_architecture_lab.core import ExperimentResult
from neural_architecture_lab.jspace_relation_meta_hypothesis import _make_logger
from experiments.structure_net.jspace_country_probe import STUDY_ID
from experiments.structure_net.jspace_relation_study import ROOT, digest, file_hash, write_json


def store(root, report, output, chromadb_path):
    root, report, output = Path(root), Path(report), Path(output)
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((root/"plan.json").read_text())
    result = json.loads((root/"result.json").read_text())
    aggregate = json.loads((root/"campaign_results.json").read_text())
    if digest({k: v for k, v in plan.items() if k != "scientific_fingerprint"}) != plan["scientific_fingerprint"]:
        raise ValueError("Plan fingerprint mismatch")
    for section in ("input_sha256", "producer_sha256"):
        for path, expected in plan[section].items():
            if file_hash(Path(path)) != expected:
                raise ValueError(f"Changed source or input: {path}")
    for path, expected in result["artifacts"].items():
        if file_hash(root/path) != expected:
            raise ValueError(f"Changed probe artifact: {path}")
    if file_hash(root/"selection.json") != result["selection_sha256"]:
        raise ValueError("Selection artifact changed")
    if (aggregate["study_id"] != STUDY_ID or aggregate["status"] != "completed"
            or aggregate["confirmed"] is not False or aggregate["task_accuracy_measured"] is not False
            or aggregate["scientific_fingerprint"] != plan["scientific_fingerprint"]
            or aggregate["scientific_fingerprint"] != result["scientific_fingerprint"]
            or aggregate["result_sha256"] != file_hash(root/"result.json")
            or aggregate["primary_layer_metrics"] != result["layers"][str(aggregate["primary_layer"]) ]):
        raise ValueError("Invalid aggregate or mismatched direct result")
    hypothesis = {"id": STUDY_ID, "name": "Intermediate-country decoding from J-derived directions and remainder",
                  "category": "representation", "tested": True, "confirmed": False,
                  "confirmation_status": aggregate["classification"], "evidence_count": 1,
                  "direct_experiment_count": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                  "description": "Exploratory CPU readouts on reused Qwen3B activations. Accuracy means country-probe accuracy; no model answer, causal, or compression claim.",
                  "tested_scope": aggregate["accuracy_scope"]}
    primary = aggregate["primary_layer_metrics"]
    metrics = {f"{arm}_country_probe_accuracy": row["accuracy"] for arm, row in primary.items()}
    metrics["accuracy"] = primary["J94"]["accuracy"]
    experiment = ExperimentResult(
        experiment_id="jspace-country-probe-"+plan["scientific_fingerprint"][:16],
        hypothesis_id=STUDY_ID, metrics=metrics, primary_metric=aggregate["primary_metric"],
        model_architecture=[94, 4], model_parameters=380, training_time=aggregate["elapsed_seconds"],
        model_checkpoint=str(root/"heads.pt"),
        observations=["Accuracy measures the fitted J94 country probe, NOT Qwen answer accuracy.",
                      "Architecture/parameter count describes that temporary ridge head, not the frozen transformer.",
                      "CPU analysis time; original model was not run or trained. Previously inspected evaluation data.",
                      aggregate["classification"], "Primary metric: "+aggregate["primary_metric_scope"], str(report)],
    )
    logger = _make_logger(Path(chromadb_path))
    if logger.hypotheses_collection is None or logger.experiments_collection is None:
        raise RuntimeError("Registry unavailable")
    logger.log_hypothesis(hypothesis)
    result_hash = logger.log_experiment_result(experiment)
    # The generic logger otherwise labels every accuracy as an unqualified model result.
    logger.experiments_collection.update(ids=[result_hash], metadatas=[{
        "accuracy_scope": aggregate["accuracy_scope"], "task_accuracy_measured": False,
        "confirmation_status": aggregate["classification"], "confirmed": False}],
        documents=[f"{STUDY_ID}: exploratory country-probe accuracy {metrics['accuracy']:.8f}; "
                   f"J94-minus-full difference {experiment.primary_metric:.8f}. No model answer accuracy or causal claim."])
    hypothesis_readback = logger.hypotheses_collection.get(ids=[STUDY_ID], include=["metadatas"])
    experiment_readback = logger.experiments_collection.get(ids=[result_hash], include=["metadatas"])
    hm, em = hypothesis_readback["metadatas"][0], experiment_readback["metadatas"][0]
    if (hypothesis_readback["ids"] != [STUDY_ID] or hm["confirmed"] is not False
            or hm["confirmation_status"] != aggregate["classification"] or hm["tested"] is not True
            or experiment_readback["ids"] != [result_hash] or em["hypothesis_id"] != STUDY_ID
            or em["primary_metric"] != experiment.primary_metric or em["accuracy"] != metrics["accuracy"]
            or em["accuracy_scope"] != aggregate["accuracy_scope"] or em["task_accuracy_measured"] is not False):
        raise RuntimeError("Registry readback mismatch")
    paths = [root/"plan.json", root/"result.json", root/"campaign_results.json", report, Path(__file__)]
    record = {"schema_version": "nal.meta-hypothesis.v1", "conformance": "EXPLORATORY_POSTHOC",
              "hypothesis": hypothesis, "result": aggregate,
              "source_sha256": {str(path): file_hash(path) for path in paths},
              "storage": {"experiment_id": experiment.experiment_id, "result_hash": result_hash,
                          "readback_verified": True, "accuracy_scope": aggregate["accuracy_scope"]}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT/"data/experiments/jspace_country_probe/20261113_v1")
    parser.add_argument("--report", type=Path, default=ROOT/"docs/08 - Analysis/jspace/2026-11-13_jspace-country-probe.md")
    parser.add_argument("--output", type=Path, default=ROOT/"data/hypothesis_registry/jspace_country_probe_20261113.json")
    parser.add_argument("--chromadb", type=Path, default=ROOT/"data/chroma_db")
    args = parser.parse_args()
    record = store(args.root, args.report, args.output, args.chromadb)
    print(json.dumps(record["storage"]))
