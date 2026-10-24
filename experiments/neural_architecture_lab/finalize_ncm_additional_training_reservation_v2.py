"""Independent metadata/count/exclusion audit for new TRAIN-source reservation."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.structure_net import ncm_additional_training_reservation as run
from experiments.neural_architecture_lab.ncm_result_registry import register_records

RESULTS = run.ROOT / "20261024_additional10000_training_reservation_results_v2"
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-additional-training-reservation-v2.md")


def finalize():
    if RESULTS.exists() or REPORT.exists():
        raise ValueError("fresh metadata aggregate required")
    if run.read(run.WORKFLOW / "tool_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("actual whole-tool completion required")
    index, reserved, _ = run.parent_metadata()
    by_id = {row["id"]: row for row in index}
    blocked = {row["source_component"] for row in index if row["blocked"]}
    journal = run.read(run.WORKFLOW / "result.json")
    if ([r["stage"] for r in journal["stages"]] != ["pilot", "reserve"] or any(r["exit_code"] != 0 for r in journal["stages"])):
        raise ValueError("both actual worker exits required")
    details = {}
    for root, quotas in ((run.PILOT, {k: 2 for k in run.QUOTAS}), (run.DEST, run.QUOTAS)):
        if run.read(root / "worker_terminal_receipt.json")["exit_code"] != 0:
            raise ValueError("actual reservation worker completion required")
        producer, result = run.read(root / "producer.json"), run.read(root / "result.json")
        run.verify(producer)
        selected = run.read(root / "training_metadata.json")
        groups = {row["source_component"] for row in selected}
        if (len(selected) != sum(quotas.values()) or len(groups) != len(selected)
                or groups & (reserved | blocked) or any(row != by_id[row["id"]] for row in selected)
                or Counter(row["bin"] for row in selected) != Counter(quotas)
                or any(not row["candidate"] or run.graph.bucket(row["tokens"]) != row["bin"] for row in selected)
                or any("source" in row or "target_text" in row or "text" in row for row in selected)
                or result["metadata_sha256"] != run.file_hash(root / "training_metadata.json")
                or result["cases"] != len(selected) or result["source_components_distinct"] != len(groups)
                or result["peak_RSS_bytes"] > run.MAX_RSS or result["new_model_trained"] or result["new_labels_generated"]
                or result["selected_final_sources_read_or_hashed"] or result["task_accuracy"] is not None):
            raise ValueError("original metadata membership, all quotas or complete component exclusions differ")
        expected, availability = run.choose(index, reserved, quotas)
        if selected != expected or any(result[k] != v for k, v in json.loads(json.dumps(availability)).items()):
            raise ValueError("fixed deterministic selection or availability replay differs")
        details["pilot" if root == run.PILOT else "full"] = result
    artifacts = {}
    for root in (run.PILOT, run.DEST, run.WORKFLOW):
        for path in root.rglob("*.json"):
            artifacts[str(path)] = run.file_hash(path)
    artifacts[str(Path(__file__))] = run.file_hash(__file__)
    amendment = Path("docs/07 - Status Reports/2026-10-24_ncm-additional-training-reservation-audit-amendment.md")
    artifacts[str(amendment)] = run.file_hash(amendment)
    failure = run.ROOT / "20261024_additional10000_training_reservation_audit_attempt_v1/actual_tool_terminal_receipt.json"
    artifacts[str(failure)] = run.file_hash(failure)
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": "additional10000_disjoint_training_sources_available_labels_unmade",
        "details": details, "confirmed": False, "task_accuracy": None, "quality_measured": False,
        "canonical_first_judgments_changed": False, "selected_final_sources_read_or_hashed": False,
        "final_test_opened": False, "speed_tested": False, "artifacts": artifacts}
    RESULTS.mkdir()
    path = RESULTS / "campaign_results.json"
    run.write_json(path, campaign)
    REPORT.write_text(
        "# Additional TRAIN-source reservation: metadata result\n\n"
        "The first audit stopped on integer-versus-JSON-string band keys. This fresh\n"
        "auditor normalizes only the expected metadata at the JSON boundary. Original\n"
        "producer source, measured outputs and failed audit receipt are preserved.\n\n"
        f"Hypothesis `{run.HYPOTHESIS}`. Campaign `{path}`, SHA256 `{run.file_hash(path)}`.\n\n"
        "The frozen source-component graph admits 10,000 additional distinct training\n"
        "components: 5,000 sources in192-511cl100k and1,250 in each of the four longer\n"
        "bands through2048. All blocked components and all448 permanent DEV/final\n"
        "reservations were excluded. The independent CPU auditor checked every selected\n"
        "row against the original metadata and replayed full deterministic quotas.\n\n"
        "This is about14.5times the current690-example training population in additional\n"
        "sources alone. It establishes data availability, not that more data fixes the\n"
        "model. No source text, selected final source file or model output was needed;\n"
        "no labels, features or new model were produced, and canonical semantic first\n"
        "judgments are unchanged. Task accuracy and speed remain unmeasured here. The\n"
        "current fixed1024 training/evaluation study is unchanged. The95%complete-summary\n"
        "objective still requires successful source-only task evaluation.\n")
    print(json.dumps({"classification": campaign["classification"], "campaign_sha256": run.file_hash(path),
        "additional_train_sources": 10000, "task_accuracy": None}), flush=True)


def register():
    register_records(RESULTS / "campaign_results.json", REPORT, RESULTS, {run.HYPOTHESIS: {
        "name": "Ten thousand additional component-disjoint NCM training sources",
        "description": "Metadata-only reservation of 10000 additional distinct TRAIN components with unchanged DEV/final exclusions; no source text, labels, model training, semantic-cache changes or quality claim.",
        "classification": "additional10000_disjoint_training_sources_available_labels_unmade",
        "training_sources_available": 10000, "metadata_only": True, "task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    {"finalize": finalize, "register": register}[args.stage]()
