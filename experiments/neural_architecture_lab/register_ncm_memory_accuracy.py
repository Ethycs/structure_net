"""Register the frozen accuracy result, retaining study-specific scalar metadata.

The original registration reached Chroma but failed its exact readback because
StandardizedLogger.log_hypothesis keeps a fixed subset of metadata. This
registration-only recovery preserves the completed campaign, report and judges.
"""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def register(root, report):
    path = root / "campaign_results.json"
    result = q.read(path)
    for file, expected in result["artifacts"].items():
        if file_hash(file) != expected:
            raise ValueError("completed evidence changed")
    score = result["models"]["trained_ncm"]
    record = {
        "id": q.HYPOTHESIS,
        "name": "End-to-end task accuracy of the memory-trained NCM",
        "description": f"The memory-trained NCM passed {score['passes']}/{score['n']} held-out "
        f"summaries ({score['task_pass_rate']:.2%}) under source-grounded blinded LLM judging. "
        "Measures complete summary faithfulness, main-point usefulness and readability. "
        "Codec reconstruction accuracy is separate. One checkpoint, native short-context range.",
        "category": "architecture", "tested": True, "confirmed": False,
        "confirmation_status": result["classification"], "status": "completed_exploratory",
        "underpowered": True, "human_rated": False,
        "task_pass_rate": score["task_pass_rate"], "task_passes": score["passes"],
        "task_cases": score["n"], "judge": result["primary_judge"],
        "application_accuracy_tested": False, "direct_experiment_count": 512,
        "speed_tested": False, "campaign": str(path), "campaign_sha256": file_hash(path),
        "report": str(report), "report_sha256": file_hash(report),
    }
    destinations = (root / "registry_readback.json",
                    Path("data/hypothesis_registry") / (q.HYPOTHESIS + ".json"))
    if any(p.exists() for p in destinations):
        raise ValueError("fresh registry receipts required")
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(
        project_name="structure_net_completed_hypotheses", queue_dir="data/experiment_queue",
        sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected", enable_wandb=False,
        auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    collection = logger.hypotheses_collection
    if collection is None:
        raise ValueError("searchable registry unavailable")
    # Extend the stored metadata explicitly; the shared logger intentionally
    # admits only selected fields. All fields in this study's record are scalars.
    existing = collection.get(ids=[q.HYPOTHESIS], include=["metadatas"])
    if existing["ids"] != [q.HYPOTHESIS]:
        raise ValueError("initial exact readback failed")
    collection.update(ids=[q.HYPOTHESIS], metadatas=[{**existing["metadatas"][0], **record}])
    exact = collection.get(ids=[q.HYPOTHESIS], include=["metadatas"])
    if exact["ids"] != [q.HYPOTHESIS] or any(
            exact["metadatas"][0].get(k) != v for k, v in record.items()):
        raise ValueError("complete exact readback failed")
    semantic = collection.query(query_texts=[record["description"]], n_results=10)
    if q.HYPOTHESIS not in semantic["ids"][0]:
        raise ValueError("semantic registry readback failed")
    receipt = {
        "status": "registered_and_read_back", "record": record,
        "exact_readback": exact["metadatas"][0], "semantic_ids": semantic["ids"][0],
        "registration_source": __file__, "registration_source_sha256": file_hash(__file__),
        "recovery": "Original registration omitted task_passes/task_cases from searchable metadata; "
        "registration-only recovery extends scalar metadata and verifies every field. "
        "No aggregate, report, judge reply, score, or producing source was changed.",
    }
    for destination in destinations:
        write_json(destination, receipt)
        if q.read(destination) != receipt:
            raise ValueError("JSON readback failed")
    print(json.dumps({"status": receipt["status"], "passes": score["passes"], "n": score["n"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    register(args.root, args.report)
