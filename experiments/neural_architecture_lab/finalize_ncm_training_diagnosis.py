"""Seal and register a forward-only, exploratory NCM failure diagnosis."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from experiments.structure_net.diagnose_ncm_training import HYPOTHESIS, aggregate
from experiments.structure_net.ncm_memory_data import file_hash, write_json


@dataclass(frozen=True)
class Evidence:
    root: Path
    report: Path

    def record(self):
        campaign = json.loads((self.root / "campaign_results.json").read_text())
        for path, expected in campaign["artifacts"].items():
            if file_hash(path) != expected:
                raise ValueError("diagnostic evidence changed")
        return {"id": HYPOTHESIS, "name": "Localize the NCM summary training failure",
            "description": "Post-hoc forward-only comparison of oracle codec reconstruction, "
            "teacher-forced and free-running concept prediction, oracle coarse codes, and "
            "published decoder penalties on 64 reused held-out cases. No new task accuracy "
            "or training; prior frozen-body 0/64 quality result remains unchanged.",
            "category": "architecture", "tested": True, "confirmed": False,
            "confirmation_status": campaign["classification"], "underpowered": True,
            "direct_experiment_count": 64, "new_training_runs": 0, "reused_training_seeds": 1,
            "task_accuracy_measured": False, "speed_tested": False,
            "campaign": str(self.root / "campaign_results.json"),
            "campaign_sha256": file_hash(self.root / "campaign_results.json"),
            "report": str(self.report), "report_sha256": file_hash(self.report)}


def seal(root):
    destination = root / "campaign_results.json"
    if destination.exists():
        raise ValueError("append-only aggregate required")
    result = json.loads((root / "result.json").read_text())
    producer = json.loads((root / "producer.json").read_text())
    records = json.loads((root / "records.json").read_text())
    if (result["status"] != "completed_exploratory_diagnosis"
            or result["summary"] != aggregate(records)
            or result["records_sha256"] != file_hash(root / "records.json")
            or result["producer_fingerprint"] != producer["fingerprint"]):
        raise ValueError("diagnostic aggregation/readback mismatch")
    oracle = [r for r in records if r.get("kind") == "oracle_codec"]
    if len(oracle) != 64:
        raise ValueError("fixed 64-case oracle cohort required")
    for arm in ("hybrid", "native"):
        for split, count in (("train", 32), ("test", 64)):
            chosen = [r for r in records if r.get("arm") == arm and r["split"] == split]
            if len(chosen) != count or len({r["id"] for r in chosen}) != count:
                raise ValueError("incomplete or duplicated diagnostic cohort")
        chosen = [r for r in records if r.get("arm") == arm and r["split"] == "test"]
        if {r["id"] for r in chosen} != {r["id"] for r in oracle}:
            raise ValueError("oracle and generation cases differ")
    files = [root / n for n in ("result.json", "producer.json", "records.json")]
    files += [Path(__file__)]
    campaign = {"schema": "nal.ncm-diagnosis-campaign.v1", "hypothesis_id": HYPOTHESIS,
        "status": "completed", "classification": "exploratory_code_generation_failure_diagnosed",
        "conformance_profile": "EXPLORATORY", "requested_runs": 1, "completed_runs": 1,
        "new_training_runs": 0, "invalidated_pilots": ["20261022_diagnosis_pilot_v1"],
        "successful_pilot": "20261022_diagnosis_pilot_v2", "config": producer["config"],
        "producer_fingerprint": producer["fingerprint"], "summary": result["summary"],
        "task_accuracy_measured": False, "parent_task_verdict_unchanged": True,
        "speed_tested": False, "underpowered": True,
        "artifacts": {str(p): file_hash(p) for p in files}}
    write_json(destination, campaign)
    if json.loads(destination.read_text()) != campaign:
        raise ValueError("campaign JSON readback failed")


def register(root, report):
    record = Evidence(root, report).record()
    paths = [root / "registry_readback.json", Path("data/hypothesis_registry") / (HYPOTHESIS+".json")]
    if any(p.exists() for p in paths):
        raise ValueError("fresh registration receipt required")
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    collection = logger.hypotheses_collection
    if collection is None:
        raise ValueError("registry unavailable")
    exact = collection.get(ids=[HYPOTHESIS], include=["metadatas"])
    collection.update(ids=[HYPOTHESIS], metadatas=[{**exact["metadatas"][0], **record}])
    exact = collection.get(ids=[HYPOTHESIS], include=["metadatas"])
    if exact["ids"] != [HYPOTHESIS] or any(exact["metadatas"][0].get(k) != v for k, v in record.items()):
        raise ValueError("exact registry readback failed")
    semantic = collection.query(query_texts=[record["description"]], n_results=10)
    if HYPOTHESIS not in semantic["ids"][0]:
        raise ValueError("semantic readback failed")
    receipt = {"status": "registered_and_read_back", "record": record,
               "exact": exact["metadatas"][0], "semantic_ids": semantic["ids"][0]}
    for path in paths:
        write_json(path, receipt)
        if json.loads(path.read_text()) != receipt:
            raise ValueError("JSON registry readback failed")
    print(json.dumps({"status": receipt["status"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["seal", "register"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    seal(args.root) if args.command == "seal" else register(args.root, args.report)
