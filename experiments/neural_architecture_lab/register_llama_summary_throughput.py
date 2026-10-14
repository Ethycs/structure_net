"""Register an audited bounded-summary result with exact and semantic readback."""
import argparse
import json
from pathlib import Path

from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from neural_architecture_lab.llama_summary_throughput_hypothesis import SummaryThroughputEvidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.campaign.read_text())
    if c["status"] != "completed_exploratory" or not c["source_review_completed"]:
        raise ValueError("completed audited source reviews required")
    for path_key in ("manifest", "reviews"):
        if sha256(c[path_key]) != c[path_key + "_sha256"]:
            raise ValueError("changed campaign evidence")
    candidate, baseline = c["arms"]["q4_draft8"], c["arms"]["fp16_draft0"]
    evidence = SummaryThroughputEvidence(
        str(args.campaign), sha256(args.campaign), str(args.report), sha256(args.report),
        c["speedup_over_fp16"], c["latency_speedup_over_fp16"], c["speedup_over_existing_q4"],
        candidate["quality_passed"], candidate["quality_total"], baseline["quality_passed"],
        c["source_review_completed"], c["direct_experiment_count"])
    record = evidence.record()
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(
        project_name="structure_net_completed_hypotheses", queue_dir="data/experiment_queue",
        sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    exact = logger.hypotheses_collection.get(ids=[record["id"]], include=["metadatas"])
    if exact["ids"] != [record["id"]]:
        raise RuntimeError("registry exact readback missing")
    metadata = exact["metadatas"][0]
    for key in ("tested", "confirmed", "confirmation_status", "direct_experiment_count"):
        if metadata[key] != record[key]:
            raise RuntimeError("registry metadata changed: " + key)
    semantic = logger.hypotheses_collection.query(query_texts=[record["description"]], n_results=10)
    if record["id"] not in semantic["ids"][0]:
        raise RuntimeError("registry semantic readback missing")
    saved = {"record": record, "exact_readback": metadata,
             "semantic_readback_ids": semantic["ids"][0], "status": "registered_and_read_back"}
    for path in (Path("data/hypothesis_registry") / f"{record['id']}.json", args.campaign.parent / "registry_readback.json"):
        if path.exists() and json.loads(path.read_text()) != saved:
            raise ValueError("preserve completed registry receipt")
        write_json(path, saved)
        if json.loads(path.read_text()) != saved:
            raise RuntimeError("registry receipt readback differs")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
