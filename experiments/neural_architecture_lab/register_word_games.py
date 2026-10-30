"""Register a completed WORD_GAMES result with exact and semantic readback."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, sha, write
from neural_architecture_lab.word_games_hypothesis import WordGamesEvidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    c, a = read(args.campaign), read(args.audit)
    if c["status"] != "completed_exploratory" or c["completed"] != c["requested"]:
        raise ValueError("complete exploratory campaign required")
    if a["status"] != "readback_verified" or a["campaign_sha256"] != sha(args.campaign):
        raise ValueError("matching verified audit required")
    for path, expected in a["artifacts"].items():
        if sha(path) != expected:
            raise ValueError("changed audited artifact: " + path)
    m = c["metrics"]
    record = WordGamesEvidence(str(args.campaign), sha(args.campaign), str(args.audit), sha(args.audit),
        str(args.report), sha(args.report), c["completed"], m["sender_operation_match"],
        m["boundary_outcome_match"], m["receiver_state_correct"], a["cases_with_disclosure_failure"]).record()
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    if logger.hypotheses_collection is None:
        raise RuntimeError("registry unavailable")
    exact = logger.hypotheses_collection.get(ids=[record["id"]], include=["metadatas"])
    if exact["ids"] != [record["id"]]:
        raise RuntimeError("exact registry readback missing")
    metadata = exact["metadatas"][0]
    for key in ("tested", "confirmed", "confirmation_status", "direct_experiment_count"):
        if metadata[key] != record[key]:
            raise RuntimeError("registry metadata mismatch: " + key)
    semantic = logger.hypotheses_collection.query(query_texts=[record["description"]], n_results=10)
    if record["id"] not in semantic["ids"][0]:
        raise RuntimeError("semantic registry readback missing")
    saved = {"record": record, "exact_readback": metadata, "semantic_readback_ids": semantic["ids"][0],
             "status": "registered_and_read_back"}
    for path in (Path("data/hypothesis_registry") / f"{record['id']}.json", args.campaign.parent / "registry_readback.json"):
        write(path, saved)
        if read(path) != saved:
            raise RuntimeError("receipt readback differs")
    print(saved["status"], record["id"])


if __name__ == "__main__":
    main()
