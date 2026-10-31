"""Hash-check speculative decoding evidence and verify research registry reads."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, sha, write
from neural_architecture_lab.word_games_speculation_hypothesis import WordGamesSpeculationEvidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    c = read(args.campaign)
    for path, expected in c["artifacts"].items():
        # Paths in worker receipts belong to its read/write container mount.
        real = args.campaign.parent.parent / Path(path).relative_to("/artifacts")
        if sha(real) != expected:
            raise ValueError("changed measured result: " + str(real))
    record = WordGamesSpeculationEvidence(c["stage"], c["classification"], c["requested_calls"],
        c["completed_calls"], len(c["pairs"]), sum(p["tokens_equal"] and p["text_equal"] for p in c["pairs"]),
        str(args.campaign), sha(args.campaign), str(args.report), sha(args.report)).record()
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    if logger.hypotheses_collection is None:
        raise RuntimeError("registry unavailable")
    exact = logger.hypotheses_collection.get(ids=[record["id"]], include=["metadatas"])
    if exact["ids"] != [record["id"]]:
        raise RuntimeError("registry exact readback missing")
    for key in ("tested", "confirmed", "confirmation_status", "direct_experiment_count"):
        if exact["metadatas"][0][key] != record[key]:
            raise RuntimeError("registry readback mismatch: " + key)
    semantic = logger.hypotheses_collection.query(query_texts=[record["description"]], n_results=10)
    if record["id"] not in semantic["ids"][0]:
        raise RuntimeError("semantic readback missing")
    saved = {"record": record, "exact_readback": exact["metadatas"][0], "semantic_ids": semantic["ids"][0],
             "status": "registered_and_read_back"}
    for path in (args.campaign.parent / "registry_readback.json", Path("data/hypothesis_registry") / (record["id"] + ".json")):
        write(path, saved)
        if read(path) != saved:
            raise RuntimeError("persisted registry receipt mismatch")
    print(saved["status"], record["confirmation_status"])


if __name__ == "__main__":
    main()
