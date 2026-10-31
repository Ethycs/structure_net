"""Register checked native Phi-4 evidence and verify exact/semantic readback."""
import argparse
from pathlib import Path

from experiments.word_games.common import read, sha, write
from neural_architecture_lab.word_games_native_speed_hypothesis import WordGamesNativeSpeedEvidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    c = read(args.campaign)
    for key in ("baseline", "optimized", "baseline_quality", "optimized_quality"):
        r = c[key]
        path = Path(r["root"]) / ("quality/campaign_results.json" if key.endswith("quality") else "campaign_results.json")
        if sha(path) != r["campaign_sha256"]:
            raise ValueError("changed evidence: " + str(path))
    if sha(c["semantic_review"]) != c["semantic_review_sha256"]:
        raise ValueError("changed manual review")
    for path, expected in read(c["semantic_review"])["artifacts"].items():
        if sha(path) != expected:
            raise ValueError("changed manually reviewed artifact")
    record = WordGamesNativeSpeedEvidence(c["optimized"]["totals"]["output_tokens_per_second"],
        c["paired_requests"], 12, c["observed_quality_gate"], c["classification"],
        str(args.campaign), sha(args.campaign), str(args.report), sha(args.report)).record()
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    if logger.hypotheses_collection is None:
        raise RuntimeError("registry unavailable")
    # The shared logger indexes only its common hypothesis fields. Preserve
    # this study's scalar measurements on its own record before readback.
    logger.hypotheses_collection.update(ids=[record["id"]], metadatas=[{
        k: v for k, v in record.items() if isinstance(v, (str, int, float, bool))}])
    exact = logger.hypotheses_collection.get(ids=[record["id"]], include=["metadatas"])
    if exact["ids"] != [record["id"]]:
        raise RuntimeError("missing exact readback")
    for k in ("tested", "confirmed", "confirmation_status", "output_tokens_per_second", "observed_quality_gate"):
        if exact["metadatas"][0][k] != record[k]:
            raise RuntimeError("incorrect readback: " + k)
    semantic = logger.hypotheses_collection.query(query_texts=[record["description"]], n_results=10)
    if record["id"] not in semantic["ids"][0]:
        raise RuntimeError("missing semantic readback")
    receipt = {"record": record, "exact_readback": exact["metadatas"][0], "semantic_ids": semantic["ids"][0]}
    for dest in (args.campaign.parent / "registry_readback.json", Path("data/hypothesis_registry") / (record["id"] + ".json")):
        write(dest, receipt)
        if read(dest) != receipt:
            raise RuntimeError("registry persistence failed")
    print("registered_and_read_back", record["confirmation_status"])


if __name__ == "__main__":
    main()
