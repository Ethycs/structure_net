"""Independent arithmetic and provenance readback for the speculative pilot/run."""
import argparse
from pathlib import Path

from experiments.word_games.common import digest, read, sha, write


def audit(root):
    root = Path(root)
    c = read(root / "campaign_results.json")
    producer = read(root / "producer.json")
    fingerprint = producer.pop("fingerprint")
    if digest(producer) != fingerprint or c["fingerprint"] != fingerprint:
        raise ValueError("producer fingerprint mismatch")
    if read(root.parent / "exit.json")["exit_code"] != 0:
        raise ValueError("worker did not exit successfully")
    for path, expected in producer["source_hashes"].items():
        if path.startswith("/workspace/"):
            current = Path(path).relative_to("/workspace")
        else:
            current = Path(path)
        if sha(current) != expected or sha(root / current.name) != expected:
            raise ValueError("source mismatch: " + path)
    runs = []
    for path, expected in c["artifacts"].items():
        real = root.parent / Path(path).relative_to("/artifacts")
        if sha(real) != expected:
            raise ValueError("result hash mismatch")
        r = read(real)
        request = read(real.with_name("request.json"))
        if sha(request["source"]) != request["source_sha256"]:
            raise ValueError("original frozen input changed")
        if request["id"] != r["request_id"]:
            raise ValueError("request identity mismatch")
        runs.append(r)
    if len(runs) != c["completed_calls"] or len(runs) != c["requested_calls"]:
        raise ValueError("completion mismatch")
    for length, total in c["totals"].items():
        rows = [r for r in runs if r["draft_length"] == int(length)]
        seconds = sum(r["elapsed_seconds"] for r in rows)
        tokens = sum(r["usage"]["completion_tokens"] for r in rows)
        if abs(seconds - total["request_seconds"]) > 1e-8 or tokens != total["output_tokens"]:
            raise ValueError("aggregate totals mismatch")
        if abs(tokens / seconds - total["output_tokens_per_second"]) > 1e-8:
            raise ValueError("throughput mismatch")
    for pair in c["pairs"]:
        b = next(r for r in runs if r["request_id"] == pair["request_id"] and r["draft_length"] == 0)
        s = next(r for r in runs if r["request_id"] == pair["request_id"] and r["draft_length"] == pair["draft_length"])
        if pair["tokens_equal"] != (b["target_tokens"] == s["target_tokens"]) or pair["text_equal"] != (b["text"] == s["text"]):
            raise ValueError("parity flag mismatch")
    output = {"status": "readback_verified", "campaign_sha256": sha(root / "campaign_results.json"),
        "measured_calls": len(runs), "exact_pairs": sum(p["tokens_equal"] and p["text_equal"] for p in c["pairs"]),
        "pairs": len(c["pairs"]), "full_benchmark_executed": c["stage"] == "full"}
    write(root / "readback_audit.json", output)
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    audit(p.parse_args().root)
