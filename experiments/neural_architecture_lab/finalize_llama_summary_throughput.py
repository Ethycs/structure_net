"""Audit a sealed, counterbalanced summary campaign and explicit source reviews."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from experiments.neural_architecture_lab.analyze_llama_summary_throughput import audit, summarize
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from neural_architecture_lab.llama_summary_throughput_hypothesis import SummaryThroughputEvidence


def review_key(example_id, text):
    return hashlib.sha256((example_id + "\0" + text).encode()).hexdigest()


def collect(manifest):
    fixture_path = Path(manifest["fixtures"])
    if sha256(fixture_path) != manifest["fixtures_sha256"]:
        raise ValueError("primary fixture identity changed")
    frozen = json.loads(fixture_path.read_text())
    if [(r["model"], r["phase"], r["repeats"]) for r in manifest["runs"]] != [
            ("fp16", "a", 2), ("q4", "a", 2), ("q4", "b", 3), ("fp16", "b", 3)]:
        raise ValueError("counterbalanced model order changed")
    groups = defaultdict(list)
    inventory = {}
    examples = None
    evidence = []
    for cell in manifest["runs"]:
        root = Path(cell["root"])
        raw, producer, current = audit(root)
        if examples is None:
            examples = current
        if current != examples or raw["config"]["drafts"] != [0, 8]:
            raise ValueError("unmatched primary inputs or strategies")
        if raw["config"]["examples"] != frozen["examples"]:
            raise ValueError("primary inputs differ from selection seal")
        if raw["config"]["repeats"] != cell["repeats"] or producer["stage"] != "heldout":
            raise ValueError("wrong primary repeat count or split")
        if producer["model_sha256"] != manifest["models"][cell["model"]]["sha256"]:
            raise ValueError("unmatched model identity")
        for row in raw["rows"]:
            key = review_key(row["id"], row["text"])
            inventory.setdefault(key, {"key": key, "id": row["id"], "text": row["text"],
                                       "fragments": current[row["id"]]["fragments"], "arms": []})
            arm = f"{cell['model']}_draft{row['draft']}"
            if arm not in inventory[key]["arms"]:
                inventory[key]["arms"].append(arm)
            groups[arm].append(row)
        evidence.append({"root": str(root), "raw_sha256": sha256(root / "raw.json"),
                         "producer_fingerprint": producer["fingerprint"], "model": cell["model"],
                         "load_seconds": raw["load_seconds"], "model_weight_bytes": raw["model_weight_bytes"],
                         "resident_increment_bytes": raw["resident_increment_bytes"]})
    expected = len(examples) * 5
    if set(groups) != {"fp16_draft0", "fp16_draft8", "q4_draft0", "q4_draft8"}:
        raise ValueError("incorrect primary arms")
    if any(len(rows) != expected for rows in groups.values()):
        raise ValueError("five repetitions per arm required")
    for rows in groups.values():
        if any(sum(r["id"] == key for r in rows) != 5 for key in examples):
            raise ValueError("five repetitions per example required")
    return groups, inventory, examples, evidence


def finalize(manifest_path, reviews_path, output):
    manifest = json.loads(manifest_path.read_text())
    groups, inventory, examples, evidence = collect(manifest)
    review = json.loads(reviews_path.read_text())
    if review["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("reviews refer to a different campaign")
    rows = review["items"]
    if len(rows) != len({r["key"] for r in rows}) or {r["key"] for r in rows} != set(inventory):
        raise ValueError("every distinct output requires exactly one source review")
    reviewed = {r["key"]: r for r in rows}
    for key, row in reviewed.items():
        if row["id"] != inventory[key]["id"] or not row["rationale"] or type(row["passed"]) is not bool:
            raise ValueError("invalid review disposition")
        if row["source_sha256"] != hashlib.sha256(json.dumps(inventory[key]["fragments"], sort_keys=True).encode()).hexdigest():
            raise ValueError("review source changed")
    arms = {}
    for name, timed in groups.items():
        summary = summarize({"rows": timed}, examples)
        quality = {key: all(reviewed[review_key(r["id"], r["text"])]["passed"]
                            for r in timed if r["id"] == key) for key in examples}
        arms[name] = {"summary": summary, "quality_passed": sum(quality.values()),
                      "quality_total": len(quality), "quality_by_request": quality,
                      "output_variants": len({review_key(r["id"], r["text"]) for r in timed})}
    def overall(name):
        return next(r for r in arms[name]["summary"] if r["workload"] == "all")
    base, candidate, q4 = [overall(k) for k in ("fp16_draft0", "q4_draft8", "q4_draft0")]
    speed = candidate["end_to_end_output_tokens_per_second"] / base["end_to_end_output_tokens_per_second"]
    latency = base["mean_request_seconds"] / candidate["mean_request_seconds"]
    q4_speed = candidate["end_to_end_output_tokens_per_second"] / q4["end_to_end_output_tokens_per_second"]
    ratios = []
    for key in examples:
        b = statistics.median(r["wall_seconds"] for r in groups["fp16_draft0"] if r["id"] == key)
        c = statistics.median(r["wall_seconds"] for r in groups["q4_draft8"] if r["id"] == key)
        ratios.append({"id": key, "median_latency_speedup": b / c})
    result = {"schema": "nal.llama-summary-campaign.v1", "status": "completed_exploratory",
              "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
              "reviews": str(reviews_path), "reviews_sha256": sha256(reviews_path),
              "arms": arms, "runs": evidence, "paired_request_ratios": ratios,
              "speedup_over_fp16": speed, "latency_speedup_over_fp16": latency,
              "speedup_over_existing_q4": q4_speed, "application_accuracy_tested": False,
              "direct_experiment_count": sum(len(v) for v in groups.values()),
              "distinct_requests": len(examples), "source_review_completed": True,
              "measurement_boundary": "warm model inference from tokenized prompt through generated token IDs; includes prefill, cache clear, token selection and decode; excludes loading, tokenization and text detokenization"}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve completed artifacts")
    if args.inventory:
        _, inventory, _, _ = collect(json.loads(args.manifest.read_text()))
        write_json(args.output, {"manifest_sha256": sha256(args.manifest), "outputs": list(inventory.values())})
    else:
        if args.reviews is None:
            raise ValueError("explicit source reviews required")
        result = finalize(args.manifest, args.reviews, args.output)
        print(json.dumps({k: result[k] for k in ("speedup_over_fp16", "latency_speedup_over_fp16", "speedup_over_existing_q4")}))


if __name__ == "__main__":
    main()
