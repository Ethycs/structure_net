"""Verify immutable native speed evidence and aggregate comparable token counts."""
import argparse
from pathlib import Path

from .common import digest, read, sha, write
from .native_speed import aggregate, PILOT, WEIGHTS_SHA


def replay(root):
    campaign = read(root / "campaign_results.json")
    producer = read(root / "producer.json")
    fingerprint = digest({k: v for k, v in producer.items() if k != "fingerprint"})
    if fingerprint != producer["fingerprint"] or campaign["fingerprint"] != fingerprint:
        raise ValueError("producer fingerprint mismatch")
    for path, expected_hash in producer["source_hashes"].items():
        if sha(root / Path(path).name) != expected_hash:
            raise ValueError("archived producing source changed")
    if producer["weights_sha256"] != WEIGHTS_SHA:
        raise ValueError("target checkpoint mismatch")
    paths = sorted(root.glob("runs/**/response.json"))
    expected = 36 if campaign["stage"] == "full" else len(PILOT)
    if len(paths) != expected or campaign["totals"]["calls"] != expected:
        raise ValueError("incomplete replay")
    rows = {}
    for p in paths:
        if sha(p) != campaign["artifacts"][str(p)]:
            raise ValueError("changed replay response")
        r = read(p)
        if r["response"]["timings"]["cache_n"] != 0:
            raise ValueError("cross-request cache reuse")
        rows[str(p.parent.relative_to(root / "runs"))] = r
    return {"root": str(root), "campaign_sha256": sha(root / "campaign_results.json"),
            "totals": aggregate(list(rows.values())), "stage": campaign["stage"],
            "spec": producer["spec"], "ngram": producer["ngram"], "draft": producer["draft"],
            "compact_json": producer.get("compact_json", False),
            "kv": producer.get("kv", "q8_0"), "batch": producer.get("batch", 512),
            "greedy_only": producer.get("greedy_only", False), "no_grammar": producer.get("no_grammar", False),
            "backend_sampling_requested": producer["backend_sampling"]}, rows


def quality(root):
    c = read(root / "quality/campaign_results.json")
    rows = {p.parent.name: read(p) for p in (root / "quality/cases").glob("*/result.json")}
    if c["completed"] != 12 or len(rows) != 12:
        raise ValueError("complete live quality suite required")
    for p, h in c["artifacts"].items():
        if sha(p) != h:
            raise ValueError("quality result changed")
    joint = [k for k, r in rows.items() if all(r[v] for v in (
        "sender_operation_match", "boundary_outcome_match", "receiver_state_correct"))]
    return {"root": str(root), "campaign_sha256": sha(root / "quality/campaign_results.json"),
            "metrics": c["metrics"], "joint_passes": sorted(joint)}, rows


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    baseline, br = replay(args.baseline)
    optimized, op = replay(args.optimized)
    if baseline["stage"] != "full" or optimized["stage"] != "full" or set(br) != set(op):
        raise ValueError("matched full corpus required")
    bq, bqr = quality(args.baseline_quality)
    oq, oqr = quality(args.optimized_quality)
    for name in bqr:
        rel = Path("quality/cases") / name / "context.json"
        if read(args.baseline_quality / rel) != read(args.optimized_quality / rel):
            raise ValueError("live fixture contexts differ")
    regressions = sorted(set(bq["joint_passes"]) - set(oq["joint_passes"]))
    component_regressions = {key: sorted(name for name in bqr if bqr[name][key] and not oqr[name][key])
                            for key in ("sender_operation_match", "boundary_outcome_match", "receiver_state_correct")}
    disclosure = any(oq["metrics"][k] for k in (
        "canary_in_receiver_input", "canary_in_receiver_output", "forbidden_source_id_in_packet"))
    review = read(args.semantic_review)
    quality_ok = not any(component_regressions.values()) and not disclosure and review["no_new_factual_errors"]
    rate = optimized["totals"]["output_tokens_per_second"]
    intact = optimized["totals"]["json_objects"] == 36 and optimized["totals"]["truncated"] == 0
    result = {"hypothesis_id": "word-games-phi4-native-40tps-v1", "confirmed": False,
              "classification": "observed_40tps_with_no_suite_regression" if rate >= 40 and quality_ok and intact else "below_40tps" if rate < 40 else "quality_gate_failed",
              "target_output_tokens_per_second": 40, "baseline": baseline, "optimized": optimized,
              "baseline_quality": bq, "optimized_quality": oq, "quality_regressions": regressions,
              "component_regressions": component_regressions,
              "semantic_review": str(args.semantic_review), "semantic_review_sha256": sha(args.semantic_review),
              "observed_quality_gate": quality_ok and intact,
              "latency_speedup": baseline["totals"]["request_seconds"] / optimized["totals"]["request_seconds"],
              "exact_text_pairs": sum(br[k]["response"]["content"] == op[k]["response"]["content"] for k in br),
              "paired_requests": 36, "pilots": [replay(p)[0] for p in args.pilot],
              "limitations": ["One GPU, one model and twelve synthetic workflows",
                              "Manual summary review, no independent semantic judge",
                              "Runtime versions and numerical paths differ from the historical Python pilot"]}
    write(args.output / "campaign_results.json", result)
    print(result["classification"], "rate", rate, "quality", quality_ok)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "optimized", "baseline-quality", "optimized-quality", "semantic-review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--pilot", action="append", type=Path, default=[])
    main(p.parse_args())
