"""Independent cell/provenance audit and source-review aid for tensor runtime."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from experiments.structure_net.llama_tensor_runtime import HYPOTHESIS_ID, digest
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH, sha256, write_json


def source_checks(row, example, tokenizer):
    lines = [line.strip() for line in row["text"].splitlines() if line.strip()]
    details = []
    for i, f in enumerate(example["fragments"]):
        line = lines[i] if i < len(lines) else ""
        flat = line.replace(",", "")
        project = f["reference_fact"].split(":", 1)[0]
        required = f["required"]
        checks = {"label": line.startswith(f["id"]+":"), "project": project in line,
                  "owner": required["owner"] in line, "budget": "$"+required["budget"] in flat,
                  "deadline": required["deadline"] in line,
                  "within_summary_limit": len(tokenizer.encode(line, add_special_tokens=False)) <= 128,
                  "gold_in_source": f["reference_fact"] in f["source"]}
        details.append({"id": f["id"], "line": line, "source_quote": f["reference_fact"],
                        "checks": checks, "tokens": len(tokenizer.encode(line, add_special_tokens=False))})
    eos = row["output_ids"][-1] in {128001, 128008, 128009}
    return {"cardinality": len(lines) == len(example["fragments"]), "eos": eos,
            "fragments": details, "diagnostic_pass": eos and len(lines) == len(example["fragments"])
            and all(all(d["checks"].values()) for d in details),
            "boundary": "line-scoped fact and limit diagnostics; explicit source review still required"}


def audit(root, examples):
    producer = json.loads((root / "producer.json").read_text())
    terminal = json.loads((root / "result.json").read_text())
    if producer["hypothesis_id"] != HYPOTHESIS_ID or digest({k:v for k,v in producer.items() if k != "fingerprint"}) != producer["fingerprint"]:
        raise ValueError("producer fingerprint mismatch")
    if terminal["status"] != "completed_pending_source_review" or terminal["producer_fingerprint"] != producer["fingerprint"] or terminal["rows_sha256"] != sha256(root / "rows.json"):
        raise ValueError("terminal integrity mismatch")
    snapshots = {sha256(p) for p in (root / "producer_source").iterdir()}
    for p, expected in producer["source_hashes"].items():
        if sha256(p) != expected or (not p.endswith(".so") and expected not in snapshots):
            raise ValueError("producing source changed or snapshot absent")
    for p, expected in producer["factor_hashes"].items():
        if sha256(p) != expected:
            raise ValueError("factor identity changed")
    payload = json.loads((root / "rows.json").read_text())
    rows = payload["rows"]
    expected = {(arm, e, r) for arm in producer["arms"] for e in examples for r in range(producer["repeats"])}
    actual = [(r["arm"], r["id"], r["repeat"]) for r in rows]
    if payload["status"] != "completed" or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("missing or duplicate cells")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    for row in rows:
        order = producer["arms"] if row["repeat"] % 2 == 0 else list(reversed(producer["arms"]))
        if row["order"] != order or tokenizer.decode(row["output_ids"], skip_special_tokens=True) != row["text"]:
            raise ValueError("timing order or decoded output mismatch")
        if not 0 < row["ttft_seconds"] <= row["wall_seconds"] or any(not math.isfinite(row[k]) for k in ("ttft_seconds", "wall_seconds")):
            raise ValueError("invalid timings")
        if row["cap"] != examples[row["id"]]["cap"] or not 0 < len(row["output_ids"]) <= row["cap"]:
            raise ValueError("invalid output length")
        if row["eos"] != (row["output_ids"][-1] in {128001, 128008, 128009}):
            raise ValueError("EOS flag mismatch")
    return rows, producer


def summarize(rows, examples):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["arm"], r["id"])].append(r)
    arms = defaultdict(list)
    for (arm, identifier), group in grouped.items():
        arm_row = {"id": identifier, "family": examples[identifier]["kind"]+"_"+str(examples[identifier]["target_prompt_tokens"]),
                   "median_seconds": statistics.median(r["wall_seconds"] for r in group),
                   "median_ttft": statistics.median(r["ttft_seconds"] for r in group),
                   "output_tokens": statistics.median(sum(t not in {128001, 128008, 128009} for t in r["output_ids"]) for r in group),
                   "unique_outputs": len({tuple(r["output_ids"]) for r in group}),
                   "proposed_tokens": sum(r.get("proposed_tokens", 0) for r in group),
                   "accepted_tokens": sum(r.get("accepted_tokens", 0) for r in group)}
        arms[arm].append(arm_row)
    result = {}
    for arm, items in arms.items():
        seconds = sum(r["median_seconds"] for r in items)
        tokens = sum(r["output_tokens"] for r in items)
        families = {}
        for family in sorted({r["family"] for r in items}):
            part = [r for r in items if r["family"] == family]
            families[family] = {"requests":len(part), "seconds":sum(r["median_seconds"] for r in part),
                "output_tokens":sum(r["output_tokens"] for r in part)}
        result[arm] = {"sum_median_seconds":seconds, "end_to_end_tokens_per_second":tokens/seconds,
            "mean_median_request_seconds":seconds/len(items), "output_tokens":tokens,
            "sum_median_ttft":sum(r["median_ttft"] for r in items), "per_case":items,
            "families":families, "all_repetitions_same_output":all(r["unique_outputs"] == 1 for r in items)}
    return result


def make_review_aid(rows, examples, tokenizer):
    result, seen = [], set()
    for row in rows:
        key = (row["arm"], row["id"], digest(row["text"]))
        if key in seen:
            continue
        seen.add(key)
        result.append({"arm":row["arm"], "id":row["id"], "text":row["text"],
            "text_sha256":digest(row["text"]), "source_sha256":digest(examples[row["id"]]["fragments"]),
            "diagnostics":source_checks(row, examples[row["id"]], tokenizer)})
    return result


def main():
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--native", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("fresh analysis path required")
    examples = {e["id"]:e for e in json.loads(args.fixtures.read_text())["examples"] if e["split"] == args.split}
    if args.native:
        from experiments.neural_architecture_lab.analyze_llama_summary_throughput import audit as audit_native
        raw, producer, native_examples = audit_native(args.root)
        for identifier, e in examples.items():
            if native_examples[identifier]["prompt_ids"] != e["prompt_ids"] or native_examples[identifier]["cap"] != e["cap"]:
                raise ValueError("native workload differs")
        rows = [dict(r, arm="q4_lookup8", output_ids=r["generated_ids"], eos=r["stopped_eos"])
                for r in raw["rows"] if r["id"] != "excluded_warmup_longest"]
        raw_path = args.root / "raw.json"
    else:
        rows, producer = audit(args.root, examples)
        raw_path = args.root / "rows.json"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    write_json(args.output, {"status":"integrity_passed_source_review_pending", "producer_fingerprint":producer["fingerprint"],
        "summary":summarize(rows, examples), "review_aid":make_review_aid(rows, examples, tokenizer),
        "rows_sha256":sha256(raw_path), "fixtures_sha256":sha256(args.fixtures)})


if __name__ == "__main__":
    main()
