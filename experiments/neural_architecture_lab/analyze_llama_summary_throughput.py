"""Independent integrity and performance analysis; fact checks are diagnostic."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json


def normalized(text):
    return re.sub(r"\s+", " ", text.lower().replace(",", "").replace("november", "nov")).strip()


def fact_diagnostic(text, example):
    value = normalized(text)
    rows = []
    for fragment in example["fragments"]:
        required = {k: normalized(v) in value for k, v in fragment["required"].items()}
        obsolete = {k: normalized(v) in value for k, v in fragment["superseded"].items()}
        rows.append({"fragment_id": fragment["id"], "required_present": required, "superseded_present": obsolete})
    return {"boundary": "string-presence diagnostic only; cannot establish attribution, entailment or absence of hallucination",
            "rows": rows, "all_required_present": all(all(r["required_present"].values()) for r in rows)}


def audit(root):
    producer = json.loads((root / "producer.json").read_text())
    terminal = json.loads((root / "result.json").read_text())
    if terminal["status"] != "completed" or terminal["raw_sha256"] != sha256(root / "raw.json"):
        raise ValueError("invalid terminal or raw hash")
    from experiments.structure_net.llama_summary_throughput import digest
    body = {k: v for k, v in producer.items() if k != "fingerprint"}
    if digest(body) != producer["fingerprint"] or terminal["producer_fingerprint"] != producer["fingerprint"]:
        raise ValueError("producer fingerprint mismatch")
    snapshots = list((root / "producer_source").iterdir())
    hashes = {sha256(p) for p in snapshots}
    if any(h not in hashes for h in producer["source_hashes"].values()):
        raise ValueError("source snapshot missing or changed")
    raw = json.loads((root / "raw.json").read_text())
    if raw["status"] != "completed":
        raise ValueError("incomplete native result")
    config = raw["config"]
    fixture_name = Path(producer["command"][producer["command"].index("--fixtures") + 1]).name
    frozen_fixtures = [p for p in snapshots if p.name.endswith("_" + fixture_name)]
    if len(frozen_fixtures) != 1 or json.loads(frozen_fixtures[0].read_text()) != config:
        raise ValueError("runtime configuration differs from frozen fixtures")
    if raw["model"] != str(Path(producer["model"]).resolve()):
        raise ValueError("runtime model differs from producer")
    examples = {e["id"]: e for e in config["examples"]}
    repeats = 1 if raw["profile"].startswith("systems_") else config["repeats"]
    expected = {(key, repeat, draft) for key in examples for repeat in range(repeats) for draft in config["drafts"]}
    actual = [(r["id"], r["repeat"], r["draft"]) for r in raw["rows"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("missing or duplicate runtime cells")
    if any(r["prompt_tokens"] != len(examples[r["id"]]["prompt_ids"]) or
           r["cap"] != examples[r["id"]]["cap"] or
           len(r["generated_ids"]) > r["cap"] or
           not 0 < r["ttft_seconds"] <= r["wall_seconds"] or
           not 0 <= r["accepted_tokens"] <= r["proposed_tokens"] for r in raw["rows"]):
        raise ValueError("invalid generation/timing record")
    for row in raw["rows"]:
        if any(not math.isfinite(row[key]) or row[key] <= 0
               for key in ("wall_seconds", "ttft_seconds", "decode_seconds")):
            raise ValueError("nonfinite or nonpositive timing")
        if not math.isclose(row["decode_seconds"] + row["ttft_seconds"], row["wall_seconds"], abs_tol=1e-9):
            raise ValueError("inconsistent timing components")
        if not 0 < row["output_tokens"] <= len(row["generated_ids"]) <= row["cap"]:
            raise ValueError("inconsistent token counts")
        if row["decode_steps"] != len(row["generated_ids"]) - 1:
            raise ValueError("inconsistent decoder step count")
        if examples[row["id"]].get("stop_eos", True) and not row["stopped_eos"] and len(row["generated_ids"]) != row["cap"]:
            raise ValueError("generation stopped before EOS or cap")
    return raw, producer, examples


def summarize(raw, examples):
    groups = defaultdict(list)
    for row in raw["rows"]:
        example = examples[row["id"]]
        groups[(row["draft"], "all")].append(row)
        groups[(row["draft"], f"{example['kind']}_{example['target_prompt_tokens']}")].append(row)
    records = []
    for (draft, workload), rows in sorted(groups.items()):
        seconds = sum(r["wall_seconds"] for r in rows)
        tokens = sum(r["output_tokens"] for r in rows)
        decode_seconds = sum(r["decode_seconds"] for r in rows)
        # EOS is included in decoder step count, separately labeled.
        records.append({"draft": draft, "workload": workload, "requests": len(rows),
                        "total_output_tokens": tokens, "total_seconds": seconds,
                        "end_to_end_output_tokens_per_second": tokens / seconds,
                        "mean_request_seconds": seconds / len(rows),
                        "median_request_seconds": statistics.median(r["wall_seconds"] for r in rows),
                        "decode_steps_per_second_including_eos": sum(r["decode_steps"] for r in rows) / decode_seconds,
                        "mean_output_tokens": tokens / len(rows),
                        "mean_ttft_seconds": sum(r["ttft_seconds"] for r in rows) / len(rows),
                        "proposed_tokens": sum(r["proposed_tokens"] for r in rows),
                        "accepted_tokens": sum(r["accepted_tokens"] for r in rows)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("analysis output is immutable")
    records = []
    for root in args.roots:
        raw, producer, examples = audit(root)
        unique = {}
        for row in raw["rows"]:
            key = (row["id"], row["text"])
            unique.setdefault(key, {"id": row["id"], "text": row["text"], "drafts": [],
                                    "facts": [f["reference_fact"] for f in examples[row["id"]]["fragments"]],
                                    "diagnostic": fact_diagnostic(row["text"], examples[row["id"]])})
            if row["draft"] not in unique[key]["drafts"]:
                unique[key]["drafts"].append(row["draft"])
        records.append({"root": str(root), "raw_sha256": sha256(root / "raw.json"),
                        "model_sha256": producer["model_sha256"], "profile": raw["profile"],
                        "model_weight_bytes": raw["model_weight_bytes"], "resident_increment_bytes": raw["resident_increment_bytes"],
                        "summary": summarize(raw, examples), "outputs_for_review": list(unique.values())})
    result = {"schema": "nal.llama-summary-analysis.v1", "status": "integrity_passed",
              "quality_status": "requires_source_grounded_review", "runs": records}
    write_json(args.output, result)
    for r in records:
        print(json.dumps({"root": r["root"], "summary": r["summary"]}), flush=True)


if __name__ == "__main__":
    main()
