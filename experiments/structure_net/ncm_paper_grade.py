"""Independent single-candidate grading, preserving negative and invalid results."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public


def grade(cases, destination, inputs, role="primary", systems_only=False):
    import tiktoken
    verify_public(cases)
    qualification = Path("data/experiments/ncm_memory/20261022_accuracy_v2") / ("calibration_"+role) / "result.json"
    if q.read(qualification)["status"] != "passed":
        raise ValueError("judge qualification required")
    p = freeze(destination, "source_grounded_whole_summary_accuracy", [*inputs, qualification],
        {"cases": len(cases), "judge": q.JUDGES[role], "rubric": q.RUBRIC,
         "workers": 4, "candidate_validation": "one candidate per request", "systems_only": systems_only})
    p["source_hashes"][str(Path(__file__))] = file_hash(__file__)
    (destination / "producer_source" / "grade_source.py").write_bytes(Path(__file__).read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    for name in ("attempts", "rows"):
        (destination / name).mkdir()
    packets = [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["text"]}]) for r in cases]
    write_json(destination / "packets.json", packets)
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(q.request, packet, role, destination): packet["case_id"] for packet in packets}
        for future in as_completed(jobs):
            row = future.result()
            results[row["case_id"]] = row
            write_json(destination / "rows" / (row["case_id"]+".json"), row)
            if len(results) % 16 == 0:
                print(json.dumps({"stage": "judge", "completed": len(results), "total": len(cases)}), flush=True)
    enc = tiktoken.get_encoding("cl100k_base")
    scores = []
    for case in cases:
        row = results[case["id"]]
        n = len(enc.encode(case["text"], disallowed_special=()))
        valid = row["status"] == "valid"
        passed = q.passes(row["evaluations"][0], case["text"], n) if valid else None
        scores.append({"id": case["id"], "source_id": case["source_id"], "text": case["text"],
            "valid": valid, "passed": passed, "tokens": n,
            "evaluations": row.get("evaluations"), "attempts": row["attempts"]})
    n, passed = len(cases), sum(s["passed"] is True for s in scores)
    invalid = sum(not s["valid"] for s in scores)
    result = {"status": "completed" if invalid == 0 else "invalid_judgments",
        "cases": n, "passed": passed, "valid": n-invalid, "invalid": invalid,
        "task_accuracy": passed/n if invalid == 0 else None,
        "all_case_pass_lower_bound": passed/n, "all_case_pass_upper_bound": (passed+invalid)/n,
        "wilson_95": q.wilson(passed, n) if invalid == 0 else None,
        "judge": q.JUDGES[role], "producer_fingerprint": p["fingerprint"],
        "retried": sum(len(r["attempts"]) > 1 for r in results.values()), "systems_only": systems_only,
        "speed_tested": False, "human_gold": False}
    verify(p)
    write_json(destination / "case_scores.json", scores)
    write_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main(args):
    if args.kind == "baseline":
        source = ROOT / "20261022_baseline_v1/outputs.json"
        cases = q.read(source)
        destination = ROOT / ("20261022_baseline_grade_"+args.role+"_v1")
    elif args.kind == "oracle":
        source = ROOT / "20261022_codec_control_v1/outputs.json"
        cases = sorted(q.read(source), key=lambda r: text_hash("paper-label-audit:"+r["id"]))[:32]
        destination = ROOT / ("20261022_oracle_grade_"+args.role+"_v1")
    else:
        source = args.outputs
        cases = q.read(source)
        destination = args.output
    grade(cases, destination, [source], args.role)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("kind", choices=["baseline", "oracle", "outputs"])
    parser.add_argument("--role", choices=q.JUDGES, default="primary")
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
