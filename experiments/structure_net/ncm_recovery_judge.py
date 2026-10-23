"""Qualify a gateway-listed recovery judge on the unchanged public controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public


def main(args):
    import tiktoken
    base = Path("data/experiments/ncm_memory/20261022_accuracy_v2")
    cases = q.read(q.DISCOVERY)
    verify_public(cases)
    packets = q.read(base / "calibration_packets.json")
    expected = q.read(base / "calibration_expected.json")
    if len(packets) != 12 or len(expected) != 48:
        raise ValueError("full existing48-control qualification required")
    for case, packet in zip(cases, packets):
        if [r["text"] for r in packet["source_lines"]] != case["source"].split("\n"):
            raise ValueError("calibration/public-source mismatch")
    p = freeze(args.output, "recovery_judge_qualification", [q.DISCOVERY,
        base / "calibration_packets.json", base / "calibration_expected.json"],
        {"model": args.model, "role": "recovery", "rubric": q.RUBRIC, "controls": 48,
         "workers": 4, "required_positive": 24, "required_negative": 24})
    p["hypothesis_id"] = "ncm-judge-recovery-v1"
    p["source_hashes"][str(Path(__file__))] = file_hash(__file__)
    (args.output / "producer_source" / "recovery_judge.py").write_bytes(Path(__file__).read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(args.output / "producer.json", p)
    q.JUDGES["recovery"] = args.model
    for name in ("attempts", "rows"):
        (args.output / name).mkdir()
    for packet in packets:
        packet["case_id"] = "recovery-"+packet["case_id"]
    first = q.request(packets[0], "recovery", args.output)
    write_json(args.output / "rows" / (first["case_id"]+".json"), first)
    results = [first]
    if first["status"] == "valid":
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(q.request, packet, "recovery", args.output) for packet in packets[1:]]
            for future in as_completed(jobs):
                row = future.result()
                results.append(row)
                write_json(args.output / "rows" / (row["case_id"]+".json"), row)
    enc = tiktoken.get_encoding("cl100k_base")
    positive = negative = 0
    decisions = []
    for row in results:
        if row["status"] != "valid":
            continue
        for grade in row["evaluations"]:
            target = expected[grade["candidate_id"]]
            passed = q.passes(grade, target["text"], len(enc.encode(target["text"], disallowed_special=())))
            if passed == target["expected_pass"]:
                positive += bool(target["expected_pass"])
                negative += not target["expected_pass"]
            decisions.append({"candidate_id": grade["candidate_id"], "passed": passed,
                              "expected_pass": target["expected_pass"]})
    valid = len(results) == 12 and all(row["status"] == "valid" for row in results)
    result = {"status": "passed" if valid and positive == 24 and negative == 24 else "qualification_failed",
        "model": args.model, "valid_packets": sum(row["status"] == "valid" for row in results),
        "packets": 12, "positive_correct": positive, "negative_correct": negative,
        "controls": 48, "producer_fingerprint": p["fingerprint"], "task_accuracy": None,
        "human_gold": False, "decisions": decisions}
    verify(p)
    write_json(args.output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
