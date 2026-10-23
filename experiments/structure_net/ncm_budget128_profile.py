"""Prospective128-token evaluation profile; preserve all old32-word judgments."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_label_quality as labels
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-budget128-evaluation-amendment.md")
OLD_CALIBRATION = Path("data/experiments/ncm_memory/20261022_accuracy_v2")
OLD_PHRASE = "brief approximately 32-word budget."
NEW_PHRASE = ("summary budget of at most128 stored tokens. There is no32-word target. "
              "Do not penalize a summary merely for using more than32 words within "
              "that128-token cap; assess whether its selected details are useful and relevant.")
if q.RUBRIC.count(OLD_PHRASE) != 1:
    raise ValueError("inherited rubric changed; budget amendment needs review")
RUBRIC = q.RUBRIC.replace(OLD_PHRASE, NEW_PHRASE)
LONG_GOOD = ("The passage suggests coffee options inside and near LAX. Inside the airport, "
    "it lists The Coffee Bean & Tea Leaf and Starbucks across several terminals, plus Peet's "
    "Coffee in Terminal 2 and La Colombe in Terminal 5. Nearby options include Menotti's "
    "Coffee Stop, about 1.5 miles away, and The Daily Dose, about 2 miles away. Menotti's "
    "and The Daily Dose are recommended for coffee while working on a laptop.")


def install():
    q.RUBRIC = RUBRIC
    q.original.RUBRIC = RUBRIC  # q.request executes in the original module globals.


def calibration():
    import copy
    packets = copy.deepcopy(q.read(OLD_CALIBRATION / "calibration_packets.json"))
    expected = copy.deepcopy(q.read(OLD_CALIBRATION / "calibration_expected.json"))
    long_bad = LONG_GOOD.replace("Peet's Coffee in Terminal 2", "Peet's Coffee in Terminal 9")
    # Keep every original candidate, and add the explicit budget-boundary pair.
    extra = {"case_id": "calibration-budget128-long", "source_lines": packets[0]["source_lines"],
             "candidates": [{"candidate_id": "budget128-long-supported", "text": LONG_GOOD},
                            {"candidate_id": "budget128-long-contradiction", "text": long_bad}]}
    expected["budget128-long-supported"] = {"expected_pass": True, "kind": "long_supported", "text": LONG_GOOD}
    expected["budget128-long-contradiction"] = {"expected_pass": False, "kind": "long_contradiction", "text": long_bad}
    return packets + [extra], expected


def qualify(role):
    labels.dependencies()
    import tiktoken
    packets, expected = calibration()
    enc = tiktoken.get_encoding("cl100k_base")
    if len(expected) != 50 or len(LONG_GOOD.split()) <= 32 or len(enc.encode(LONG_GOOD)) > 128:
        raise ValueError("fixed50/budget-boundary controls differ")
    # Verify the exact source bytes against the public corpus used by the already
    # qualified original profile before sending any calibration material.
    from experiments.structure_net.ncm_frozen_body_public_data import PUBLIC_FILE
    corpus = PUBLIC_FILE.read_text()
    public = json.loads(corpus)
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for v in value.values(): yield from strings(v)
        elif isinstance(value, list):
            for v in value: yield from strings(v)
    all_text = set(strings(public))
    if any("\n".join(line["text"] for line in p["source_lines"]) not in all_text for p in packets):
        raise ValueError("calibration source bytes not present in public corpus")
    root = ROOT / ("20261023_budget128_judge_" + role + "_v1")
    inputs = [OLD_CALIBRATION / n for n in ("calibration_packets.json", "calibration_expected.json")]
    p = freeze(root, "judge_profile_qualification_not_model_quality", inputs + [Path(__file__), PLAN, PUBLIC_FILE],
        {"judge": q.JUDGES[role], "rubric": RUBRIC, "controls": 50, "workers": 4,
         "max_attempts": 3, "semantic_retries": False, "independent_human_gold": False})
    for path in (Path(__file__), PLAN):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("budget128_" + path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    install()
    (root / "attempts").mkdir()
    (root / "rows").mkdir()
    write_json(root / "packets.json", packets)
    write_json(root / "expected.json", expected)
    rows = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(q.request, packet, role, root) for packet in packets]
        for future in as_completed(futures):
            row = future.result()
            rows[row["case_id"]] = row
            write_json(root / "rows" / (row["case_id"] + ".json"), row)
    scores = []
    for packet in packets:
        row = rows[packet["case_id"]]
        grades = {g["candidate_id"]: g for g in row.get("evaluations", [])}
        for candidate in packet["candidates"]:
            name = candidate["candidate_id"]
            n = len(enc.encode(candidate["text"], disallowed_special=()))
            actual = q.passes(grades[name], candidate["text"], n) if row["status"] == "valid" else None
            scores.append({"candidate_id": name, "expected": expected[name]["expected_pass"],
                "actual": actual, "correct": actual == expected[name]["expected_pass"] if actual is not None else False,
                "tokens": n, "evaluation": grades.get(name)})
    valid = sum(row["status"] == "valid" for row in rows.values())
    correct = sum(s["correct"] for s in scores)
    result = {"status": "passed" if valid == len(packets) and correct == 50 else "failed_qualification",
        "cases": len(packets), "valid_requests": valid, "controls": 50, "correct": correct,
        "judge": q.JUDGES[role], "profile": "budget128-v1", "NCM_success_established": False,
        "producer_fingerprint": p["fingerprint"], "task_accuracy": None}
    write_json(root / "scores.json", scores)
    verify(p)
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("role", choices=["primary", "secondary"])
    qualify(parser.parse_args().role)
