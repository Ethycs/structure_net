"""Curate training labels using first valid judgments, never regrading failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, TEACHER, freeze, verify
from experiments.structure_net.ncm_paper_grade import grade
from experiments.structure_net.ncm_recipe_teacher import verify_public

NOTE = Path("docs/07 - Status Reports/2026-10-22_ncm-paper-recovery-v2-plan.md")
OLD = ROOT / "20261022_teacher_audit_v1"
CURATED = ROOT / "20261022_curated_v2"


def producing(root, stage, inputs):
    p = freeze(root, stage, inputs, {"workers": 8, "selector": q.JUDGES["secondary"],
        "rubric": q.RUBRIC, "minimum_selected": 1024, "valid_failure_retry": False})
    p["hypothesis_id"] = "ncm-paper-regime-recovery-v2"
    for source in (Path(__file__), NOTE):
        p["source_hashes"][str(source)] = file_hash(source)
        (root / "producer_source" / ("curation_"+source.name)).write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    for name in ("rows", "attempts"):
        (root / name).mkdir()
    return p


def packet(case):
    return q.packet(case["id"], case["source"], [{"candidate_id": "teacher-label", "text": case["target"]}])


def main(command):
    import tiktoken
    cases = q.read(TEACHER / "train.json")
    old_rows = {path.stem: path for path in (OLD / "rows").glob("*.json")}
    if command == "pilot":
        cases = sorted([c for c in cases if c["id"] in old_rows], key=lambda c: c["id"])[:8]
    verify_public(cases)
    if command == "audit":
        if q.read(CURATED / "result.json")["status"] != "passed":
            raise ValueError("curation gate required")
        selected = sorted(q.read(CURATED / "train.json"), key=lambda c: text_hash("curated-independent-audit-v2:"+c["id"]))[:32]
        selected = [{**c, "text": c["target"]} for c in selected]
        result = grade(selected, ROOT / "20261022_curated_audit_v2", [CURATED / "train.json"], "primary")
        if result["status"] != "completed" or result["passed"] < 31:
            raise RuntimeError("curated independent-of-selector audit failed")
        return
    root = ROOT / "20261022_curation_pilot_v2" if command == "pilot" else CURATED
    if command != "pilot" and q.read(ROOT / "20261022_curation_pilot_v2/result.json")["status"] != "passed":
        raise ValueError("eight-worker secondary concurrency pilot required")
    inputs = [TEACHER / "train.json", OLD / "result.json", *old_rows.values(), *list((OLD / "attempts").glob("*.json"))]
    p = producing(root, "systems_concurrency_only_no_quality_reuse" if command == "pilot" else "training_label_curation", inputs)
    rows, pending = {}, []
    for case in cases:
        original = old_rows.get(case["id"])
        if command != "pilot" and original is not None:
            row = q.read(original)
            if row["status"] != "valid":
                raise ValueError("unexpected invalid prior judgment; separate recovery needed")
            raw = q.read(row["attempts"][-1])
            if json.loads(raw["request"]["messages"][-1]["content"]) != packet(case):
                raise ValueError("prior grading input mismatch")
            if q.parse(raw["response"]["choices"][0]["message"]["content"], packet(case)) != row["evaluations"]:
                raise ValueError("prior decision mismatch")
            row = {**row, "reused_from": str(original), "reused_sha256": file_hash(original)}
            rows[case["id"]] = row
            write_json(root / "rows" / (case["id"]+".json"), row)
        else:
            pending.append(case)
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(q.request, packet(case), "secondary", root) for case in pending]
        for future in as_completed(jobs):
            row = future.result()
            rows[row["case_id"]] = row
            write_json(root / "rows" / (row["case_id"]+".json"), row)
            if len(rows) % 32 == 0:
                print(json.dumps({"stage": "curation", "completed": len(rows), "total": len(cases)}), flush=True)
                write_json(root / "progress.json", {"completed": len(rows), "total": len(cases)})
    enc = tiktoken.get_encoding("cl100k_base")
    accepted, rejected = [], []
    for case in cases:
        row = rows[case["id"]]
        passed = row["status"] == "valid" and q.passes(row["evaluations"][0], case["target"],
            len(enc.encode(case["target"], disallowed_special=())))
        (accepted if passed else rejected).append(case)
    invalid = sum(row["status"] != "valid" for row in rows.values())
    verify(p)
    if command != "pilot":
        write_json(root / "train.json", accepted)
        write_json(root / "rejected.json", rejected)
    result = {"status": "passed" if (invalid == 0 if command == "pilot" else len(accepted) >= 1024) else "failed",
        "cases": len(cases), "accepted": len(accepted), "rejected": len(rejected), "invalid": invalid,
        "reused_first_valid": 0 if command == "pilot" else len(old_rows),
        "retries": sum(len(row["attempts"]) > 1 for row in rows.values()),
        "producer_fingerprint": p["fingerprint"], "label": p["stage"],
        "heldout_accuracy": None, "systems_decisions_used_for_curation": False}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "curate", "audit"])
    main(parser.parse_args().command)
