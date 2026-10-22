"""Recover conservative failures without changing judge decisions or requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import re
import shutil

from experiments.structure_net import ncm_memory_accuracy as original
from experiments.structure_net.ncm_memory_accuracy import *

PRIOR = Path("data/experiments/ncm_memory/20261022_accuracy_v1")
AMENDMENT = Path("docs/07 - Status Reports/2026-10-22_ncm-memory-accuracy-parser-recovery.md")
_strict_parse = original.parse
_freeze = original.freeze
_execute = original.execute


def parse(response, p):
    stripped = response.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    obj = json.loads(stripped)
    conservative = set()
    for grade in obj.get("evaluations", []):
        claims = grade.get("claims")
        if grade.get("faithful") is False and isinstance(claims, list) and claims and all(
                c.get("verdict") == "supported" for c in claims):
            conservative.add(grade.get("candidate_id"))
            grade["faithful"] = True  # Validate claim/citation structure, then restore exactly.
    grades = _strict_parse(json.dumps(obj), p)
    for grade in grades:
        if grade["candidate_id"] in conservative:
            grade["faithful"] = False
    return grades


def prepare(args):
    args.root.mkdir(parents=True, exist_ok=False)
    for name in ("packets.json", "cases.json", "candidate_map.json", "calibration_packets.json",
                 "calibration_expected.json", "inputs.json"):
        shutil.copyfile(PRIOR / name, args.root / name)
    inputs = [p for p in PRIOR.rglob("*.json") if "producer_source" not in p.parts]
    write_json(args.root / "recovery_sources.json", {str(p): file_hash(p) for p in inputs})
    write_json(args.root / "recovery_protocol.json", {
        "amendment_sha256": file_hash(AMENDMENT), "prior_root": str(PRIOR),
        "decision_changes": False, "rubric_changes": False,
        "rule": "First valid prior response under conservative-failure validation wins; only missing packets requested."})
    print(json.dumps({"status": "prepared_parser_recovery", "prior_files": len(inputs)}), flush=True)


def freeze(root, stage, base, role):
    producer = _freeze(root, stage, base, role)
    paths = [Path(__file__), AMENDMENT,
             Path("experiments/neural_architecture_lab/finalize_ncm_memory_accuracy.py"),
             Path("experiments/neural_architecture_lab/finalize_ncm_memory_accuracy_v2.py")]
    for i, path in enumerate(paths):
        producer["source_hashes"][str(path)] = file_hash(path)
        shutil.copyfile(path, root / "producer_source" / f"v2_{i}_{path.name}")
    producer["parser_protocol"] = "unfaithful_is_conservative_failure"
    if role == "primary" and stage == "blinded_whole_summary_accuracy":
        pilot = read(base / "primary_concurrency" / "result.json")
        producer["workers"] = 8 if pilot["status"] == "passed" else 4
    if role == "secondary" and stage == "blinded_whole_summary_accuracy":
        producer["workers"] = 4
    if stage == "systems_concurrency_only_not_quality_evidence":
        producer["workers"] = 8 if role == "primary" else 4
    producer["fingerprint"] = digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(root / "producer.json", producer)
    return producer


def recover(p, role, root):
    manifest = read(root.parent / "recovery_sources.json")
    prior = PRIOR / root.name / "attempts"
    files = sorted(prior.glob(p["case_id"]+"_*.json")) if prior.exists() else []
    tried = []
    for path in files:
        if manifest.get(str(path)) != file_hash(path):
            raise ValueError("prior raw response changed")
        raw = read(path)
        corrected = copy.deepcopy(raw)
        corrected.update(imported_from=str(path), origin_sha256=file_hash(path),
                         original_valid=raw["valid"], original_error=raw.get("error"))
        try:
            if raw.get("http_status") != 200 or raw["response"].get("model") != JUDGES[role]:
                raise ValueError("prior transport invalid")
            evaluations = parse(raw["response"]["choices"][0]["message"]["content"], p)
            corrected["valid"] = True
        except Exception:
            corrected["valid"] = False
        dest = root / "attempts" / path.name
        corrected.pop("error", None)
        write_json(dest, corrected)
        tried.append(str(dest))
        if corrected["valid"]:
            return {"case_id": p["case_id"], "packet_sha256": digest(p), "status": "valid",
                    "judge": JUDGES[role], "evaluations": evaluations, "attempts": tried,
                    "recovered": True, "later_prior_replies_ignored": [str(x) for x in files[len(tried):]]}
    if tried:
        raise ValueError("unrecoverable prior responses need a separate infrastructure decision")
    return None


def execute(packets, role, root, serial_stop=False):
    recovered, missing = [], []
    for p in packets:
        row = recover(p, role, root)
        if row is None:
            missing.append(p)
        else:
            write_json(root / "rows" / (row["case_id"]+".json"), row)
            recovered.append(row)
    print(json.dumps({"stage": root.name, "recovered": len(recovered), "new_packets": len(missing)}), flush=True)
    if root.name in ("judge_primary", "judge_secondary"):
        pilot = read(root.parent / (role+"_concurrency") / "result.json")
        if role == "secondary" and pilot["status"] != "passed":
            raise ValueError("secondary concurrency pilot required")
        workers = 8 if role == "primary" and pilot["status"] == "passed" else 4
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(original.request, p, role, root) for p in missing]
            for f in as_completed(jobs):
                row = f.result()
                write_json(root / "rows" / (row["case_id"]+".json"), row)
                results.append(row)
                if len(results) % 8 == 0:
                    print(json.dumps({"stage": root.name, "completed": len(results), "total": len(missing)}), flush=True)
    else:
        results = _execute(missing, role, root, serial_stop=serial_stop) if missing else []
    return sorted(recovered+results, key=lambda r: r["case_id"])


def concurrency(args):
    if read(args.root / ("calibration_"+args.role) / "result.json")["status"] != "passed":
        raise ValueError("judge qualification required")
    root = args.root / (args.role+"_concurrency")
    producer = freeze(root, "systems_concurrency_only_not_quality_evidence", args.root, args.role)
    workers = 8 if args.role == "primary" else 4
    packets = copy.deepcopy(read(args.root / "calibration_packets.json")[:workers])
    for p in packets:
        p["case_id"] = "systems-"+p["case_id"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda p: original.request(p, args.role, root), packets))
    original.finish(root, producer, results, status="passed" if all(r["status"] == "valid" for r in results) else "failed",
                    label="systems_concurrency_only_not_quality_evidence", workers=workers)


original.parse = parse
original.freeze = freeze
original.execute = execute


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "calibrate", "grade", "concurrency"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=JUDGES, default="primary")
    args = parser.parse_args()
    if args.command in ("prepare", "concurrency"):
        globals()[args.command](args)
    else:
        getattr(original, args.command)(args)


if __name__ == "__main__":
    main()
