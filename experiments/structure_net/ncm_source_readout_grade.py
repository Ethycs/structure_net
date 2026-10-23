"""Preserve the first judgment for every exact readout packet and judge role."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net import ncm_source_readout as readout
from experiments.structure_net.ncm_recipe_teacher import verify_public

AMENDMENT = Path("docs/07 - Status Reports/2026-10-23_ncm-readout-first-judgment-amendment.md")
DEST = ROOT / "20261023_source_readout_grading_v2"
PILOT = ROOT / "20261023_source_readout_judge_pilot_v1"


def packet_key(packet, role):
    return q.digest({"packet": packet, "rubric": q.RUBRIC, "role": role, "model": q.JUDGES[role]})


def remember(cache, key, entry):
    if key in cache and cache[key] != entry:
        raise ValueError("first judgment cannot be overwritten")
    cache.setdefault(key, entry)


def audit_row(packet, role, row):
    if row["case_id"] != packet["case_id"] or row["packet_sha256"] != q.digest(packet) or row["judge"] != q.JUDGES[role]:
        raise ValueError("cached judgment packet/role mismatch")
    attempts = [q.read(path) for path in row["attempts"]]
    if not 1 <= len(attempts) <= 3:
        raise ValueError("invalid attempt count")
    valid = []
    for raw in attempts:
        payload = raw["request"]
        if (payload["model"] != q.JUDGES[role] or payload["messages"][0]["content"] != q.RUBRIC or
                json.loads(payload["messages"][1]["content"]) != packet or
                payload["temperature"] != 0 or payload["max_tokens"] != 4096):
            raise ValueError("raw request identity changed")
        parsed = None
        try:
            if raw["http_status"] == 200 and raw["response"].get("model") == q.JUDGES[role]:
                parsed = q.parse(raw["response"]["choices"][0]["message"]["content"], packet)
        except (ValueError, KeyError, TypeError, IndexError):
            pass
        if bool(raw["valid"]) != (parsed is not None):
            raise ValueError("cached raw validity differs from parser replay")
        valid.append(parsed)
    if row["status"] == "valid":
        if valid[-1] != row["evaluations"] or valid[-1] is None or any(v is not None for v in valid[:-1]):
            raise ValueError("first valid judgment was not preserved")
    elif row["status"] != "invalid" or len(attempts) != 3 or any(v is not None for v in valid):
        raise ValueError("exhausted-invalid judgment was not preserved")


def freeze(root, stage, inputs, config):
    p = readout.freeze(root, stage, inputs, config)
    for path in (Path(__file__), AMENDMENT):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("first_judgment_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def grade_condition(cases, condition, role, cache, inputs):
    import tiktoken
    verify_public(cases)
    qualification = Path("data/experiments/ncm_memory/20261022_accuracy_v2") / ("calibration_"+role) / "result.json"
    if q.read(qualification)["status"] != "passed":
        raise ValueError("qualified judge required")
    root = DEST / (role+"_"+condition)
    p = freeze(root, "whole_summary_first_judgment_comparison", inputs+[qualification],
        {"role": role, "cases": len(cases), "condition": condition, "workers": 4,
         "max_attempts": 3, "first_valid_preserved_across_conditions": True, "one_candidate_per_request": True})
    packets = [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["text"]}]) for r in cases]
    write_json(root / "packets.json", packets)
    reused = sum(packet_key(packet, role) in cache for packet in packets)
    missing = [packet for packet in packets if packet_key(packet, role) not in cache]
    if len({packet_key(packet, role) for packet in missing}) != len(missing):
        raise ValueError("duplicate uncached packet within a condition")
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {}
        for packet in missing:
            key = packet_key(packet, role)
            request_root = DEST / "requests" / role / key
            (request_root / "attempts").mkdir(parents=True, exist_ok=False)
            jobs[pool.submit(q.request, packet, role, request_root)] = (key, packet, request_root)
        for i, future in enumerate(as_completed(jobs), 1):
            key, packet, request_root = jobs[future]
            row = future.result()
            audit_row(packet, role, row)
            write_json(request_root / "row.json", row)
            remember(cache, key, {"packet": packet, "role": role, "row": row, "origin": str(request_root)})
            write_json(DEST / "judgment_cache.json", cache)
            if i % 16 == 0 or i == len(missing):
                print(json.dumps({"condition": condition, "role": role, "new_completed": i,
                                  "new_total": len(missing), "reused": reused}), flush=True)
    enc = tiktoken.get_encoding("cl100k_base")
    scores = []
    for case, packet in zip(cases, packets):
        entry = cache[packet_key(packet, role)]
        row = entry["row"]
        audit_row(packet, role, row)
        n = len(enc.encode(case["text"], disallowed_special=()))
        valid = row["status"] == "valid"
        passed = q.passes(row["evaluations"][0], case["text"], n) if valid else None
        scores.append({"id": case["id"], "source_id": case["source_id"], "text": case["text"],
            "tokens": n, "valid": valid, "passed": passed, "evaluations": row.get("evaluations"),
            "attempts": row["attempts"], "packet_key": packet_key(packet, role), "first_origin": entry["origin"]})
    passed = sum(row["passed"] is True for row in scores)
    invalid = sum(not row["valid"] for row in scores)
    result = {"status": "completed" if not invalid else "invalid_judgments", "cases": len(cases),
        "passed": passed, "valid": len(cases)-invalid, "invalid": invalid,
        "task_accuracy": passed/len(cases) if not invalid else None,
        "all_case_pass_lower_bound": passed/len(cases), "all_case_pass_upper_bound": (passed+invalid)/len(cases),
        "wilson_95": q.wilson(passed, len(cases)) if not invalid else None,
        "new_requests": len(missing), "reused_first_judgments": reused, "judge": q.JUDGES[role],
        "producer_fingerprint": p["fingerprint"], "NCM_trained": False, "speed_tested": False}
    verify(p)
    write_json(root / "case_scores.json", scores); write_json(root / "result.json", result)
    print(json.dumps({"condition": condition, "role": role, **result}), flush=True)
    return result


def main():
    readout.dependencies()
    terminal = q.read(readout.AUDIT / "worker_terminal_receipt.json")
    if terminal["status"] != "observed_worker_terminal" or terminal["exit_code"] != 0:
        raise ValueError("previous API worker must be authoritatively terminal")
    pilot = q.read(PILOT / "result.json")
    if pilot["status"] != "completed" or pilot["valid"] != 4 or not pilot["systems_only"]:
        raise ValueError("four valid judge lifecycle outcomes required")
    verify(q.read(PILOT / "producer.json")); verify(q.read(readout.MEASURE / "producer.json"))
    for path, expected in q.read(readout.MEASURE / "result.json")["output_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("readout selection changed")
    inputs = [readout.MEASURE/n for n in ("outputs.json", "result.json", "producer.json")]
    inputs += [PILOT / "result.json", PILOT / "packets.json", readout.AUDIT / "worker_terminal_receipt.json"]
    inputs += sorted((PILOT / "rows").glob("*.json"))+sorted((PILOT / "attempts").glob("*.json"))
    p = freeze(DEST, "first_judgment_preserved_readout_comparison", inputs,
        {"conditions": readout.CONDITIONS, "development_cases": 64, "reference_cases": 20,
         "joint_pass_minimum": 61, "oracle_pass_minimum": 19, "role_separated_cache": True})
    cache = {}
    for packet in q.read(PILOT / "packets.json"):
        row = q.read(PILOT / "rows" / (packet["case_id"]+".json"))
        audit_row(packet, "secondary", row)
        remember(cache, packet_key(packet, "secondary"),
                 {"packet": packet, "role": "secondary", "row": row, "origin": str(PILOT)})
    write_json(DEST / "judgment_cache.json", cache)
    outputs = q.read(readout.MEASURE / "outputs.json")
    results = {}
    for condition in (*readout.CONDITIONS, "reference_query"):
        rows = outputs[condition]
        secondary = grade_condition(rows, condition, "secondary", cache, inputs)
        primary = None
        if condition != "reference_query" and secondary["status"] == "completed" and secondary["passed"] >= 61:
            primary = grade_condition(rows, condition, "primary", cache, inputs)
        results[condition] = {"secondary": secondary, "primary": primary}
        write_json(DEST / "progress.json", results)
    verify(p)
    result = {"status": "completed", "results": results, "unique_judgments": len(cache),
        "pilot_judgments_reused": 4, "producer_fingerprint": p["fingerprint"],
        "NCM_trained": False, "NCM_success_established": False, "confirmed": False,
        "underpowered": True, "final_test_opened": False, "speed_tested": False}
    write_json(DEST / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["grade"])
    parser.parse_args()
    main()
