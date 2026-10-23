"""Local, source-grounded judge with preserved raw replies and strict calibration."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

MODEL = "ncm-qwen3-8b-q6k"
URL = "http://127.0.0.1:18081/v1/chat/completions"
BASE = Path("data/experiments/ncm_local_judge")
PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-local-judge-recovery-plan.md")
PLACEMENT = Path("docs/07 - Status Reports/2026-10-22_ncm-local-judge-gpu2-amendment.md")
JSON_NOTE = Path("docs/07 - Status Reports/2026-10-22_ncm-local-judge-json-amendment.md")


def response_format(packet):
    claim = {"type": "object", "additionalProperties": False,
        "properties": {"claim": {"type": "string", "minLength": 1},
            "verdict": {"type": "string", "enum": ["supported", "contradicted", "insufficient evidence"]},
            "evidence_ids": {"type": "array", "items": {"type": "string",
                "enum": [line["line_id"] for line in packet["source_lines"]]}}},
        "required": ["claim", "verdict", "evidence_ids"]}
    evaluation = {"type": "object", "additionalProperties": False,
        "properties": {"candidate_id": {"type": "string", "enum": [x["candidate_id"] for x in packet["candidates"]]},
            **{key: {"type": "boolean"} for key in ("faithful", "salient", "readable")},
            "claims": {"type": "array", "items": claim}, "reason": {"type": "string", "minLength": 1}},
        "required": ["candidate_id", "faithful", "salient", "readable", "claims", "reason"]}
    schema = {"type": "object", "additionalProperties": False,
        "properties": {"evaluations": {"type": "array", "items": evaluation,
            "minItems": len(packet["candidates"]), "maxItems": len(packet["candidates"])}},
        "required": ["evaluations"]}
    return {"type": "json_schema", "json_schema": {"name": "summary_grades", "strict": True, "schema": schema}}


def request(packet, destination):
    import requests
    payload = {"model": MODEL, "temperature": 0, "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False}, "response_format": response_format(packet),
        "messages": [{"role": "system", "content": q.RUBRIC},
                     {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}]}
    session = requests.Session()
    session.trust_env = False  # Loopback requests never use an external proxy.
    attempts = []
    for attempt in range(3):
        path = destination / "attempts" / f"{packet['case_id']}_{attempt}.json"
        if path.exists():
            raise ValueError("local-judge attempt overwrite")
        raw = {"request": payload, "request_sha256": q.digest(payload), "valid": False}
        start = time.monotonic()
        try:
            response = session.post(URL, json=payload, timeout=300)
            raw.update(http_status=response.status_code, response=response.json())
            response.raise_for_status()
            if raw["response"].get("model") != MODEL:
                raise ValueError("local judge alias mismatch")
            grades = q.parse(raw["response"]["choices"][0]["message"]["content"], packet)
            raw["valid"] = True
        except Exception as error:
            raw["error_type"] = type(error).__name__
            if isinstance(error, ValueError):
                raw["error"] = str(error)[:200]
        raw["seconds"] = time.monotonic()-start
        write_json(path, raw)
        attempts.append(str(path))
        if raw["valid"]:
            return {"case_id": packet["case_id"], "status": "valid", "evaluations": grades, "attempts": attempts}
    return {"case_id": packet["case_id"], "status": "invalid", "attempts": attempts}


def producing(destination, stage, inputs):
    server = BASE / "20261022_server_gpu2_v1"
    startup = q.read(server / "startup_result.json")
    launch = q.read(server / "producer.json")
    pid = startup["pid"]
    command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
    environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    gpu_uuid = subprocess.check_output(["nvidia-smi", "--id=2", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    allocations = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], text=True)
    allocation = [line.split(", ") for line in allocations.splitlines() if line.startswith(str(pid)+", ")]
    if (startup["status"] != "ready" or command != launch["command"]
        or b"CUDA_VISIBLE_DEVICES=2" not in environment
        or launch["gpu_layers"] != 28 or len(allocation) != 1
        or allocation[0][1] != gpu_uuid or int(allocation[0][2]) < 1024):
        raise ValueError("fixed GPU launch/allocation was not established; CPU-only execution is prohibited")
    p = freeze(destination, stage, inputs, {"model": MODEL, "endpoint": URL, "workers": 1,
        "rubric": q.RUBRIC, "max_tokens": 2048, "max_attempts": 3, "timeout": 300})
    p["requested_gpu_layers"] = 28
    p["verified_gpu_allocation_mib"] = int(allocation[0][2])
    p["verified_gpu_uuid"] = gpu_uuid
    p["hypothesis_id"] = "ncm-local-judge-qualification-v1"
    for source in (Path(__file__), PLAN, PLACEMENT, JSON_NOTE):
        p["source_hashes"][str(source)] = file_hash(source)
        (destination / "producer_source" / ("local_"+source.name)).write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    for name in ("attempts", "rows"):
        (destination / name).mkdir()
    return p


def execute(packets, destination, serial_first=False):
    rows = []
    if serial_first:
        first = request(packets[0], destination)
        rows.append(first)
        write_json(destination / "rows" / (first["case_id"]+".json"), first)
        if first["status"] != "valid":
            return rows
        packets = packets[1:]
    with ThreadPoolExecutor(max_workers=1) as pool:
        jobs = [pool.submit(request, packet, destination) for packet in packets]
        for future in as_completed(jobs):
            row = future.result()
            rows.append(row)
            write_json(destination / "rows" / (row["case_id"]+".json"), row)
            if len(rows) % 8 == 0:
                print(json.dumps({"stage": "local_judge", "completed": len(rows)}), flush=True)
    return rows


def main(args):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    if args.command in ("pilot", "qualify"):
        cases = q.read(q.DISCOVERY)
        verify_public(cases)
        previous = Path("data/experiments/ncm_memory/20261022_accuracy_v2")
        packets = q.read(previous / "calibration_packets.json")
        expected = q.read(previous / "calibration_expected.json")
        destination = BASE / ("20261022_pilot_gpu2_json_v1" if args.command == "pilot" else "20261022_qualification_gpu2_json_v1")
        if args.command == "pilot":
            packets = packets[:2]  # Two serial requests at the intended single-slot setting.
            for packet in packets:
                packet["case_id"] = "local-systems-"+packet["case_id"]
        else:
            if q.read(BASE / "20261022_pilot_gpu2_json_v1/result.json")["status"] != "passed":
                raise ValueError("local memory/concurrency pilot required")
        for case, packet in zip(cases, packets):
            if [line["text"] for line in packet["source_lines"]] != case["source"].split("\n"):
                raise ValueError("public calibration source mismatch")
        p = producing(destination, "systems_only_not_quality_evidence" if args.command == "pilot" else "48_control_qualification",
            [previous / "calibration_packets.json", previous / "calibration_expected.json", q.DISCOVERY,
             BASE / "20261022_download_v1/result.json", BASE / "20261022_server_gpu2_v1/producer.json"])
        rows = execute(packets, destination, serial_first=True)
        valid = len(rows) == len(packets) and all(row["status"] == "valid" for row in rows)
        positive = negative = 0
        for row in rows:
            for grade in row.get("evaluations", []):
                target = expected[grade["candidate_id"]]
                passed = q.passes(grade, target["text"], len(enc.encode(target["text"], disallowed_special=())))
                if passed == target["expected_pass"]:
                    positive += bool(target["expected_pass"])
                    negative += not target["expected_pass"]
        success = valid if args.command == "pilot" else valid and positive == 24 and negative == 24
        verify(p)
        result = {"status": "passed" if success else "failed", "stage": p["stage"], "model": MODEL,
            "packets": len(packets), "valid_packets": sum(row["status"] == "valid" for row in rows),
            "positive_correct": positive, "negative_correct": negative,
            "task_accuracy": None, "producer_fingerprint": p["fingerprint"], "human_gold": False}
    else:
        qualification = BASE / "20261022_qualification_gpu2_json_v1/result.json"
        if q.read(qualification)["status"] != "passed":
            raise ValueError("full local-judge qualification required")
        cases = q.read(args.outputs)
        verify_public(cases)
        destination = args.output
        p = producing(destination, "local_source_grounded_task_accuracy", [args.outputs, qualification,
            BASE / "20261022_server_gpu2_v1/producer.json"])
        packets = [q.packet(case["id"], case["source"], [{"candidate_id": "summary", "text": case["text"]}]) for case in cases]
        rows = {row["case_id"]: row for row in execute(packets, destination)}
        scores = []
        for case in cases:
            row = rows[case["id"]]
            valid = row["status"] == "valid"
            passed = q.passes(row["evaluations"][0], case["text"],
                len(enc.encode(case["text"], disallowed_special=()))) if valid else None
            scores.append({"id": case["id"], "valid": valid, "passed": passed, "text": case["text"],
                           "evaluations": row.get("evaluations"), "attempts": row["attempts"]})
        invalid = sum(not score["valid"] for score in scores)
        passed = sum(score["passed"] is True for score in scores)
        verify(p)
        write_json(destination / "case_scores.json", scores)
        result = {"status": "completed" if invalid == 0 else "invalid_judgments", "model": MODEL,
            "cases": len(cases), "passed": passed, "invalid": invalid,
            "task_accuracy": passed/len(cases) if invalid == 0 else None,
            "wilson_95": q.wilson(passed, len(cases)) if invalid == 0 else None,
            "human_gold": False, "producer_fingerprint": p["fingerprint"]}
    write_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "qualify", "grade"])
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
