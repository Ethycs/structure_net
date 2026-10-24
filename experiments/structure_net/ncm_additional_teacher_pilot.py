"""Generate fixed64 TRAIN-label candidates; semantic qualification is a later stage."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.structure_net import ncm_additional_training_sources as data
from experiments.structure_net import ncm_lm1b_codec as native
from experiments.structure_net.ncm_native_parent_control import SYSTEM as SINGLE_SYSTEM
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

ROOT = data.ROOT
MODULE = "experiments.structure_net.ncm_additional_teacher_pilot"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-additional-teacher-pilot-plan.md")
TEST = Path("tests/structure_net/test_ncm_additional_teacher_pilot.py")
PREP = ROOT / "20261024_additional64_teacher_prepare_v1"
PILOT = ROOT / "20261024_additional64_teacher_systems_pilot_v1"
GENERATE = ROOT / "20261024_additional64_teacher_candidates_v1"
WORKFLOW = ROOT / "20261024_additional64_teacher_workflow_v1"
HYPOTHESIS = "ncm-additional64-teacher-label-generation-v1"
MODEL = "codex_sdk/gpt-5.6-terra"
SOURCE_SHA = "3e07a05b0cc1f1409fade64335dc995f83edfead733a1e6f9922409385d6f29d"
QUOTAS = {0: 32, 1: 8, 2: 8, 3: 8, 4: 8}
SYSTEM = SINGLE_SYSTEM + "\nApply these instructions independently to each passage. Never mix passages. Return only JSON " \
    '{"summaries":[{"slot":0,"summary":"paragraph"}]}, one object for every provided integer slot. '
SYSTEM += "Return one final summary per passage; no alternatives, reasoning or evidence metadata."


def select(rows):
    selected = []
    for band, n in QUOTAS.items():
        values = sorted((r for r in rows if r["bin"] == band), key=lambda r: (text_hash(HYPOTHESIS + ":case:" + r["id"]), r["id"]))
        if len(values) < n:
            raise ValueError("complete64 teacher sample unavailable")
        selected.extend(values[:n])
    if len(selected) != 64 or len({r["source_component"] for r in selected}) != 64:
        raise ValueError("distinct complete64 TRAIN components required")
    return selected


def payload(rows):
    return {"model": MODEL, "temperature": 0, "max_tokens": 768, "messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps({"passages": [
            {"slot": i, "text": row["source"]} for i, row in enumerate(rows)]}, ensure_ascii=False)}]}


def batches(rows, encoding):
    result, current = [], []
    for row in rows:
        trial = [*current, row]
        if len(trial) > 4 or len(encoding.encode(json.dumps(payload(trial)["messages"], ensure_ascii=False), disallowed_special=())) + 16 > 6300:
            if not current:
                raise ValueError("complete input cannot fit6300 prompt tokens; clipping prohibited")
            result.append(current)
            current = [row]
        else:
            current = trial
        if len(encoding.encode(json.dumps(payload(current)["messages"], ensure_ascii=False), disallowed_special=())) + 16 > 6300:
            raise ValueError("complete input cannot fit6300 prompt tokens; clipping prohibited")
    if current:
        result.append(current)
    return result


def parse(content, n):
    obj = json.loads(content)
    if not isinstance(obj, dict) or set(obj) != {"summaries"} or not isinstance(obj["summaries"], list):
        raise ValueError("exact teacher response schema required")
    found = {}
    for row in obj["summaries"]:
        if (not isinstance(row, dict) or set(row) != {"slot", "summary"} or type(row["slot"]) is not int
                or not 0 <= row["slot"] < n or row["slot"] in found or not isinstance(row["summary"], str)):
            raise ValueError("complete unique integer slot schema required")
        found[row["slot"]] = row["summary"]
    if set(found) != set(range(n)):
        raise ValueError("all teacher slots required")
    return [found[i] for i in range(n)]


def produce(root, inputs, config):
    files = [Path(__file__), PLAN, TEST, Path(data.__file__), Path(native.__file__),
             Path("experiments/structure_net/ncm_native_parent_control.py")]
    root.mkdir(exist_ok=False)
    (root / "producer_source").mkdir()
    for i, path in enumerate(files):
        (root / "producer_source" / f"{i}_{path.name}").write_bytes(path.read_bytes())
    p = {"hypothesis_id": HYPOTHESIS, "source_hashes": {str(p): file_hash(p) for p in files},
        "input_hashes": {str(p): file_hash(p) for p in inputs}, "config": {**config, "system": SYSTEM,
            "model": MODEL, "temperature": 0, "max_tokens": 768, "maximum_batch": 4, "maximum_prompt_tokens": 6300,
            "workers": 4, "max_attempts": 3, "semantic_retries": False, "semantic_qualification_pending": True,
            "student_training_admitted": False, "canonical_semantic_cache_changed": False}}
    write_json(root / "producer.json", p)
    return p


def prepare():
    if data.reservation.read(data.WORKFLOW / "tool_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("actual full source-preparation exit required")
    data.reservation.verify(data.reservation.read(data.DEST / "producer.json"))
    if file_hash(data.DEST / "cases.json") != SOURCE_SHA:
        raise ValueError("unchanged complete10000 TRAIN source inputs required")
    rows = select(data.reservation.read(data.DEST / "cases.json"))
    p = produce(PREP, [data.DEST / n for n in ("producer.json", "cases.json", "result.json", "worker_terminal_receipt.json")]
        + [data.WORKFLOW / "tool_terminal_receipt.json"], {"cases": 64, "CPU_only": True})
    write_json(PREP / "cases.json", rows)
    ids = [rows[0]["id"], min(rows, key=lambda r: (-r["source_cl100k_tokens"], r["id"]))["id"]]
    if len(set(ids)) != 2:
        raise ValueError("two distinct first/longest pilot inputs required")
    write_json(PREP / "pilot_ids.json", ids)
    data.reservation.verify(p)
    write_json(PREP / "result.json", {"status": "fixed64_training_label_sources_prepared", "cases": 64,
        "cases_sha256": file_hash(PREP / "cases.json"), "task_accuracy": None})


def request(index, rows, root):
    import requests
    body = payload(rows)
    key = next(v.split("=", 1)[1].strip().strip('"').strip("'") for v in Path(".env").read_text().splitlines() if v.startswith("LITELLM_KEY="))
    attempts = []
    for attempt in range(3):
        path = root / "attempts" / f"batch_{index:04d}_{attempt}.json"
        if path.exists():
            raise ValueError("teacher attempt overwrite")
        raw = {"request": body, "valid": False}
        start = time.monotonic()
        try:
            response = requests.post("https://central-dev.zt:4000/v1/chat/completions",
                headers={"Authorization": "Bearer " + key}, json=body,
                verify="/home/rabbit/.config/lakefs/caddy-root.crt", timeout=180)
            raw.update(http_status=response.status_code, response=response.json())
            response.raise_for_status()
            if raw["response"].get("model") != MODEL or raw["response"]["choices"][0]["finish_reason"] != "stop":
                raise ValueError("teacher model or complete response termination differs")
            texts = parse(raw["response"]["choices"][0]["message"]["content"], len(rows))
            raw["valid"] = True
        except Exception as error:
            raw.update(error_type=type(error).__name__, error=str(error))
        raw["elapsed_seconds"] = time.monotonic() - start
        write_json(path, raw)
        attempts.append(str(path))
        if raw["valid"]:
            return {"index": index, "status": "valid", "case_ids": [r["id"] for r in rows], "texts": texts, "attempts": attempts}
    return {"index": index, "status": "invalid", "case_ids": [r["id"] for r in rows], "texts": [], "attempts": attempts}


def generate(pilot=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("gateway training-label generation needs no local GPU")
    native.vendor.corpus.quality.dependencies()
    data.reservation.verify(data.reservation.read(PREP / "producer.json"))
    all_rows = data.reservation.read(PREP / "cases.json")
    ids = data.reservation.read(PREP / "pilot_ids.json")
    prior = {}
    inputs = [PREP / n for n in ("producer.json", "cases.json", "pilot_ids.json", "worker_terminal_receipt.json")]
    if not pilot:
        if (data.reservation.read(PILOT / "worker_terminal_receipt.json")["exit_code"] != 0
                or data.reservation.read(PILOT / "result.json")["status"] != "two_case_teacher_format_and_capacity_passed"):
            raise ValueError("actual complete-data-path two-case teacher pilot required")
        prior = {r["id"]: r for r in data.reservation.read(PILOT / "labels.json")}
        inputs += [PILOT / n for n in ("producer.json", "labels.json", "result.json", "worker_terminal_receipt.json")]
    rows = [r for r in all_rows if r["id"] in ids] if pilot else [r for r in all_rows if r["id"] not in prior]
    root = PILOT if pilot else GENERATE
    inputs += sorted(path for path in (native.INVENTORY / "bert-tokenizer").iterdir() if path.is_file())
    p = produce(root, inputs, {"cases": 2 if pilot else 64, "retained_pilot_labels": 0 if pilot else 2})
    (root / "attempts").mkdir()
    (root / "batches").mkdir()
    encoding, tokenizer = data.storage_encoding(), native.tokenizer()
    grouped, results = batches(rows, encoding), []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(request, i, batch, root) for i, batch in enumerate(grouped)]
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            write_json(root / "batches" / f"batch_{result['index']:04d}.json", result)
            print(json.dumps({"teacher_batches": len(results), "total_batches": len(grouped)}), flush=True)
    generated = {identity: text for r in results if r["status"] == "valid" for identity, text in zip(r["case_ids"], r["texts"])}
    labels = []
    for row in (rows if pilot else all_rows):
        if row["id"] in prior:
            labels.append(prior[row["id"]])
            continue
        text = generated.get(row["id"], "")
        lexical = tokenizer.encode(text, add_special_tokens=False)
        stored = len(encoding.encode(text, disallowed_special=()))
        fits = bool(text.strip()) and 0 < stored <= 128 and len(lexical) <= 126 and not set(lexical) & {0, 101}
        labels.append({**row, "target_text": text, "teacher_valid": row["id"] in generated,
            "stored_tokens": stored, "native_lexical_tokens": len(lexical), "capacity_fit": fits,
            "native_word_ids": [101, *lexical, 101] + [0] * (126 - len(lexical)) if fits else None,
            "semantic_qualified": False})
    write_json(root / "labels.json", labels)
    capacity = sum(r["capacity_fit"] for r in labels)
    valid = sum(r["teacher_valid"] for r in labels)
    status = ("two_case_teacher_format_and_capacity_passed" if valid == capacity == 2 else "two_case_teacher_systems_failed") if pilot else "fixed64_teacher_candidate_attempts_completed_quality_pending"
    data.reservation.verify(p)
    result = {"status": status, "cases": len(labels), "valid": valid, "capacity_fit": capacity,
        "labels_sha256": file_hash(root / "labels.json"), "semantic_qualification_pending": True,
        "student_training_admitted": False, "canonical_semantic_cache_changed": False, "task_accuracy": None}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)
    if pilot and (valid != 2 or capacity != 2):
        raise ValueError("teacher systems pilot failed; all candidates preserved")


def workflow():
    if any(p.exists() for p in (WORKFLOW, PREP, PILOT, GENERATE)):
        raise ValueError("fresh fixed64 teacher roots required")
    WORKFLOW.mkdir()
    receipts = []
    for stage, root in (("prepare", PREP), ("pilot", PILOT), ("generate", GENERATE)):
        command = [sys.executable, "-u", "-m", MODULE, stage]
        child = subprocess.Popen(command)
        code = child.wait()
        receipt = {"stage": stage, "command": command, "child_pid": child.pid, "observer_pid": os.getpid(),
            "observer": "subprocess.wait", "exit_code": code}
        write_json(WORKFLOW / (stage + "_receipt.json"), receipt)
        if root.exists():
            write_json(root / "worker_terminal_receipt.json", receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError("teacher candidate stage failed:" + stage)
    write_json(WORKFLOW / "result.json", {"status": "teacher_candidate_generation_completed_semantic_quality_pending", "stages": receipts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "generate", "workflow"))
    args = parser.parse_args()
    {"prepare": prepare, "pilot": lambda: generate(True), "generate": generate, "workflow": workflow}[args.stage]()
