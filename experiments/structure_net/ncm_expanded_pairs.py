"""Broaden public training sources without touching the running NCM curriculum."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body import prompt_ids
from experiments.structure_net.ncm_frozen_body_public_data import PUBLIC_FILE, PUBLIC_SHA256
from experiments.structure_net.ncm_llama import GPT2, label_ids, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_data import DATA, PREP, ROOT, SYSTEM, freeze, parse_teacher, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.train_ncm_memory import rows

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-expanded-pairs-plan.md")
AMENDMENT = Path("docs/07 - Status Reports/2026-10-22_ncm-expanded-public-filter-amendment.md")
PREPARED = ROOT / "20261022_expanded_data_v2"
TEACHER = ROOT / "20261022_expanded_teacher_v2"
HYPOTHESIS = "ncm-expanded-public-pairs-v1"
TEACHER_SYSTEM = SYSTEM + "\nPreserve explicit qualifiers, modality and negation. Do not infer a person's role or change often/some into always/each. Omit details that the passage does not establish."


@dataclass(frozen=True)
class Config:
    training_sources: int = 8192
    pilot_sources: int = 20
    passages_per_request: int = 4
    workers: int = 4
    model: str = q.JUDGES["primary"]
    temperature: int = 0
    max_tokens: int = 1024
    timeout: int = 180
    max_attempts: int = 3
    minimum_eligible: int = 7000
    audit_sources: int = 128


def producing(root, stage, inputs):
    p = freeze(root, stage, inputs, {**asdict(Config()), "teacher_system": TEACHER_SYSTEM})
    p["hypothesis_id"] = HYPOTHESIS
    for source in (Path(__file__), PLAN, AMENDMENT):
        p["source_hashes"][str(source)] = file_hash(source)
        (root / "producer_source" / ("expanded_"+source.name)).write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def select_sources(records, forbidden, tokenizer, count, accept_public=None):
    groups, sessions, hashes = (set(forbidden[key]) for key in ("group_id", "source_id", "source_sha256"))
    selected = []
    for row in sorted(records, key=lambda r: text_hash("ncm-expanded-pairs-v1:"+r["id"])):
        if row["group_id"] in groups or row["source_id"] in sessions or row["source_sha256"] in hashes:
            continue
        if row["split"] != "train" or not row["native_eligible"] or row["source_gpt2_tokens"] > 512:
            continue
        if accept_public is not None and not accept_public(row):
            continue
        parent = prompt_ids(tokenizer, row["source"])
        if len(parent) > 512:
            continue
        keep = ("id", "source_id", "source", "source_sha256", "group_id", "spans")
        selected.append({**{key: row[key] for key in keep}, "parent_ids": parent})
        groups.add(row["group_id"]); sessions.add(row["source_id"]); hashes.add(row["source_sha256"])
        if len(selected) == count:
            return selected
    raise ValueError("not enough fresh disjoint eligible training groups")


def public_index(samples):
    index = {}
    for sample in samples:
        for sid, session in zip(sample["haystack_session_ids"], sample["haystack_sessions"]):
            for turn in session:
                index[(str(sid), turn["role"], text_hash(turn["content"]))] = turn["content"]
    return index


def matches_public(row, index):
    parts = []
    for span in row["spans"]:
        turn = index.get((span["source_id"], span["role"], span["turn_text_sha256"]))
        if turn is None or not 0 <= span["start_char"] < span["end_char"] <= len(turn):
            return False
        part = turn[span["start_char"]:span["end_char"]]
        if text_hash(part) != span["span_text_sha256"]:
            return False
        parts.append(part)
    return bool(parts) and "\n".join(parts) == row["source"] and text_hash(row["source"]) == row["source_sha256"]


def prepare():
    cfg = Config(); manifest = q.read(DATA / "manifest.json")
    paths = [DATA / name for name in ("native_train.jsonl", "native_test.jsonl")]
    for path in paths:
        if file_hash(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError("original partition changed")
    old = q.read(PREP / "train_sources.json")
    test = rows(DATA / "native_test.jsonl")  # Existing source partition only; no sealed model outputs/labels.
    forbidden = {key: {r[key] for r in old+test} for key in ("group_id", "source_id", "source_sha256")}
    if file_hash(PUBLIC_FILE) != PUBLIC_SHA256:
        raise ValueError("public release identity changed")
    index = public_index(q.read(PUBLIC_FILE))
    rejected = []
    def accept_public(row):
        passed = matches_public(row, index)
        if not passed:
            rejected.append({"id": row["id"], "source_sha256": row["source_sha256"], "reason": "no exact canonical public-span proof"})
        return passed
    _, llama = tokenizers()
    selected = select_sources(rows(DATA / "native_train.jsonl"), forbidden, llama, cfg.training_sources+cfg.pilot_sources, accept_public)
    proof = verify_public(selected)
    p = producing(PREPARED, "fresh_public_training_source_selection", [DATA / "manifest.json", *paths,
        PREP / "train_sources.json", PUBLIC_FILE])
    write_json(PREPARED / "pilot_sources.json", selected[:cfg.pilot_sources])
    write_json(PREPARED / "train_sources.json", selected[cfg.pilot_sources:])
    write_json(PREPARED / "unverified_exclusions.json", rejected)
    write_json(PREPARED / "public_proof.json", {"public_file_sha256": PUBLIC_SHA256, "exact_sources": proof,
        "matched": len(proof), "outgoing_old_targets": False})
    verify(p)
    result = {"status": "passed", "training_sources": cfg.training_sources, "pilot_sources": cfg.pilot_sources,
        "fresh_groups": len({r["group_id"] for r in selected}), "split_overlap": False,
        "unverified_rows_excluded": len(rejected),
        "producer_fingerprint": p["fingerprint"], "task_accuracy": None, "trained": False}
    write_json(PREPARED / "result.json", result); print(json.dumps(result), flush=True)


def request(index, cases, root):
    import requests
    cfg = Config()
    payload = {"model": cfg.model, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
        "messages": [{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": json.dumps(
            {"passages": [{"id": row["id"], "text": row["source"]} for row in cases]}, ensure_ascii=False)}]}
    key = next(x.split("=", 1)[1].strip().strip('"').strip("'")
        for x in Path(".env").read_text().splitlines() if x.startswith("LITELLM_KEY="))
    attempts = []
    for attempt in range(cfg.max_attempts):
        path = root / "attempts" / f"batch_{index:05d}_{attempt}.json"
        if path.exists():
            raise ValueError("teacher attempt overwrite")
        raw = {"request": payload, "request_sha256": q.digest(payload), "valid": False}
        start = time.monotonic()
        try:
            response = requests.post("https://central-dev.zt:4000/v1/chat/completions",
                headers={"Authorization": "Bearer "+key}, json=payload,
                verify="/home/rabbit/.config/lakefs/caddy-root.crt", timeout=cfg.timeout)
            raw.update(http_status=response.status_code, response=response.json())
            response.raise_for_status()
            if raw["response"].get("model") != cfg.model:
                raise ValueError("teacher alias mismatch")
            summaries = parse_teacher(raw["response"]["choices"][0]["message"]["content"], [r["id"] for r in cases])
            raw["valid"] = True
        except Exception as error:
            raw["error_type"] = type(error).__name__
        raw["elapsed_seconds"] = time.monotonic()-start
        write_json(path, raw); attempts.append(str(path))
        if raw["valid"]:
            return {"batch": index, "status": "valid", "summaries": summaries, "attempts": attempts}
    return {"batch": index, "status": "invalid", "summaries": {}, "attempts": attempts}


def teacher(pilot):
    cfg = Config()
    source = PREPARED / ("pilot_sources.json" if pilot else "train_sources.json")
    if q.read(PREPARED / "result.json")["status"] != "passed":
        raise ValueError("public source preparation required")
    cases = q.read(source); verify_public(cases)
    root = ROOT / "20261022_expanded_teacher_pilot_v2" if pilot else TEACHER
    inputs = [source, PREPARED / "producer.json", PREPARED / "public_proof.json", PUBLIC_FILE,
              *[path for path in GPT2.iterdir() if path.is_file()]]
    if not pilot:
        pr = ROOT / "20261022_expanded_teacher_pilot_v2"
        if q.read(pr / "result.json")["status"] != "passed":
            raise ValueError("representative teacher pilot required")
        inputs.extend([pr / "result.json", pr / "producer.json"])
    p = producing(root, "systems_only_not_quality_evidence" if pilot else "expanded_teacher_generation", inputs)
    if not pilot and q.read(pr / "producer.json")["source_hashes"] != p["source_hashes"]:
        raise ValueError("teacher sources differ from passed pilot")
    for name in ("attempts", "rows"):
        (root / name).mkdir()
    batches = [cases[i:i+cfg.passages_per_request] for i in range(0, len(cases), cfg.passages_per_request)]
    results = []
    if pilot:
        first = request(0, batches[0], root); results.append(first)
        write_json(root / "rows/batch_00000.json", first)
        if first["status"] != "valid":
            write_json(root / "result.json", {"status": "failed_serial_pilot", "task_accuracy": None})
            return
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        jobs = [pool.submit(request, i, batch, root) for i, batch in enumerate(batches) if not pilot or i > 0]
        for future in as_completed(jobs):
            row = future.result(); results.append(row)
            write_json(root / "rows" / f"batch_{row['batch']:05d}.json", row)
            if len(results) % 16 == 0:
                write_json(root / "progress.json", {"completed_batches": len(results), "total_batches": len(batches)})
                print(json.dumps({"batches": len(results), "total": len(batches)}), flush=True)
    mapping = {key: value for row in results for key, value in row["summaries"].items()}
    gpt, _ = tokenizers()
    accepted, excluded = [], []
    for case in cases:
        text = mapping.get(case["id"])
        try:
            if not text:
                raise ValueError("empty or missing label")
            ids = label_ids(text, gpt)
            accepted.append({**case, "target": text, "target_sha256": text_hash(text), "native_target_ids": ids,
                "target_gpt2_tokens": len(gpt.encode(text, add_special_tokens=False)), "target_producer": cfg.model})
        except ValueError as error:
            excluded.append({"id": case["id"], "target": text, "reason": str(error)})
    verify(p)
    write_json(root / "train.json", accepted); write_json(root / "excluded.json", excluded)
    valid = sum(row["status"] == "valid" for row in results)
    success = valid == len(batches) and len(accepted) >= (16 if pilot else cfg.minimum_eligible)
    result = {"status": "passed" if success else "failed_data_gate", "requested": len(cases), "eligible": len(accepted),
        "excluded": len(excluded), "batches": len(batches), "valid_batches": valid,
        "retried_batches": sum(len(row["attempts"]) > 1 for row in results), "quality_audited": False,
        "systems_only": pilot, "task_accuracy": None, "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


def audit(pilot):
    from experiments.structure_net.ncm_paper_grade import grade
    if q.read(TEACHER / "result.json")["status"] != "passed":
        raise ValueError("completed eligible teacher data required")
    ordered = sorted(q.read(TEACHER / "train.json"), key=lambda row: text_hash("expanded-audit-v1:"+row["id"]))
    chosen = ordered[128:132] if pilot else ordered[:128]
    cases = [{**row, "text": row["target"]} for row in chosen]
    sources = [TEACHER / "train.json", TEACHER / "result.json", Path(__file__), PLAN, AMENDMENT]
    root = ROOT / ("20261022_expanded_audit_pilot_v2" if pilot else "20261022_expanded_audit_v2")
    if not pilot:
        previous = ROOT / "20261022_expanded_audit_pilot_v2/result.json"
        result = q.read(previous)
        if result["status"] != "completed" or result["invalid"] or result["cases"] != 4 or not result["systems_only"]:
            raise ValueError("fresh4-request independent audit pilot required")
        sources.append(previous)
    grade(cases, root, sources, role="secondary", systems_only=pilot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "pilot", "teacher", "audit-pilot", "audit"])
    command = parser.parse_args().command
    if command == "prepare":
        prepare()
    elif command in ("audit-pilot", "audit"):
        audit(command == "audit-pilot")
    else:
        teacher(command == "pilot")
