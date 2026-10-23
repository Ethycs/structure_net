"""Fixed one-revision public training labels, separate from student evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_budget128_profile as profile
from experiments.structure_net import ncm_revised_label_pipeline as teacher
from experiments.structure_net import ncm_revised_label_capacity_recovery as capacity
from experiments.structure_net import ncm_source_readout_grade as first
from experiments.structure_net import ncm_application_range_sources as sources
from experiments.structure_net.ncm_native_label_quality import dependencies
from experiments.structure_net.ncm_frozen_body import MODEL_PATH
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-wide-vetted-label-plan.md")
SYSTEM = teacher.SYSTEM + ("\nPrefer40-60words. Revision critiques include exact stored and native token counts "
    "and128-token ceilings. Shorten any over-cap draft in the single revision while preserving "
    "the main point and every material qualification. Review every draft once, including passing drafts.")


def root(pilot, stage):
    return ROOT / ("20261023_wide_labels_" + ("pilot" if pilot else "main") + "_" + stage + "_v1")


def terminal(path):
    if q.read(path / "worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("previous worker must be observed terminal success")
    verify(q.read(path / "producer.json"))


def label_fits(row):
    return bool(row["teacher_valid"] and row["text"].strip() and row["claims"]
        and all(c["claim"].strip() and c["evidence_ids"] for c in row["claims"])
        and row["stored_tokens"] <= 128 and row["native_tokens_including_eos"] <= 128
        and row["native_roundtrip_exact"] and not row["native_special_tokens_in_text"])


def setup(pilot):
    dependencies(); profile.install(); teacher.SYSTEM = SYSTEM
    terminal(sources.POOL)
    cases = q.read(sources.POOL / "train_sources.json")
    if (len(cases) != 112 or any(c["pool_split"] != "train" for c in cases)
            or file_hash(sources.POOL / "train_sources.json") != q.read(sources.POOL / "result.json")["files"][str(sources.POOL / "train_sources.json")]):
        raise ValueError("fixed112 source-only training cases required")
    verify_public(cases)
    for role in ("secondary", "primary"):
        qual = ROOT / ("20261023_budget128_judge_" + role + "_v1")
        terminal(qual)
        if q.read(qual / "result.json")["status"] != "passed":
            raise ValueError("both qualified128-token judges required")
    if not pilot:
        terminal(root(True, "audit"))
        if not q.read(root(True, "audit") / "result.json")["pilot_gate"]:
            raise ValueError("fixed4/4final pilot gate failed; main not admitted")
    if pilot:
        cases = [next(c for c in cases if c["bin"] == bucket) for bucket in range(4)]
    return cases


def producing(destination, stage, inputs, config):
    files = [Path(__file__), PLAN, Path(teacher.__file__), teacher.PLAN, Path(capacity.__file__), capacity.PLAN,
             Path(profile.__file__), profile.PLAN, Path(first.__file__), first.AMENDMENT]
    qualifications = []
    for role in ("secondary", "primary"):
        qualifications += [ROOT / ("20261023_budget128_judge_" + role + "_v1") / n
                           for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    tokenizers = [MODEL_PATH / n for n in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json")]
    p = freeze(destination, stage, inputs + files + qualifications + tokenizers,
        {**config, "teacher": q.JUDGES["primary"], "auditor": q.JUDGES["secondary"], "system": SYSTEM,
         "rubric": profile.RUBRIC, "workers": 4, "generation_batch": 2, "generation_max_tokens": 2048,
         "temperature": 0, "max_attempts": 3, "semantic_rerolls": False, "student_training_admitted": False})
    for path in files:
        p["source_hashes"][str(path)] = file_hash(path)
        (destination / "producer_source" / ("wide_label_" + path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    return p


def cache_from(paths):
    cache = {}
    for path in paths:
        terminal(path)
        for key, entry in q.read(path / "judgment_cache.json").items():
            first.audit_row(entry["packet"], entry["role"], entry["row"])
            if key != first.packet_key(entry["packet"], entry["role"]):
                raise ValueError("first judgment cache identity differs")
            first.remember(cache, key, entry)
    return cache


def generate(pilot, revision):
    cases = setup(pilot)
    stage = "revision" if revision else "draft"
    destination = root(pilot, stage)
    inputs = [sources.POOL / n for n in ("producer.json", "result.json", "train_sources.json", "worker_terminal_receipt.json")]
    drafts = critiques = None
    if revision:
        terminal(root(pilot, "critique")); terminal(root(pilot, "draft"))
        inputs += [root(pilot, "draft") / "labels.json", root(pilot, "critique") / "judgment_cache.json"]
        drafts = {r["id"]: r for r in q.read(inputs[-2])}
        cache = q.read(inputs[-1]); critiques = {}
        for case in cases:
            packet = q.packet(case["id"], case["source"], [{"candidate_id": "summary", "text": drafts[case["id"]]["text"]}])
            entry = cache[first.packet_key(packet, "secondary")]
            if entry["row"]["status"] != "valid":
                raise ValueError("invalid critique stops revision")
            critiques[case["id"]] = capacity.capacity_critique(drafts[case["id"]], entry["row"]["evaluations"][0])
    retained = {}
    if not pilot:
        terminal(root(True, stage))
        inputs += [root(True, stage) / n for n in ("labels.json", "producer.json", "result.json", "worker_terminal_receipt.json")]
        retained = {r["id"]: copy.deepcopy(r) for r in q.read(root(True, stage) / "labels.json")}
    pending = [c for c in cases if c["id"] not in retained]
    p = producing(destination, "public_wide_teacher_" + stage, inputs,
                  {"pilot": pilot, "cases": len(cases), "retained_pilot_texts": len(retained), "one_revision_per_case": True})
    (destination / "attempts").mkdir()
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(teacher.request, i // 2, pending[i:i+2], destination, drafts, critiques)
                for i in range(0, len(pending), 2)]
        for future in as_completed(jobs):
            rows.append(future.result())
    values = {key: value for row in rows for key, value in row["values"].items()}
    if set(values) & set(retained):
        raise ValueError("pilot teacher candidate regenerated")
    import tiktoken
    from transformers import AutoTokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    labels = []
    for case in cases:
        if case["id"] in retained:
            labels.append(retained[case["id"]]); continue
        value = values.get(case["id"], {})
        text = value.get("summary", "")
        claims = copy.deepcopy(value.get("claims", []))
        lines = case["source"].split("\n")
        for claim in claims:
            claim["source_quotes"] = [lines[int(key[1:])] for key in claim["evidence_ids"]]
        native = tokenizer.encode(text, add_special_tokens=False)
        labels.append({**case, "text": text, "target": text, "claims": claims,
            "teacher_valid": case["id"] in values, "stored_tokens": len(enc.encode(text, disallowed_special=())),
            "native_tokens_including_eos": len(native) + 1,
            "native_roundtrip_exact": tokenizer.decode(native, skip_special_tokens=False) == text,
            "native_special_tokens_in_text": any(v in tokenizer.all_special_ids for v in native)})
    write_json(destination / "labels.json", labels); write_json(destination / "batches.json", rows)
    verify(p)
    result = {"status": "generated" if all(r["teacher_valid"] for r in labels) else "invalid_generation_rows",
        "cases": len(labels), "pilot": pilot, "format_capacity_fit": sum(label_fits(r) for r in labels),
        "retained_pilot_texts": len(retained), "student_training_admitted": False,
        "producer_fingerprint": p["fingerprint"], "NCM_success_established": False}
    write_json(destination / "result.json", result); print(json.dumps(result), flush=True)


def grade(pilot, audit):
    cases = setup(pilot)
    source = root(pilot, "revision" if audit else "draft")
    terminal(source)
    if q.read(source / "result.json")["status"] != "generated":
        raise ValueError("all first teacher schemas must be valid before judging")
    labels = q.read(source / "labels.json")
    if [r["id"] for r in labels] != [r["id"] for r in cases]:
        raise ValueError("fixed source denominator/order changed")
    stage = "audit" if audit else "critique"
    destination = root(pilot, stage)
    prior = [] if pilot else [root(True, "critique"), root(True, "audit")]
    if audit:
        prior += [root(pilot, "critique")]
    cache = cache_from(list(dict.fromkeys(prior)))
    inputs = [source / n for n in ("producer.json", "result.json", "labels.json", "worker_terminal_receipt.json")]
    inputs += [path / "judgment_cache.json" for path in prior]
    p = producing(destination, "whole_summary_wide_teacher_" + stage, inputs, {"pilot": pilot, "cases": len(cases)})
    write_json(destination / "judgment_cache.json", cache)
    first.DEST, first.freeze = destination, producing
    endpoint = first.grade_condition(labels, stage, "secondary", cache, inputs)
    scores = q.read(destination / ("secondary_" + stage) / "case_scores.json")
    accepted = [r for r, score in zip(labels, scores) if score["passed"] is True and label_fits(r)] if audit else []
    if audit:
        write_json(destination / "accepted_labels.json", accepted)
    verify(p)
    result = {"status": "completed", "pilot": pilot, "endpoint": endpoint,
        "pilot_gate": (endpoint["valid"] == 4 and endpoint["passed"] == 4 and len(accepted) == 4) if pilot and audit else None,
        "accepted_labels": len(accepted) if audit else None,
        "accepted_by_bin": [sum(r["bin"] == i for r in accepted) for i in range(4)] if audit else None,
        "format_capacity_failures": sum(not label_fits(r) for r in labels),
        "student_training_admitted": False, "teacher_population95_claim": False,
        "NCM_success_established": False, "final_test_opened": False, "producer_fingerprint": p["fingerprint"]}
    write_json(destination / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["draft", "critique", "revision", "audit"])
    parser.add_argument("--main", action="store_true")
    args = parser.parse_args()
    if args.stage in ("draft", "revision"):
        generate(not args.main, args.stage == "revision")
    else:
        grade(not args.main, args.stage == "audit")
