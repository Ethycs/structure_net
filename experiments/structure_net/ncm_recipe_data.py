"""Select quality-checked, provably public targets for a bounded NCM update."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body_public_data import (
    PUBLIC_FILE, PUBLIC_SHA256, verify_cases)
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_recipe_control import PLAN, ROOT

PARENT = Path("data/experiments/ncm_frozen_body/20261022_train_v1")
QUALITY = ROOT / "20261022_quality_v1"
NOTE = Path("docs/07 - Status Reports/2026-10-22_ncm-recipe-public-target-clarification.md")
CALIBRATION = Path("data/experiments/ncm_memory/20261022_accuracy_v2/calibration_primary/result.json")


def public_check(cases):
    if file_hash(PUBLIC_FILE) != PUBLIC_SHA256:
        raise ValueError("public LongMemEval file SHA mismatch")
    receipt = q.read(PARENT / "public_data_readback.json")
    if receipt["public_file_metadata"]["lfs"]["oid"] != PUBLIC_SHA256:
        raise ValueError("missing published SHA verification")
    matched = verify_cases(q.read(PUBLIC_FILE), cases)
    for case in cases:
        if case["target"] not in case["source"]:
            raise ValueError("candidate is not literal public source text")
    return matched


def prepare():
    if QUALITY.exists():
        raise ValueError("fresh quality root required")
    records = q.read(PARENT / "selection.json")["train"]
    eligible = [r for r in records if 20 <= len(r["target"].split()) <= 40 and r["target"] in r["source"]]
    cases = sorted(eligible, key=lambda r: text_hash("recipe-quality:"+r["id"]))[:128]
    if len(cases) != 128:
        raise ValueError("insufficient fixed candidate pool")
    matched = public_check(cases)
    QUALITY.mkdir(parents=True)
    for sub in ("attempts", "rows", "producer_source"):
        (QUALITY / sub).mkdir()
    packets = [q.packet(case["id"], case["source"], [{"candidate_id": "public-extract-"+case["id"][:16],
                                                   "text": case["target"]}]) for case in cases]
    write_json(QUALITY / "cases.json", cases)
    write_json(QUALITY / "packets.json", packets)
    write_json(QUALITY / "public_readback.json", {"status": "verified_public_sources_and_targets",
        "public_sha256": PUBLIC_SHA256, "sources": matched,
        "target_substrings": {r["id"]: {"start": r["source"].index(r["target"]),
            "length": len(r["target"]), "sha256": text_hash(r["target"])} for r in cases},
        "payload_scope": "Only exact public source spans and their literal public target substrings."})
    sources = [Path(__file__), PLAN, NOTE, Path("experiments/structure_net/ncm_memory_accuracy.py"),
        Path("experiments/structure_net/ncm_memory_accuracy_v2.py"),
        Path("experiments/structure_net/ncm_frozen_body_public_data.py")]
    inputs = [PARENT / "selection.json", PARENT / "public_data_readback.json", PUBLIC_FILE, CALIBRATION]
    inputs += [QUALITY / n for n in ("cases.json", "packets.json", "public_readback.json")]
    for i, path in enumerate(sources):
        shutil.copyfile(path, QUALITY / "producer_source" / f"{i:03d}_{path.name}")
    producer = {"schema": "nal.ncm-recipe-target-quality.v1", "hypothesis_id": "ncm-recipe-update-v1",
        "judge": q.JUDGES["primary"], "rubric": q.RUBRIC, "workers": 4,
        "source_hashes": {str(p): file_hash(p) for p in sources},
        "input_hashes": {str(p): file_hash(p) for p in inputs},
        "purpose": "training target selection, not held-out model accuracy", "candidates": 128}
    producer["fingerprint"] = q.digest(producer)
    write_json(QUALITY / "producer.json", producer)
    print(json.dumps({"status": "prepared", "public_candidates": len(cases)}), flush=True)


def grade():
    if (QUALITY / "result.json").exists():
        raise ValueError("completed quality selection is append-only")
    producer = q.read(QUALITY / "producer.json")
    for path, expected in {**producer["source_hashes"], **producer["input_hashes"]}.items():
        if file_hash(path) != expected:
            raise ValueError("quality producer changed")
    calibration = q.read(CALIBRATION)
    if calibration["status"] != "passed" or calibration["counts"].get("positive_accepted") != 24 or calibration["counts"].get("negative_rejected") != 24:
        raise ValueError("unchanged qualified judge required")
    cases, packets = q.read(QUALITY / "cases.json"), q.read(QUALITY / "packets.json")
    public_check(cases)  # Required before the first external request.
    # Ensure every outgoing field is rebuilt solely from the verified public text.
    expected = [q.packet(r["id"], r["source"], [{"candidate_id": "public-extract-"+r["id"][:16], "text": r["target"]}]) for r in cases]
    if packets != expected:
        raise ValueError("nonpublic or modified judge payload")
    with ThreadPoolExecutor(max_workers=4) as pool:
        judgments = []
        for row in pool.map(lambda p: q.request(p, "primary", QUALITY), packets):
            write_json(QUALITY / "rows" / (row["case_id"]+".json"), row)
            judgments.append(row)
            if len(judgments) % 16 == 0:
                print(json.dumps({"stage": "public_training_target_quality", "completed": len(judgments)}), flush=True)
    passing = []
    for case, packet, row in zip(cases, packets, judgments):
        if row["status"] != "valid":
            continue
        raw = q.read(row["attempts"][-1])
        parsed = q.parse(raw["response"]["choices"][0]["message"]["content"], packet)
        if parsed != row["evaluations"] or any(q.read(p)["valid"] for p in row["attempts"][:-1]):
            raise ValueError("raw judge audit failed or valid failure retried")
        # Native eligible labels <=47 GPT2 tokens; still measure the actual limit.
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        if q.passes(parsed[0], case["target"], len(enc.encode(case["target"], disallowed_special=()))):
            passing.append(case)
    write_json(QUALITY / "judgments.json", judgments)
    selected = passing[:32]
    write_json(QUALITY / "selected.json", {"train": selected, "selection": "first 32 accepted in fixed candidate order"})
    result = {"status": "passed" if len(selected) == 32 and all(r["status"] == "valid" for r in judgments) else "target_quality_gate_failed",
        "candidates": 128, "accepted": len(passing), "selected": len(selected),
        "invalid": sum(r["status"] != "valid" for r in judgments),
        "retried": sum(len(r["attempts"]) > 1 for r in judgments),
        "producer_fingerprint": producer["fingerprint"], "heldout_accuracy": False}
    write_json(QUALITY / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "grade"])
    globals()[parser.parse_args().command]()
