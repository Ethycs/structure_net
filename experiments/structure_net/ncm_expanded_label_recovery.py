"""Recover existing valid public labels without rewriting a failed teacher campaign."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_llama import label_ids, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-expanded-label-recovery-plan.md")
PARENT = ROOT / "20261022_expanded_teacher_v2"
SOURCES = ROOT / "20261022_expanded_data_v2/train_sources.json"
RECOVERY = ROOT / "20261022_expanded_label_recovery_v1"


def checked_teacher_batch(row, batch, read=q.read):
    from experiments.structure_net.ncm_paper_data import parse_teacher
    if row["status"] not in ("valid", "invalid") or not 1 <= len(row["attempts"]) <= 3:
        raise ValueError("teacher batch record inconsistent")
    attempts = [read(p) for p in row["attempts"]]
    parsed_responses = []
    for raw in attempts:
        expected = [{"id": r["id"], "text": r["source"]} for r in batch]
        outgoing = json.loads(raw["request"]["messages"][1]["content"])["passages"]
        if outgoing != expected:
            raise ValueError("raw teacher source membership changed")
        parsed = None
        try:
            if raw["http_status"] == 200 and raw["response"].get("model") == q.JUDGES["primary"]:
                parsed = parse_teacher(raw["response"]["choices"][0]["message"]["content"], [r["id"] for r in batch])
        except (ValueError, KeyError, TypeError, IndexError):
            pass
        if bool(raw["valid"]) != (parsed is not None):
            raise ValueError("raw validity differs from independent parser replay")
        parsed_responses.append(parsed)
    if row["status"] == "valid":
        if sum(p is not None for p in parsed_responses) != 1 or parsed_responses[-1] is None:
            raise ValueError("first-valid response preservation failed")
        if parsed_responses[-1] != row["summaries"]:
            raise ValueError("accepted summaries differ from the first valid raw response")
    elif any(p is not None for p in parsed_responses) or row["summaries"] or len(attempts) != 3:
        raise ValueError("invalid batch did not preserve the failed retry cap")
    return row


def prepare():
    if RECOVERY.exists():
        raise ValueError("fresh recovery root required")
    result_path = PARENT / "result.json"
    if not result_path.exists():
        raise ValueError("completed teacher campaign required; never recover an unfinished live producer")
    original = q.read(result_path)
    if original["status"] not in ("passed", "failed_data_gate") or original["systems_only"]:
        raise ValueError("completed non-systems teacher result required")
    verify(q.read(PARENT / "producer.json"))
    sources = q.read(SOURCES)
    if len(sources) != 8192 or original["batches"] != 2048:
        raise ValueError("full fixed source schedule required")
    batches = [sources[i:i+4] for i in range(0, len(sources), 4)]
    rows = []
    for i, batch in enumerate(batches):
        path = PARENT / "rows" / f"batch_{i:05d}.json"
        row = q.read(path)
        if row["batch"] != i:
            raise ValueError("teacher batch record inconsistent")
        rows.append(checked_teacher_batch(row, batch))
    mapping = {key: value for row in rows for key, value in row["summaries"].items()}
    gpt, _ = tokenizers()
    accepted, excluded = [], []
    for case in sources:
        text = mapping.get(case["id"])
        try:
            if not text:
                raise ValueError("empty or missing label")
            ids = label_ids(text, gpt)
            accepted.append({**case, "target": text, "target_sha256": text_hash(text),
                "native_target_ids": ids, "target_gpt2_tokens": len(gpt.encode(text, add_special_tokens=False)),
                "target_producer": q.JUDGES["primary"]})
        except ValueError as error:
            excluded.append({"id": case["id"], "target": text, "reason": str(error)})
    if accepted != q.read(PARENT / "train.json") or excluded != q.read(PARENT / "excluded.json"):
        raise ValueError("independently reconstructed accepted/excluded sets differ")
    if (len(accepted) != original["eligible"] or len(excluded) != original["excluded"]
            or len(accepted) < 7000 or sum(r["status"] == "valid" for r in rows) != original["valid_batches"]):
        raise ValueError("label counts or minimum availability gate failed")
    for key in ("id", "source_id", "group_id", "source_sha256"):
        if len({r[key] for r in accepted}) != len(accepted):
            raise ValueError("recovered source uniqueness failed")
    verify_public(accepted)
    inputs = [SOURCES, PLAN, Path(__file__), result_path, PARENT / "producer.json",
              PARENT / "train.json", PARENT / "excluded.json"]
    inputs.extend(p for p in PARENT.rglob("*.json") if "producer_source" not in p.parts)
    p = freeze(RECOVERY, "valid_public_label_recovery_not_original_campaign_success", inputs,
        {"minimum_eligible": 7000, "new_teacher_requests": 0, "quality_selection": False,
         "audit_hash_prefix": "expanded-audit-v1:", "audit_cases": 128, "audit_pass_minimum": 122})
    p["hypothesis_id"] = "ncm-expanded-valid-label-recovery-v1"
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(RECOVERY / "producer.json", p)
    (RECOVERY / "source.py").write_bytes(Path(__file__).read_bytes())
    (RECOVERY / "plan.md").write_bytes(PLAN.read_bytes())
    (RECOVERY / "train.json").write_bytes((PARENT / "train.json").read_bytes())
    verify(p)
    result = {"status": "available_labels_recovered_pending_quality_audit", "eligible": len(accepted),
        "excluded": len(excluded), "original_status": original["status"], "invalid_batches": 2048-original["valid_batches"],
        "retried_batches": original["retried_batches"], "new_teacher_requests": 0,
        "original_campaign_reclassified": False, "reconstructed_first_valid_labels_equal": True,
        "train_sha256": file_hash(RECOVERY / "train.json"), "producer_fingerprint": p["fingerprint"],
        "task_accuracy": None, "student_trained": False, "final_test_opened": False, "speed_tested": False}
    write_json(RECOVERY / "result.json", result); print(json.dumps(result), flush=True)


def audit(pilot):
    from experiments.structure_net.ncm_paper_grade import grade
    if q.read(RECOVERY / "result.json")["status"] != "available_labels_recovered_pending_quality_audit":
        raise ValueError("completed exact label recovery required")
    source = RECOVERY / "train.json"
    if file_hash(source) != q.read(RECOVERY / "result.json")["train_sha256"]:
        raise ValueError("recovered labels changed")
    ordered = sorted(q.read(source), key=lambda r: text_hash("expanded-audit-v1:"+r["id"]))
    selected = ordered[128:132] if pilot else ordered[:128]
    cases = [{**row, "text": row["target"]} for row in selected]
    inputs = [source, RECOVERY / "result.json", PLAN, Path(__file__)]
    root = ROOT / ("20261022_expanded_recovery_audit_pilot_v1" if pilot else "20261022_expanded_recovery_audit_v1")
    if not pilot:
        previous = ROOT / "20261022_expanded_recovery_audit_pilot_v1/result.json"
        r = q.read(previous)
        if r["status"] != "completed" or r["valid"] != 4 or not r["systems_only"]:
            raise ValueError("fresh four-request independent audit pilot required")
        inputs.append(previous)
    grade(cases, root, inputs, role="secondary", systems_only=pilot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "audit-pilot", "audit"])
    command = parser.parse_args().command
    prepare() if command == "prepare" else audit(command == "audit-pilot")
