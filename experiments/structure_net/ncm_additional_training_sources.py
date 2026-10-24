"""Materialize only the newly reserved TRAIN sources from the shared public corpus."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

from experiments.structure_net import ncm_additional_training_reservation as reservation
from experiments.structure_net import ncm_public_exchange_fragments as fragments
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

ROOT = reservation.ROOT
ARCHIVE = ROOT / "20261024_additional10000_training_reservation_results_v2"
CAMPAIGN_SHA = "f747883d8339726a1ad01f5cc8449e77df5d4a8e0b24bb466c344e9b7539ece9"
METADATA_SHA = "b311b207dbd69d6d56930355967c7ef8a54ecacf1efdbeab63588504eaeabcbb"
PUBLIC = Path("data/experiments/ncm_memory/20261022_shared_v1/source/data/longmemeval_s_cleaned.json")
PUBLIC_SHA = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
INVENTORY = reservation.PARENT / "public_inventory.json"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-additional-training-sources-plan.md")
TEST = Path("tests/structure_net/test_ncm_additional_training_sources.py")
MODULE = "experiments.structure_net.ncm_additional_training_sources"
PILOT = ROOT / "20261024_additional10000_training_sources_pilot_v1"
DEST = ROOT / "20261024_additional10000_training_sources_v1"
WORKFLOW = ROOT / "20261024_additional10000_training_sources_workflow_v1"


def storage_encoding():
    extras = str(Path("data/experiments/ncm_memory/dependencies").resolve())
    if extras not in sys.path:
        sys.path.append(extras)
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def metadata():
    path = ARCHIVE / "campaign_results.json"
    data = reservation.read(path)
    receipt = ARCHIVE / (reservation.HYPOTHESIS + "_registry_readback.json")
    proof = reservation.read(receipt)
    selected = reservation.DEST / "training_metadata.json"
    old = reservation.read(reservation.ARCHIVE / "campaign_results.json")
    if (file_hash(path) != CAMPAIGN_SHA or data["classification"] != "additional10000_disjoint_training_sources_available_labels_unmade"
            or proof["record"]["campaign_sha256"] != CAMPAIGN_SHA or proof["exact"]["campaign_sha256"] != CAMPAIGN_SHA
            or file_hash(selected) != METADATA_SHA or data["artifacts"].get(str(selected)) != METADATA_SHA
            or file_hash(reservation.ARCHIVE / "campaign_results.json") != reservation.PARENT_SHA
            or file_hash(INVENTORY) != old["artifacts"].get(str(INVENTORY)) or file_hash(PUBLIC) != PUBLIC_SHA):
        raise ValueError("audited exact training-only metadata and public corpus required")
    rows = reservation.read(selected)
    if len(rows) != 10000 or len({r["source_component"] for r in rows}) != 10000:
        raise ValueError("complete distinct10000 training components required")
    return rows, [path, receipt, selected, INVENTORY, PUBLIC, reservation.ARCHIVE / "campaign_results.json"]


def origin(row):
    return min(row["public_origins"], key=lambda x: (text_hash(json.dumps(x, sort_keys=True, ensure_ascii=False)),
                                                    json.dumps(x, sort_keys=True, ensure_ascii=False)))


def materialize(row, chosen, session, encoding):
    start, end = chosen["start_turn"], chosen["end_turn"]
    if not 0 <= start < end <= len(session):
        raise ValueError("complete in-range public turn interval required")
    if chosen["kind"] == "whole_turn":
        if end != start + 1:
            raise ValueError("whole-turn origin must contain one complete turn")
    elif chosen["kind"] == "exchange_window":
        boundaries = fragments.exchange_boundaries(session)
        if start not in boundaries or end not in boundaries:
            raise ValueError("complete user-led exchange boundaries required")
    else:
        raise ValueError("known original source kind required")
    turns = session[start:end]
    source = "\n".join(t["content"] for t in turns)
    sid = chosen["source_id"]
    spans = []
    for turn in turns:
        sha = text_hash(turn["content"])
        spans.append({"source_id": sid, "role": turn["role"], "turn_id": "public:" + text_hash(sid + "|" + turn["role"] + "|" + sha),
            "turn_text_sha256": sha, "span_text_sha256": sha, "start_char": 0, "end_char": len(turn["content"])})
    tokens = len(encoding.encode(source, disallowed_special=()))
    if (spans != chosen["spans"] or text_hash(source) != row["source_sha256"] or len(source) != row["source_characters"]
            or tokens != row["tokens"] or reservation.graph.bucket(tokens) != row["bin"]):
        raise ValueError("exact public source, provenance or full length differs")
    return {"id": row["id"], "source_id": sid, "group_id": row["source_component"], "source_component": row["source_component"],
        "source_sha256": row["source_sha256"], "source": source, "spans": spans, "public_origin": chosen,
        "source_cl100k_tokens": tokens, "bin": row["bin"], "split": "additional_train", "source_truncated": False}


def sessions_for(rows, inventory):
    requested = {origin(row)["session_sha256"]: origin(row)["source_id"] for row in rows}
    families = set(requested.values())
    found = {}
    for sample in reservation.read(PUBLIC):
        if len(sample["haystack_session_ids"]) != len(sample["haystack_sessions"]):
            raise ValueError("public session alignment differs")
        for sid_, session in zip(sample["haystack_session_ids"], sample["haystack_sessions"]):
            sid = str(sid_)
            if sid not in families:
                continue
            values = []
            for turn in session:
                sha = text_hash(turn["content"])
                entry = inventory["turns"][sha]
                if entry["characters"] != len(turn["content"]):
                    raise ValueError("original public turn size differs")
                values.append({"role": turn["role"], **entry})
            key = text_hash(sid + "|" + json.dumps(values, sort_keys=True, ensure_ascii=False))
            if key not in requested:
                continue
            if inventory["sessions"].get(key) != {"source_id": sid, "turns": values}:
                raise ValueError("exact original session metadata differs")
            if key in found and found[key] != session:
                raise ValueError("public session collision")
            found[key] = session
    if set(found) != set(requested):
        raise ValueError("all selected TRAIN sessions required without substitute sources")
    return found


def run(pilot=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU source preparation only")
    rows, paths = metadata()
    chosen = [rows[0], min(rows, key=lambda r: (-r["tokens"], r["id"]))] if pilot else rows
    if not pilot:
        if reservation.read(PILOT / "worker_terminal_receipt.json")["exit_code"] != 0:
            raise ValueError("actual first/longest source-materialization pilot required")
        paths += [PILOT / n for n in ("result.json", "cases.json", "worker_terminal_receipt.json")]
    root = PILOT if pilot else DEST
    files = [Path(__file__), PLAN, TEST, Path(fragments.__file__), Path(reservation.__file__)]
    root.mkdir(exist_ok=False)
    (root / "producer_source").mkdir()
    for i, path in enumerate(files):
        (root / "producer_source" / f"{i}_{path.name}").write_bytes(path.read_bytes())
    producer = {"source_hashes": {str(p): file_hash(p) for p in files}, "input_hashes": {str(p): file_hash(p) for p in paths},
        "config": {"cases": len(chosen), "CPU_only": True, "training_sources_only": True, "final_sources_opened": False,
            "labels_generated": False, "model_training_executed": False, "date_string": "23 Oct 2026"}}
    write_json(root / "producer.json", producer)
    inventory = reservation.read(INVENTORY)
    lookup = sessions_for(chosen, inventory)
    encoding = storage_encoding()
    values = [materialize(row, origin(row), lookup[origin(row)["session_sha256"]], encoding) for row in chosen]
    if len({r["source_id"] for r in values}) != len(values):
        raise ValueError("distinct selected training sessions required")
    if not pilot:
        by_id = {r["id"]: r for r in values}
        if any(by_id[r["id"]] != r for r in reservation.read(PILOT / "cases.json")):
            raise ValueError("full materialization differs from retained pilot")
    write_json(root / "cases.json", values)
    reservation.verify(producer)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if peak > reservation.MAX_RSS:
        raise ValueError("source preparation exceeds8GB host memory")
    result = {"status": "additional_training_sources_exactly_materialized_no_labels", "cases": len(values),
        "cases_sha256": file_hash(root / "cases.json"), "all_source_turns_retained": True, "source_components_distinct": len(values),
        "all_sources_verified_against_exact_public_spans": True, "source_tokens": sum(r["source_cl100k_tokens"] for r in values),
        "selected_final_sources_read_or_hashed": False, "labels_generated": False, "model_trained": False,
        "task_accuracy": None, "canonical_semantic_cache_changed": False, "peak_RSS_bytes": peak}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def workflow():
    if any(p.exists() for p in (WORKFLOW, PILOT, DEST)):
        raise ValueError("fresh append-only source-preparation roots required")
    WORKFLOW.mkdir()
    receipts = []
    for stage, root in (("pilot", PILOT), ("prepare", DEST)):
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
            raise RuntimeError("additional training source preparation failed:" + stage)
    write_json(WORKFLOW / "result.json", {"status": "additional_training_source_preparation_completed", "stages": receipts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("pilot", "prepare", "workflow"))
    args = parser.parse_args()
    {"pilot": lambda: run(True), "prepare": run, "workflow": workflow}[args.stage]()
