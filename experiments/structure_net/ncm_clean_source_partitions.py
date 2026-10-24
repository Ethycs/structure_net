"""Reserve full-size, component-disjoint public NCM development/final sources."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

import torch

from experiments.structure_net import ncm_clean_source_graph as core
from experiments.structure_net import ncm_dense_source_overlap_audit as overlap
from experiments.structure_net.ncm_frozen_body_public_data import PUBLIC_FILE, PUBLIC_SHA256, verify_cases
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, PREP, DATA, freeze, verify

HYPOTHESIS = "ncm-clean-full-size-source-partitions-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-clean-source-partitions-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-clean-source-partitions.md")
WORKFLOW = ROOT / "20261023_clean_source_partitions_workflow_v1"
PILOT = ROOT / "20261023_clean_source_partitions_pilot_v1"
DEST = ROOT / "20261023_clean_source_partitions_v1"
RESULTS = ROOT / "20261023_clean_source_partitions_results_v1"
MODULE = "experiments.structure_net.ncm_clean_source_partitions"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_clean_source_partitions"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_clean_source_partitions.py")
NAMESPACE = "ncm-clean-full-size-source-partitions-v1"
FINAL = {0: 256, 1: 16, 2: 16, 3: 16, 4: 16}
DEVELOPMENT = {0: 64, 1: 16, 2: 16, 3: 16, 4: 16}
KNOWN = [overlap.CASES, overlap.study.reference.SOURCES / "cases.json",
    PREP / "train_sources.json", PREP / "development_sources.json",
    ROOT / "20261023_application_source_pool_v1/train_sources.json",
    ROOT / "20261023_application_source_pool_v1/diagnostic_sources.json"]
LONG_METADATA = ROOT / "20261023_application_source_pool_results_v1/selection_metadata_replay.json"
NATIVE_TEST_SHA = "89c890bb35f32e36de0d62f7d709b449b721dd3c8bf03f4e702bd3063622e25f"
METADATA_FIELDS = ("id", "source_id", "group_id", "source_sha256")
MAX_RSS = 24 * 1024 ** 3


def configure():
    overlap.study.training.clean.dependencies()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("source partition preparation is explicitly CPU only")


def flatten(value):
    if isinstance(value, dict):
        if all(k in value for k in (*METADATA_FIELDS, "source", "spans")):
            yield value
        else:
            for child in value.values():
                yield from flatten(child)
    elif isinstance(value, list):
        for child in value:
            yield from flatten(child)


def reserved_metadata():
    """Read existing sidecars and original dataset metadata, never sealed files."""
    q = overlap.study.reference.q
    known_ids = {r["id"] for path in (PREP / "train_sources.json", PREP / "development_sources.json")
        for r in flatten(q.read(path))}
    proof = q.read(PREP / "public_proof.json")["exact_sources"]
    final = {key: value for key, value in proof.items() if key not in known_ids}
    if len(final) != 256 or file_hash(DATA / "native_test.jsonl") != NATIVE_TEST_SHA:
        raise ValueError("original256 final metadata and original raw data identity required")
    recovered = {}
    with (DATA / "native_test.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in final:
                if row["source_sha256"] != final[row["id"]]:
                    raise ValueError("reserved original final identity differs")
                recovered[row["id"]] = {**{k: row[k] for k in METADATA_FIELDS},
                    "spans": [{k: span[k] for k in ("source_id", "role", "turn_text_sha256")} for span in row["spans"]]}
    long = q.read(LONG_METADATA)["sealed_final"]
    if len(recovered) != 256 or len(long) != 64:
        raise ValueError("all old final metadata required without sealed source-file access")
    return [*recovered.values(), *long]


def inputs():
    configure()
    q = overlap.study.reference.q
    archive = overlap.DEST / "campaign_results.json"
    receipt = q.read(overlap.DEST / (overlap.HYPOTHESIS + "_registry_readback.json"))
    terminal = q.read(overlap.WORKFLOW / "result.json")
    if (receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != file_hash(archive)
            or receipt["exact"]["campaign_sha256"] != file_hash(archive)
            or q.read(archive)["classification"] != "development_split_contaminated"
            or terminal["status"] != "source_overlap_audit_completed"
            or any(r["exit_code"] for r in terminal["stages"])):
        raise ValueError("registered contamination amendment required before new source selection")
    if file_hash(PUBLIC_FILE) != PUBLIC_SHA256 or file_hash(overlap.CASES) != overlap.CASES_SHA:
        raise ValueError("immutable public release and full690 TRAIN cases required")
    known = {}
    for path in KNOWN:
        for row in flatten(q.read(path)):
            if text_hash(row["source"]) != row["source_sha256"]:
                raise ValueError("known source bytes differ from identity")
            if row["id"] in known and known[row["id"]]["source_sha256"] != row["source_sha256"]:
                raise ValueError("known source identity collision")
            known[row["id"]] = row
    reserved = reserved_metadata()
    paths = [PUBLIC_FILE, *KNOWN, LONG_METADATA, PREP / "public_proof.json", DATA / "native_test.jsonl",
        archive, overlap.DEST / (overlap.HYPOTHESIS + "_registry_readback.json"), overlap.WORKFLOW / "result.json"]
    return list(known.values()), reserved, paths


def producing(destination, stage, extra=()):
    _, _, inputs_ = inputs()
    files = [Path(__file__), Path(core.__file__), Path(core.guard.__file__), PLAN, AUDITOR_FILE,
        Path(overlap.__file__), Path(verify_cases.__code__.co_filename)]
    p = freeze(destination, stage, [*inputs_, *files, *extra],
        {"hypothesis_id": HYPOTHESIS, "CPU_only": True, "torch": torch.__version__,
         "namespace": NAMESPACE, "final_quotas": FINAL, "development_quotas": DEVELOPMENT,
         "public_sources": "complete_public_turns_at_least192_cl100k;candidate_max2048",
         "roles": "all_public_roles_without_quality_filtering", "pilot_public_samples": 2,
         "source_grouping": "connected_components_family_group_publicturn_normalized_text_and_exact_word5gram_overlap",
         "near_overlap": "at_least32_shared_5grams_and80percent_shorter_containment",
         "maximum_RSS_bytes": MAX_RSS, "model_execution": False, "labels_generated": False,
         "old_sealed_source_files_read_or_hashed": False, "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (destination / "producer_source" / f"clean_sources_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = overlap.study.reference.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    return p


def build_pool(samples, known, reserved, encoding, progress=None):
    nodes = {}
    blocked_families = {core.guard.source_family(r["source_id"]) for r in [*known, *reserved]}
    blocked_families.update(core.guard.source_family(s["source_id"]) for r in [*known, *reserved] for s in r.get("spans", []))
    blocked_groups = {r["group_id"] for r in [*known, *reserved]}
    blocked_hashes = {r["source_sha256"] for r in [*known, *reserved]}
    blocked_turns = {s["turn_text_sha256"] for r in [*known, *reserved] for s in r.get("spans", [])}
    counts = Counter()

    def node(text):
        sha = text_hash(text)
        if sha not in nodes:
            tokens = len(encoding.encode(text, disallowed_special=()))
            nodes[sha] = {"id": core.digest(NAMESPACE + ":node:" + sha), "source": text,
                "source_sha256": sha, "tokens": tokens, "bin": core.bucket(tokens),
                "families": set(), "groups": set(), "spans": [], "origins": set(),
                "known": False, "candidate": False, "blocked": False}
        return nodes[sha]

    for row in known:
        value = node(row["source"])
        value["known"] = value["blocked"] = True
        value["families"].add(core.guard.source_family(row["source_id"]))
        value["groups"].add(row["group_id"])
        value["spans"].extend(row["spans"])
        value.setdefault("source_id", row["source_id"])
    for sample_number, sample in enumerate(samples, 1):
        if len(sample["haystack_session_ids"]) != len(sample["haystack_sessions"]):
            raise ValueError("public session alignment differs")
        for sid, session in zip(sample["haystack_session_ids"], sample["haystack_sessions"]):
            sid = str(sid)
            for turn in session:
                counts["observed_public_turns"] += 1
                value = node(turn["content"])
                if value["tokens"] < 192:
                    counts["bypass_turns_not_grouped_by_text"] += 1
                    continue
                family = core.guard.source_family(sid)
                value["families"].add(family)
                value["groups"].add(family)
                value["origins"].add((sid, turn["role"]))
                value.setdefault("source_id", sid)
                value["candidate"] = value["bin"] is not None
                if (family in blocked_families or family in blocked_groups
                        or value["source_sha256"] in blocked_hashes or value["source_sha256"] in blocked_turns):
                    value["blocked"] = True
        if progress is not None and (sample_number % 16 == 0 or sample_number == len(samples)):
            write_json(progress, {"phase": "public_pool", "samples": sample_number, "total_samples": len(samples), "unique_texts": len(nodes)})
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > MAX_RSS:
            raise ValueError("public pool exceeds24GB RSS limit")
    result = []
    for value in nodes.values():
        if not value["known"] and not value["origins"]:
            continue
        if value["origins"]:
            sid, role = min(value["origins"])
            value["spans"].append({"source_id": sid, "role": role, "turn_text_sha256": value["source_sha256"]})
        value["group_id"] = min(value["groups"])
        result.append(value)
    return sorted(result, key=lambda r: r["source_sha256"]), dict(counts)


def export(row, component, split):
    sid, role = min(row["origins"], key=lambda x: (core.digest(NAMESPACE + ":origin:" + "|".join(x)), x))
    sha = row["source_sha256"]
    return {"id": row["id"], "source_id": sid, "group_id": component, "source_component": component,
        "source_sha256": sha, "source": row["source"], "source_cl100k_tokens": row["tokens"],
        "bin": row["bin"], "split": split, "spans": [{"source_id": sid, "role": role,
            "turn_id": "public:" + core.digest(sid + "|" + role + "|" + sha),
            "turn_text_sha256": sha, "span_text_sha256": sha,
            "start_char": 0, "end_char": len(row["source"])}]}


def metadata(row):
    return {k: row[k] for k in (*METADATA_FIELDS, "source_component", "source_cl100k_tokens", "bin", "split", "spans")}


def selected_checks(known, selected, samples):
    if len({r["source_component"] for r in selected}) != len(selected):
        raise ValueError("one selected case per complete component required")
    core.guard.require_disjoint(known, selected)
    signatures = [core.guard.signature(r) for r in selected]
    for i, a in enumerate(selected):
        if any(core.guard.overlap(a, b, signatures[i], signatures[j]) is not None for j, b in enumerate(selected[:i])):
            raise ValueError("selected cohorts share source family, public turn or text")
    proof = verify_cases(samples, selected)
    if proof != {r["id"]: r["source_sha256"] for r in selected}:
        raise ValueError("every selected source must be an exact complete public turn")
    return proof


def prepare(pilot=False):
    known, reserved, _ = inputs()
    destination = PILOT if pilot else DEST
    extra = []
    if not pilot:
        terminal(PILOT)
        if overlap.study.reference.q.read(PILOT / "result.json")["status"] != "systems_clean_source_partitions_passed":
            raise ValueError("actual public-source lifecycle pilot required")
        extra = [PILOT / n for n in ("producer.json", "result.json", "pilot_sources.json", "worker_terminal_receipt.json")]
    p = producing(destination, "systems_lifecycle_only_not_quality_evidence" if pilot else "reserve_full_size_clean_sources_before_model_outputs", extra)
    import tiktoken
    samples = json.loads(PUBLIC_FILE.read_text())
    if len(samples) != 500:
        raise ValueError("complete500-sample public release required")
    used_samples = samples[:2] if pilot else samples
    pool, counts = build_pool(used_samples, known, reserved, tiktoken.get_encoding("cl100k_base"), destination / "progress.json")
    graph = core.SourceGraph()
    for i, row in enumerate(pool):
        graph.add(row)
        if i % 512 == 0:
            write_json(destination / "progress.json", {"phase": "components", "nodes": i + 1, "total_nodes": len(pool)})
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > MAX_RSS:
                raise ValueError("source graph exceeds24GB RSS limit")
    components = graph.components()
    edges = dict(graph.edges)
    del graph
    gc.collect()
    if pilot:
        chosen = {}
        for b in FINAL:
            options = [(key, r) for key, value in components.items() if not value["blocked"] and key not in chosen
                for r in value["rows"] if r["candidate"] and r["bin"] == b]
            if options:
                key, row = min(options, key=lambda x: (core.digest(NAMESPACE + ":pilot:" + x[1]["id"]), x[0]))
                chosen[key] = row
        if not chosen:
            raise ValueError("actual public-source pilot needs an eligible fresh component")
        parts = {"pilot": [export(row, key, "pilot") for key, row in chosen.items()]}
        availability = None
    else:
        chosen, availability = core.assign_components(components, {b: FINAL[b] + DEVELOPMENT[b] for b in FINAL}, NAMESPACE)
        assignment = core.split_assignment(chosen, FINAL, DEVELOPMENT, NAMESPACE)
        parts = {split: [export(r, r["source_component"], split) for r in rows] for split, rows in assignment.items()}
    all_selected = [r for values in parts.values() for r in values]
    proof = selected_checks(known, all_selected, samples)
    index = []
    for component, value in sorted(components.items()):
        for row in value["rows"]:
            index.append({"id": row["id"], "source_sha256": row["source_sha256"], "source_component": component,
                "blocked": value["blocked"], "candidate": row["candidate"], "bin": row["bin"],
                "tokens": row["tokens"], "source_characters": len(row["source"]),
                "families": sorted(row["families"]), "groups": sorted(row["groups"]),
                "public_origins": [list(v) for v in sorted(row["origins"])]})
    write_json(destination / "component_index.json", index)
    write_json(destination / "selection_metadata.json", {s: [metadata(r) for r in values] for s, values in parts.items()})
    seals = {}
    for split, rows in parts.items():
        name = "pilot_sources.json" if pilot else split + "_sources.json"
        path = destination / name
        write_json(path, rows)
        sha = file_hash(path)
        if split.startswith("final_"):
            stat = path.stat()
            seals[split] = {"path": str(path), "creation_sha256": sha, "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino, "sealed_after_creation": True}
        elif overlap.study.reference.q.read(path) != rows:
            raise ValueError("exact unsealed source export readback failed")
    write_json(destination / "seal_receipts.json", seals)
    verify(p)
    result = {"status": "systems_clean_source_partitions_passed" if pilot else "clean_full_size_source_partitions_prepared",
        "counts": {k: len(v) for k, v in parts.items()}, "known_sources_excluded": len(known),
        "old_reserved_metadata_excluded": len(reserved), "public_samples": len(used_samples),
        "pool_nodes": len(pool), "components": len(components), "edges": edges, "counts_observed": counts,
        "available_components_by_bin": availability, "component_index_sha256": file_hash(destination / "component_index.json"),
        "selection_metadata_sha256": file_hash(destination / "selection_metadata.json"),
        "seal_receipts_sha256": file_hash(destination / "seal_receipts.json"),
        "selected_exact_public_source_hashes": proof, "exact_selected_text_overlap_pairs": 0,
        "exact_selected_known_overlap_pairs": 0, "max_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "old_sealed_source_files_read_or_hashed": False, "new_final_sources_reopened_after_seal": False,
        "model_execution": False, "labels_generated": False, "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(destination / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "selected_exact_public_source_hashes"}), flush=True)


def terminal(root):
    if overlap.study.reference.q.read(root / "worker_terminal_receipt.json")["exit_code"]:
        raise ValueError("actual successful source worker exit required")
    verify(overlap.study.reference.q.read(root / "producer.json"))


def workflow():
    inputs()
    if any(p.exists() for p in (WORKFLOW, PILOT, DEST, RESULTS, REPORT)):
        raise ValueError("fresh append-only clean partition study required")
    p = producing(WORKFLOW, "supervise_full_size_clean_source_reservation")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    try:
        for stage, root in (("pilot", PILOT), ("prepare", DEST), ("finalize", RESULTS), ("register", RESULTS)):
            command = [sys.executable, "-u", "-m", AUDITOR if stage in ("finalize", "register") else MODULE, stage]
            verify(p)
            child = subprocess.Popen(command, env=os.environ)
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"status": "observed_worker_terminal", "stage": stage, "exit_code": code,
                "observer_pid": os.getpid(), "child_pid": child.pid, "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if root.exists():
                write_json(root / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json"), receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("clean source-partition stage failed: " + stage)
        verify(p)
        journal.update(status="clean_source_partitions_completed", active_stage=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "pilot", "prepare", "workflow"))
    args = parser.parse_args()
    configure()
    if args.stage == "check":
        known, reserved, _ = inputs()
        print(json.dumps({"known_sources": len(known), "reserved_metadata": len(reserved),
            "public_sha256": PUBLIC_SHA256, "final_quotas": FINAL, "development_quotas": DEVELOPMENT,
            "old_sealed_source_files_read_or_hashed": False}))
    elif args.stage == "workflow":
        workflow()
    else:
        prepare(args.stage == "pilot")
