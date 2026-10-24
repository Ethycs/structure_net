"""Reserve full evaluation cohorts using public turns and complete exchanges."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

from experiments.structure_net import ncm_clean_source_partitions as prior
from experiments.structure_net import ncm_public_exchange_fragments as fragments
from experiments.structure_net.ncm_compact_source_graph import CompactSourceGraph
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

ROOT = prior.ROOT
HYPOTHESIS = "ncm-clean-exchange-source-partitions-v2"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-clean-exchange-partitions-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-clean-exchange-partitions.md")
WORKFLOW = ROOT / "20261024_clean_exchange_partitions_workflow_v2"
PILOT = ROOT / "20261024_clean_exchange_partitions_pilot_v2"
DEST = ROOT / "20261024_clean_exchange_partitions_v2"
RESULTS = ROOT / "20261024_clean_exchange_partitions_results_v2"
MODULE = "experiments.structure_net.ncm_clean_exchange_partitions"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_clean_exchange_partitions"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_clean_exchange_partitions.py")
FAILURE = ROOT / "20261024_clean_source_partition_failure_results_v1"
FAILURE_SHA = "733f237ccf38130e6fc3d735f96401ffd3781b76493f13d9c07444035d4f803f"
NAMESPACE = "ncm-clean-exchange-source-partitions-v2"
FINAL, DEVELOPMENT = prior.FINAL, prior.DEVELOPMENT
core = prior.core
MAX_RSS = prior.MAX_RSS


def inputs():
    known, reserved, paths = prior.inputs()
    q = prior.overlap.study.reference.q
    path = FAILURE / "campaign_results.json"
    receipt = q.read(FAILURE / (prior.HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(path) != FAILURE_SHA or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != FAILURE_SHA
            or receipt["exact"]["campaign_sha256"] != FAILURE_SHA
            or q.read(path)["classification"] != "insufficient_component_disjoint_whole_turn_sources"):
        raise ValueError("registered unchanged whole-turn availability failure required")
    return known, reserved, [*paths, path, FAILURE / (prior.HYPOTHESIS + "_registry_readback.json")]


def producing(root, stage, extra=()):
    _, _, paths = inputs()
    files = [Path(__file__), PLAN, AUDITOR_FILE, Path(prior.__file__), prior.PLAN,
        Path(core.__file__), Path(core.guard.__file__), Path(fragments.__file__),
        Path(CompactSourceGraph.add.__code__.co_filename), Path(prior.verify_cases.__code__.co_filename),
        Path("experiments/neural_architecture_lab/finalize_ncm_clean_source_partitions.py"),
        Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    p = prior.freeze(root, stage, [*paths, *files, *extra],
        {"hypothesis_id": HYPOTHESIS, "CPU_only": True, "namespace": NAMESPACE,
         "final_quotas": FINAL, "development_quotas": DEVELOPMENT, "maximum_RSS_bytes": MAX_RSS,
         "candidate_sources": "whole_public_turns_and_complete_contiguous_user_exchanges_192_to2048cl",
         "shared_turn_hash_minimum_full_turn_tokens": 192,
         "literal_source_and_provenance_spans_retained": True,
         "grouping": "exact_dictionary_word_and5gram_ids_with_original_overlap_thresholds",
         "model_outputs_or_labels_used": False, "quality_filtering": False,
         "old_sealed_source_files_read_or_hashed": False, "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"exchange_partitions_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = prior.overlap.study.reference.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def origin_key(origin):
    return json.dumps(origin, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def choose_origin(origins):
    return min(origins, key=lambda o: (core.digest(NAMESPACE + ":origin:" + origin_key(o)), origin_key(o)))


class CountingEncoding:
    def __init__(self, encoding):
        self.encoding, self.counts = encoding, {}

    def encode(self, text, **kwargs):
        sha = text_hash(text)
        if sha not in self.counts:
            self.counts[sha] = len(self.encoding.encode(text, **kwargs))
        return range(self.counts[sha])


def inventory(samples, encoding):
    turns, sessions = {}, {}
    for sample in samples:
        if len(sample["haystack_session_ids"]) != len(sample["haystack_sessions"]):
            raise ValueError("public session alignment differs")
        for sid, session in zip(sample["haystack_session_ids"], sample["haystack_sessions"]):
            values = []
            for turn in session:
                sha = text_hash(turn["content"])
                value = {"sha256": sha, "characters": len(turn["content"]),
                    "tokens": len(encoding.encode(turn["content"], disallowed_special=()))}
                if sha in turns and turns[sha] != value:
                    raise ValueError("public turn hash collision")
                turns[sha] = value
                values.append({"role": turn["role"], **value})
            key = text_hash(str(sid) + "|" + json.dumps(values, sort_keys=True, ensure_ascii=False))
            sessions[key] = {"source_id": str(sid), "turns": values}
    return turns, sessions


def pool(samples, known, reserved, encoding, turns, sessions, progress=None):
    full_counts = {key: v["tokens"] for key, v in turns.items()}
    for row in [*known, *reserved]:
        if any(s["turn_text_sha256"] not in turns for s in row.get("spans", [])):
            raise ValueError("all prior full-turn provenance needs a public token count")
    blocked_families = {core.guard.source_family(r["source_id"]) for r in [*known, *reserved]}
    blocked_families.update(core.guard.source_family(s["source_id"]) for r in [*known, *reserved] for s in r.get("spans", []))
    blocked_groups = {r["group_id"] for r in [*known, *reserved]}
    blocked_hashes = {r["source_sha256"] for r in [*known, *reserved]}
    blocked_turns = {s["turn_text_sha256"] for r in [*known, *reserved] for s in r.get("spans", [])
        if full_counts[s["turn_text_sha256"]] >= 192}
    nodes, seen_sessions = {}, set()

    def add(text, source_id, spans, *, old_group=None, origin=None, is_known=False):
        sha = text_hash(text)
        if sha not in nodes:
            tokens = len(encoding.encode(text, disallowed_special=()))
            nodes[sha] = {"id": core.digest(NAMESPACE + ":node:" + sha), "source": text,
                "source_sha256": sha, "source_id": source_id, "tokens": tokens,
                "bin": core.bucket(tokens), "blocked": False, "candidate": False,
                "groups": set(), "families": set(), "span_aliases": set(), "origins": {}}
        value = nodes[sha]
        families = {core.guard.source_family(v) for v in [source_id, *[s["source_id"] for s in spans]]}
        value["families"].update(families)
        value["groups"].update(families)
        if old_group is not None:
            value["groups"].add(old_group)
        for span in spans:
            if full_counts[span["turn_text_sha256"]] >= 192:
                value["span_aliases"].add((span["source_id"], span["role"], span["turn_text_sha256"]))
        if origin is not None:
            value["origins"][origin_key(origin)] = origin
            value["candidate"] = value["bin"] is not None
        if (is_known or families & blocked_families or value["groups"] & blocked_groups
                or sha in blocked_hashes or any(s["turn_text_sha256"] in blocked_turns for s in spans)):
            value["blocked"] = True

    for row in known:
        add(row["source"], row["source_id"], row["spans"], old_group=row["group_id"], is_known=True)
    for number, sample in enumerate(samples, 1):
        for sid_, session in zip(sample["haystack_session_ids"], sample["haystack_sessions"]):
            sid = str(sid_)
            turn_meta = [{"role": t["role"], **turns[text_hash(t["content"]) ]} for t in session]
            session_sha = text_hash(sid + "|" + json.dumps(turn_meta, sort_keys=True, ensure_ascii=False))
            if sessions[session_sha] != {"source_id": sid, "turns": turn_meta}:
                raise ValueError("exact ordered public session inventory required")
            if session_sha in seen_sessions:
                continue
            seen_sessions.add(session_sha)
            for i, turn in enumerate(session):
                sha = text_hash(turn["content"])
                if turns[sha]["tokens"] < 192:
                    continue
                spans = [{"source_id": sid, "role": turn["role"], "turn_id": "public:" + text_hash(sid + "|" + turn["role"] + "|" + sha),
                    "turn_text_sha256": sha, "span_text_sha256": sha, "start_char": 0, "end_char": len(turn["content"])}]
                add(turn["content"], sid, spans, origin={"source_id": sid, "spans": spans, "kind": "whole_turn",
                    "session_sha256": session_sha, "start_turn": i, "end_turn": i + 1})
            for row in fragments.windows(sid, session, encoding, namespace=NAMESPACE):
                add(row["source"], sid, row["spans"], origin={"source_id": sid, "spans": row["spans"], "kind": "exchange_window",
                    "session_sha256": session_sha, "start_turn": row["exchange_start_turn"], "end_turn": row["exchange_end_turn"]})
        if progress is not None and (number % 16 == 0 or number == len(samples)):
            write_json(progress, {"phase": "pool", "samples": number, "total_samples": len(samples), "nodes": len(nodes)})
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > MAX_RSS:
            raise ValueError("exchange pool exceeds24GiB RSS")
    result = []
    for value in nodes.values():
        value["group_id"] = min(value["groups"])
        value["spans"] = [{"source_id": sid, "role": role, "turn_text_sha256": sha}
            for sid, role, sha in sorted(value.pop("span_aliases"))]
        result.append(value)
    return sorted(result, key=lambda r: r["source_sha256"]), len(seen_sessions)


def export(row, component, split):
    origin = choose_origin(list(row["origins"].values()))
    return {"id": row["id"], "source_id": origin["source_id"], "source_sha256": row["source_sha256"],
        "group_id": component, "source_component": component, "source": row["source"],
        "spans": origin["spans"], "public_origin": origin, "source_cl100k_tokens": row["tokens"],
        "bin": row["bin"], "split": split, "source_truncated": False}


def metadata(row):
    return {k: v for k, v in row.items() if k != "source"}


def selected_checks(known, values, samples, turns):
    counts = {k: v["tokens"] for k, v in turns.items()}
    views = [fragments.comparison_view(r, counts) for r in values]
    core.guard.require_disjoint([fragments.comparison_view(r, counts) for r in known], views)
    signatures = [core.guard.signature(r) for r in views]
    for i, a in enumerate(views):
        if any(core.guard.overlap(a, b, signatures[i], signatures[j]) is not None for j, b in enumerate(views[:i])):
            raise ValueError("selected sources share informative turns, families or text")
    proof = prior.verify_cases(samples, values)
    if proof != {r["id"]: r["source_sha256"] for r in values} or len({r["source_component"] for r in values}) != len(values):
        raise ValueError("complete distinct components and exact literal public spans required")
    return proof


def prepare(pilot=False):
    known, reserved, _ = inputs()
    root = PILOT if pilot else DEST
    extra = []
    if not pilot:
        terminal(PILOT)
        if prior.overlap.study.reference.q.read(PILOT / "result.json")["status"] != "systems_exchange_partitions_passed":
            raise ValueError("actual exchange-source lifecycle pilot required")
        extra = [PILOT / n for n in ("producer.json", "result.json", "pilot_sources.json", "worker_terminal_receipt.json")]
    p = producing(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "full_exchange_component_source_reservation", extra)
    import tiktoken
    encoding = CountingEncoding(tiktoken.get_encoding("cl100k_base"))
    samples = json.loads(prior.PUBLIC_FILE.read_text())
    if len(samples) != 500:
        raise ValueError("complete500-sample public release required")
    turns, sessions = inventory(samples, encoding)
    write_json(root / "public_inventory.json", {"turns": turns, "sessions": sessions})
    nodes, session_count = pool(samples[:2] if pilot else samples, known, reserved, encoding,
        turns, sessions, root / "progress.json")
    encoding.counts.clear()
    graph = CompactSourceGraph()
    for i, row in enumerate(nodes):
        graph.add(row)
        if i % 512 == 0:
            write_json(root / "progress.json", {"phase": "components", "nodes": i + 1, "total_nodes": len(nodes)})
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > MAX_RSS:
                raise ValueError("exact exchange graph exceeds24GiB RSS")
    components = graph.components()
    graph_counts = {"word_ids": len(graph.words), "gram_ids": len(graph.grams), "edges": dict(graph.edges)}
    del graph
    gc.collect()
    if pilot:
        chosen = {}
        for band in FINAL:
            options = [(key, r) for key, value in components.items() if not value["blocked"] and key not in chosen
                for r in value["rows"] if r["candidate"] and r["bin"] == band]
            if options:
                key, row = min(options, key=lambda x: (core.digest(NAMESPACE + ":pilot:" + x[1]["id"]), x[0]))
                chosen[key] = row
        if not chosen:
            raise ValueError("actual public exchange pilot needs an eligible source")
        parts, availability = {"pilot": [export(r, key, "pilot") for key, r in chosen.items()]}, None
    else:
        chosen, availability = core.assign_components(components, {b: FINAL[b] + DEVELOPMENT[b] for b in FINAL}, NAMESPACE)
        assignment = core.split_assignment(chosen, FINAL, DEVELOPMENT, NAMESPACE)
        parts = {s: [export(r, r["source_component"], s) for r in rows] for s, rows in assignment.items()}
    values = [r for rows in parts.values() for r in rows]
    proof = selected_checks(known, values, samples, turns)
    index = []
    for key, value in sorted(components.items()):
        for row in value["rows"]:
            index.append({"id": row["id"], "source_sha256": row["source_sha256"], "source_component": key,
                "blocked": value["blocked"], "candidate": row["candidate"], "bin": row["bin"],
                "tokens": row["tokens"], "source_characters": len(row["source"]), "families": sorted(row["families"]),
                "groups": sorted(row["groups"]), "informative_turns": sorted({s["turn_text_sha256"] for s in row["spans"]}),
                "public_origins": [row["origins"][k] for k in sorted(row["origins"])]})
    write_json(root / "component_index.json", index)
    write_json(root / "selection_metadata.json", {s: [metadata(r) for r in rows] for s, rows in parts.items()})
    seals = {}
    for split, rows in parts.items():
        path = root / ("pilot_sources.json" if pilot else split + "_sources.json")
        write_json(path, rows)
        sha = file_hash(path)
        if split.startswith("final_"):
            stat = path.stat()
            seals[split] = {"path": str(path), "creation_sha256": sha, "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino, "sealed_after_creation": True}
        elif prior.overlap.study.reference.q.read(path) != rows:
            raise ValueError("unsealed source export reload differs")
    write_json(root / "seal_receipts.json", seals)
    prior.verify(p)
    result = {"status": "systems_exchange_partitions_passed" if pilot else "clean_exchange_sources_reserved",
        "counts": {s: len(rows) for s, rows in parts.items()}, "public_samples": 2 if pilot else 500,
        "known_sources": len(known), "old_reserved_metadata": len(reserved), "pool_nodes": len(nodes),
        "components": len(components), "session_variants": session_count, "graph": graph_counts,
        "available_components_by_bin": availability, "selected_exact_public_source_hashes": proof,
        "selected_known_text_overlap_pairs": 0, "selected_pairwise_text_overlap_pairs": 0,
        "max_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "file_hashes": {n: file_hash(root / n) for n in
            ("component_index.json", "selection_metadata.json", "seal_receipts.json", "public_inventory.json")},
        "model_outputs_or_labels_used": False, "old_sealed_source_files_read_or_hashed": False,
        "new_final_sources_reopened_after_seal": False, "final_test_opened": False, "task_accuracy": None,
        "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    if result["max_RSS_bytes"] > MAX_RSS:
        raise ValueError("source reservation peak RSS exceeds24GiB")
    write_json(root / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "selected_exact_public_source_hashes"}), flush=True)


def terminal(root):
    prior.terminal(root)


def workflow():
    inputs()
    if any(p.exists() for p in (WORKFLOW, PILOT, DEST, RESULTS, REPORT)):
        raise ValueError("fresh exchange partition study required")
    p = producing(WORKFLOW, "supervise_full_size_exchange_source_partitions")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    try:
        for stage, root in (("pilot", PILOT), ("prepare", DEST), ("finalize", RESULTS), ("register", RESULTS)):
            command = [sys.executable, "-u", "-m", AUDITOR if stage in ("finalize", "register") else MODULE, stage]
            prior.verify(p)
            child = subprocess.Popen(command, env=os.environ)
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"stage": stage, "exit_code": code, "child_pid": child.pid,
                "observer_pid": os.getpid(), "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if root.exists():
                write_json(root / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json"), receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("exchange source stage failed: " + stage)
        prior.verify(p)
        journal.update(status="clean_exchange_partitions_completed", active_stage=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "pilot", "prepare", "workflow"))
    args = parser.parse_args()
    prior.configure()
    if args.stage == "check":
        known, reserved, _ = inputs()
        print(json.dumps({"known": len(known), "reserved_metadata": len(reserved), "final": FINAL,
            "development": DEVELOPMENT, "model_outputs_or_labels_used": False}))
    elif args.stage == "workflow":
        workflow()
    else:
        prepare(args.stage == "pilot")
