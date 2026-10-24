"""Replay exchange-partition metadata without opening sealed final content."""
import argparse
import json

from experiments.structure_net import ncm_clean_exchange_partitions as run
from experiments.neural_architecture_lab.finalize_ncm_clean_source_partitions import seal_stat
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def check_origin(origin, inventory, expected_characters):
    session = inventory["sessions"][origin["session_sha256"]]
    start, end = origin["start_turn"], origin["end_turn"]
    turns = session["turns"]
    if (session["source_id"] != origin["source_id"] or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(turns)):
        raise ValueError("literal public session interval differs")
    if origin["kind"] == "whole_turn":
        if end != start + 1:
            raise ValueError("whole-turn origin must contain one complete turn")
    elif origin["kind"] == "exchange_window":
        if (start and turns[start]["role"] != "user") or (end < len(turns) and turns[end]["role"] != "user"):
            raise ValueError("complete user-led exchange boundaries required")
    else:
        raise ValueError("known source construction required")
    spans = []
    for turn in turns[start:end]:
        sha, sid, role = turn["sha256"], origin["source_id"], turn["role"]
        if inventory["turns"][sha] != {k: turn[k] for k in ("sha256", "characters", "tokens")}:
            raise ValueError("public full-turn inventory differs")
        spans.append({"source_id": sid, "role": role, "turn_id": "public:" + run.text_hash(sid + "|" + role + "|" + sha),
            "turn_text_sha256": sha, "span_text_sha256": sha, "start_char": 0, "end_char": turn["characters"]})
    characters = sum(t["characters"] for t in turns[start:end]) + end - start - 1
    if origin["spans"] != spans or characters != expected_characters:
        raise ValueError("complete public spans or joined source length differ")


def replay_assignment(index):
    components = {}
    for row in index:
        value = components.setdefault(row["source_component"], {"blocked": row["blocked"], "rows": []})
        if (value["blocked"] != row["blocked"] or row["bin"] != run.core.bucket(row["tokens"])
                or row["id"] != run.core.digest(run.NAMESPACE + ":node:" + row["source_sha256"])):
            raise ValueError("source component, token band or stable node differs")
        value["rows"].append(row)
    if any(key != min(r["source_sha256"] for r in value["rows"]) for key, value in components.items()):
        raise ValueError("complete component identity differs")
    chosen, availability = run.core.assign_components(components, {b: run.FINAL[b] + run.DEVELOPMENT[b] for b in run.FINAL}, run.NAMESPACE)
    return run.core.split_assignment(chosen, run.FINAL, run.DEVELOPMENT, run.NAMESPACE), availability


def finalize():
    known, reserved, _ = run.inputs()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh exchange source aggregate/report required")
    q = run.prior.overlap.study.reference.q
    artifacts = {}
    for root in (run.PILOT, run.DEST):
        run.terminal(root)
        result = q.read(root / "result.json")
        for name, expected in result["file_hashes"].items():
            if name not in ("component_index.json", "selection_metadata.json", "seal_receipts.json", "public_inventory.json"):
                raise ValueError("only unsealed metadata belongs in generic rehashing")
            path = root / name
            if file_hash(path) != expected:
                raise ValueError("bound source metadata changed")
            artifacts[str(path)] = expected
        if (result["known_sources"] != len(known) or result["old_reserved_metadata"] != 320
                or result["model_outputs_or_labels_used"] or result["task_accuracy"] is not None
                or result["old_sealed_source_files_read_or_hashed"] or result["new_final_sources_reopened_after_seal"]
                or result["final_test_opened"] or result["speed_tested"] or result["max_RSS_bytes"] > run.MAX_RSS
                or result["selected_known_text_overlap_pairs"] or result["selected_pairwise_text_overlap_pairs"]):
            raise ValueError("scope, full exclusion, exact guard or memory evidence differs")
        for name in ("producer.json", "result.json", "worker_terminal_receipt.json"):
            path = root / name
            artifacts[str(path)] = file_hash(path)
    pilot = q.read(run.PILOT / "result.json")
    pilot_sources = q.read(run.PILOT / "pilot_sources.json")
    if (pilot["status"] != "systems_exchange_partitions_passed" or pilot["public_samples"] != 2
            or not 1 <= len(pilot_sources) <= 5 or pilot["counts"] != {"pilot": len(pilot_sources)}):
        raise ValueError("actual public exchange lifecycle pilot required")
    artifacts[str(run.PILOT / "pilot_sources.json")] = file_hash(run.PILOT / "pilot_sources.json")
    result = q.read(run.DEST / "result.json")
    index = q.read(run.DEST / "component_index.json")
    inventory = q.read(run.DEST / "public_inventory.json")
    counts = {key: v["tokens"] for key, v in inventory["turns"].items()}
    parts, availability = replay_assignment(index)
    selection = q.read(run.DEST / "selection_metadata.json")
    expected = {"final_short": 256, "final_long": 64, "development_short": 64, "development_long": 64}
    if (result["status"] != "clean_exchange_sources_reserved" or result["public_samples"] != 500
            or result["counts"] != expected or {s: len(v) for s, v in selection.items()} != expected
            or result["available_components_by_bin"] != {str(k): v for k, v in availability.items()}
            or len(index) != result["pool_nodes"] or len({r["id"] for r in index}) != len(index)
            or len({r["source_sha256"] for r in index}) != len(index)
            or len({r["source_component"] for r in index}) != result["components"]
            or not {r["source_sha256"] for r in known} <= {r["source_sha256"] for r in index}):
        raise ValueError("full graph, known exclusions and128/320 quotas required")
    protected_families = {run.core.guard.source_family(v) for r in [*known, *reserved]
        for v in [r["source_id"], *[s["source_id"] for s in r.get("spans", [])]]}
    protected_groups = {r["group_id"] for r in [*known, *reserved]}
    protected_hashes = {r["source_sha256"] for r in [*known, *reserved]}
    protected_turns = {s["turn_text_sha256"] for r in [*known, *reserved] for s in r.get("spans", [])
        if counts[s["turn_text_sha256"]] >= 192}
    blocked = {r["source_component"] for r in index if (set(r["families"]) & protected_families
        or set(r["groups"]) & protected_groups or r["source_sha256"] in protected_hashes
        or set(r["informative_turns"]) & protected_turns)}
    by_id = {r["id"]: r for r in index}
    all_metadata = []
    for split, selected in parts.items():
        rows = selection[split]
        if [r["id"] for r in rows] != [r["id"] for r in selected]:
            raise ValueError("fixed hashed component/row/cohort allocation differs")
        for row in rows:
            original = by_id[row["id"]]
            origin = run.choose_origin(original["public_origins"])
            check_origin(origin, inventory, original["source_characters"])
            if (row["public_origin"] != origin or row["source_id"] != origin["source_id"] or row["spans"] != origin["spans"]
                    or row["source_sha256"] != original["source_sha256"] or row["source_component"] != original["source_component"]
                    or row["group_id"] != row["source_component"] or row["source_component"] in blocked or original["blocked"]
                    or row["source_cl100k_tokens"] != original["tokens"] or row["bin"] != original["bin"]
                    or row["split"] != split or row["source_truncated"]):
                raise ValueError("literal source origin, component exclusion or capacity differs")
            informative = {s["turn_text_sha256"] for s in origin["spans"] if counts[s["turn_text_sha256"]] >= 192}
            if not informative <= set(original["informative_turns"]):
                raise ValueError("informative public-turn aliases omitted")
        if split.startswith("development_"):
            path = run.DEST / (split + "_sources.json")
            sources = q.read(path)
            if [run.metadata(r) for r in sources] != rows or any(run.text_hash(r["source"]) != r["source_sha256"] for r in sources):
                raise ValueError("unsealed development bytes differ")
            artifacts[str(path)] = file_hash(path)
        all_metadata += rows
    if (len({r["source_component"] for r in all_metadata}) != 448
            or result["selected_exact_public_source_hashes"] != {r["id"]: r["source_sha256"] for r in all_metadata}):
        raise ValueError("complete448 independent source components required")
    seals = q.read(run.DEST / "seal_receipts.json")
    if set(seals) != {"final_short", "final_long"}:
        raise ValueError("full newly sealed final cohorts required")
    for split, receipt in seals.items():
        seal_stat(receipt, run.DEST / (split + "_sources.json"))
    for stage in ("pilot", "prepare"):
        path = run.WORKFLOW / "stage_receipts" / (stage + ".json")
        if q.read(path)["exit_code"]:
            raise ValueError("actual successful source-preparation child exits required")
        artifacts[str(path)] = file_hash(path)
    producer = q.read(run.WORKFLOW / "producer.json")
    run.prior.verify(producer)
    artifacts.update(producer["input_hashes"])
    artifacts[str(run.WORKFLOW / "producer.json")] = file_hash(run.WORKFLOW / "producer.json")
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": "clean_exchange_sources_reserved_quality_unmeasured",
        "counts": expected, "available_components_by_bin": availability, "pool_nodes": result["pool_nodes"],
        "components": result["components"], "known_sources_excluded": len(known), "old_reserved_metadata_excluded": 320,
        "new_final_creation_receipts": seals, "sealed_final_files_reopened_by_auditor": False,
        "audit_scope": "independent metadata/quota/origin/dev-bytes/stat replay; exact full-text checks executed by frozen producer before sealing",
        "model_outputs_or_labels_used": False, "task_accuracy": None, "confirmed": False, "NCM_success_established": False,
        "final_test_opened": False, "speed_tested": False, "artifacts": artifacts, "producer_fingerprint": producer["fingerprint"]}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    table = "\n".join(f"| {s} | {n} |" for s, n in expected.items())
    run.REPORT.write_text(f"""# Full NCM evaluation sources reserved from public exchanges

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **{campaign['classification']}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`.

| Source partition | Cases |
| --- | --- |
{table}

Candidates contain whole public turns or contiguous complete user-led exchanges.
All constituent turn text and literal spans are retained. Selection precedes
model outputs and labels and uses no quality scores. Source components join
families, known group aliases, normalized text and exact word5gram overlap
(at least32 shared and80% shorter containment). Word and gram IDs use exact
dictionary equality. All known sources and old final metadata are excluded.

Shared-turn-hash aliases apply only to full turns of at least192 cl100k tokens,
so generic bypass greetings do not join unrelated families. Complete source
text still undergoes normalization and overlap checks; all source-family IDs
and all original provenance spans remain. The original whole-turn-only quota
failure stays failed. Full128 development and320 final quotas are unchanged;
both long cohorts retain16 sources in each of four512–2048-token bands.

The preparation worker checked exact public spans and direct full-text guards
against known sources and across all448 selected cases before sealing. The CPU
auditor independently replayed metadata, complete exchange boundaries, quota
selection, development bytes and final creation/stat receipts. Final files were
not reopened. Their content hashes and text guards need independent checks
when final evaluation becomes eligible. Neither original sealed file was opened.

No student quality, labels or speed improvement is established here. Future
development needs61/64 same cases passing both qualified judges per cohort;
final needs244/256 short and61/64 long, all valid, after fixed-candidate
development and original frozen-body integration checks. The95% task goal
remains unmet. All prior failures and canonical first judgments remain intact.
""")
    print(json.dumps({"classification": campaign["classification"], "counts": expected,
        "campaign_sha256": file_hash(path)}), flush=True)


def register():
    run.prior.configure()
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Full size clean NCM exchange source partitions",
        "description": "Full128 development and320 sealed final public sources, complete exchange fragments, exact component exclusion; no student accuracy measured.",
        "classification": "clean_exchange_sources_reserved_quality_unmeasured"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.prior.configure()
    finalize() if args.stage == "finalize" else register()
