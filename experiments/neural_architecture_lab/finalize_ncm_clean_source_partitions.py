"""Audit full-size partition metadata without reopening sealed final sources."""
import argparse
import json

from experiments.structure_net import ncm_clean_source_partitions as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def replay_assignment(index):
    components = {}
    for row in index:
        key = row["source_component"]
        value = components.setdefault(key, {"blocked": row["blocked"], "rows": []})
        if (value["blocked"] != row["blocked"] or row["bin"] != run.core.bucket(row["tokens"])
                or row["id"] != run.core.digest(run.NAMESPACE + ":node:" + row["source_sha256"])):
            raise ValueError("complete component/block/length identity differs")
        value["rows"].append(row)
    for key, value in components.items():
        if key != min(r["source_sha256"] for r in value["rows"]):
            raise ValueError("stable complete source-component identity differs")
    chosen, availability = run.core.assign_components(components,
        {b: run.FINAL[b] + run.DEVELOPMENT[b] for b in run.FINAL}, run.NAMESPACE)
    return run.core.split_assignment(chosen, run.FINAL, run.DEVELOPMENT, run.NAMESPACE), availability


def seal_stat(receipt, path):
    if receipt["path"] != str(path) or not receipt["sealed_after_creation"] or len(receipt["creation_sha256"]) != 64:
        raise ValueError("original final source creation receipt required")
    stat = path.stat()
    if (stat.st_size != receipt["size_bytes"] or stat.st_mtime_ns != receipt["mtime_ns"]
            or stat.st_ino != receipt["inode"]):
        raise ValueError("sealed source stat changed after creation; do not open it")


def finalize():
    known, reserved, _ = run.inputs()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh full-size source aggregate/report required")
    q = run.overlap.study.reference.q
    artifacts = {}
    for root in (run.PILOT, run.DEST):
        run.terminal(root)
        result = q.read(root / "result.json")
        for name, key in (("component_index.json", "component_index_sha256"),
                ("selection_metadata.json", "selection_metadata_sha256"), ("seal_receipts.json", "seal_receipts_sha256")):
            path = root / name
            if file_hash(path) != result[key]:
                raise ValueError("bound source metadata changed")
            artifacts[str(path)] = file_hash(path)
        if (result["old_reserved_metadata_excluded"] != 320 or result["known_sources_excluded"] != len(known)
                or result["old_sealed_source_files_read_or_hashed"] or result["new_final_sources_reopened_after_seal"]
                or result["model_execution"] or result["labels_generated"] or result["task_accuracy"] is not None
                or result["final_test_opened"] or result["speed_tested"] or result["max_RSS_bytes"] > run.MAX_RSS
                or result["exact_selected_text_overlap_pairs"] or result["exact_selected_known_overlap_pairs"]):
            raise ValueError("source preparation validity, scope or memory controls differ")
        for n in ("producer.json", "result.json", "worker_terminal_receipt.json"):
            p = root / n
            artifacts[str(p)] = file_hash(p)
    pilot = q.read(run.PILOT / "result.json")
    if pilot["status"] != "systems_clean_source_partitions_passed" or pilot["public_samples"] != 2:
        raise ValueError("actual two-public-sample lifecycle pilot required")
    pilot_rows = q.read(run.PILOT / "pilot_sources.json")
    if len(pilot_rows) != pilot["counts"]["pilot"] or not pilot_rows or len(pilot_rows) > 5:
        raise ValueError("complete nonempty pilot source export required")
    artifacts[str(run.PILOT / "pilot_sources.json")] = file_hash(run.PILOT / "pilot_sources.json")
    result = q.read(run.DEST / "result.json")
    index = q.read(run.DEST / "component_index.json")
    if (len(index) != result["pool_nodes"] or len({r["id"] for r in index}) != len(index)
            or len({r["source_sha256"] for r in index}) != len(index)
            or not {r["source_sha256"] for r in known} <= {r["source_sha256"] for r in index}
            or len({r["source_component"] for r in index}) != result["components"]):
        raise ValueError("complete unique source graph and all known exclusions required")
    parts, availability = replay_assignment(index)
    selection = q.read(run.DEST / "selection_metadata.json")
    expected_counts = {"final_short": 256, "final_long": 64, "development_short": 64, "development_long": 64}
    if (result["status"] != "clean_full_size_source_partitions_prepared" or result["public_samples"] != 500
            or result["counts"] != expected_counts or {s: len(v) for s, v in selection.items()} != expected_counts
            or result["available_components_by_bin"] != {str(k): v for k, v in availability.items()}):
        raise ValueError("unchanged full128 development and320 final allocation required")
    blocked_families = {run.core.guard.source_family(r["source_id"]) for r in [*known, *reserved]}
    blocked_families.update(run.core.guard.source_family(s["source_id"]) for r in [*known, *reserved] for s in r.get("spans", []))
    blocked_groups = {r["group_id"] for r in [*known, *reserved]}
    blocked_hashes = {r["source_sha256"] for r in [*known, *reserved]}
    blocked_turns = {s["turn_text_sha256"] for r in [*known, *reserved] for s in r.get("spans", [])}
    by_id = {r["id"]: r for r in index}
    blocked_components = {r["source_component"] for r in index if (set(r["families"]) & blocked_families
        or set(r["groups"]) & blocked_groups or r["source_sha256"] in blocked_hashes | blocked_turns)}
    all_metadata = []
    for split, values in parts.items():
        actual = selection[split]
        if [r["id"] for r in values] != [r["id"] for r in actual]:
            raise ValueError("fixed hashed component/row/cohort assignment differs")
        for row in actual:
            original = by_id[row["id"]]
            sid, role = min(original["public_origins"],
                key=lambda x: (run.core.digest(run.NAMESPACE + ":origin:" + "|".join(x)), x))
            sha = original["source_sha256"]
            expected_span = {"source_id": sid, "role": role, "turn_id": "public:" + run.core.digest(sid + "|" + role + "|" + sha),
                "turn_text_sha256": sha, "span_text_sha256": sha, "start_char": 0, "end_char": original["source_characters"]}
            if (row["source_id"] != sid or row["spans"] != [expected_span] or row["source_sha256"] != sha
                    or row["source_component"] != original["source_component"] or row["group_id"] != row["source_component"]
                    or row["source_component"] in blocked_components or original["blocked"]
                    or row["bin"] != original["bin"] or row["source_cl100k_tokens"] != original["tokens"]
                    or row["split"] != split or result["selected_exact_public_source_hashes"].get(row["id"]) != sha):
                raise ValueError("source identity, complete span, exclusion or length binding differs")
        if split.startswith("development_"):
            path = run.DEST / (split + "_sources.json")
            sources = q.read(path)
            if [run.metadata(r) for r in sources] != actual:
                raise ValueError("unsealed development sources differ from exact selected metadata")
            for row in sources:
                if run.text_hash(row["source"]) != row["source_sha256"]:
                    raise ValueError("unsealed source bytes changed")
            artifacts[str(path)] = file_hash(path)
        all_metadata += actual
    if len({r["source_component"] for r in all_metadata}) != 448:
        raise ValueError("all448 evaluation sources need distinct complete components")
    if result["selected_exact_public_source_hashes"] != {r["id"]: r["source_sha256"] for r in all_metadata}:
        raise ValueError("exact complete selected public-source proof required")
    seals = q.read(run.DEST / "seal_receipts.json")
    if set(seals) != {"final_short", "final_long"}:
        raise ValueError("exact two newly sealed full-size final source files required")
    for split, receipt in seals.items():
        seal_stat(receipt, run.DEST / (split + "_sources.json"))
    for stage in ("pilot", "prepare"):
        p = run.WORKFLOW / "stage_receipts" / (stage + ".json")
        if q.read(p)["exit_code"] != 0:
            raise ValueError("actual successful source-preparation exits required")
        artifacts[str(p)] = file_hash(p)
    producer = q.read(run.WORKFLOW / "producer.json")
    run.verify(producer)
    artifacts.update(producer["input_hashes"])
    artifacts[str(run.WORKFLOW / "producer.json")] = file_hash(run.WORKFLOW / "producer.json")
    run.RESULTS.mkdir()
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": "clean_full_size_sources_reserved_quality_unmeasured",
        "counts": expected_counts, "available_components_by_bin": availability, "known_sources_excluded": len(known),
        "old_reserved_metadata_excluded": 320, "component_count": result["components"], "pool_nodes": result["pool_nodes"],
        "new_final_creation_receipts": seals, "sealed_files_reopened_by_auditor": False,
        "independent_audit_scope": "metadata/component/quota/export/stat replay; exact full-text checks ran before sealing in frozen producer",
        "old_sealed_source_files_read_or_hashed": False, "model_training_or_inference": False,
        "task_accuracy": None, "NCM_success_established": False, "confirmed": False, "final_test_opened": False,
        "speed_tested": False, "artifacts": artifacts, "producer_fingerprint": producer["fingerprint"]}
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    table = "\n".join(f"| {s} | {n} |" for s, n in expected_counts.items())
    run.REPORT.write_text(f"""# Full size NCM sources reserved with stronger split controls

Date: 2026-10-23. Hypothesis `{run.HYPOTHESIS}`.
Classification **{campaign['classification']}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`.

| New source partition | Cases |
| --- | --- |
{table}

Complete public turns are grouped before selection by source families, old
group aliases, public-turn identity, normalized text and substantial exact
word5gram overlap. Connected components propagate exclusion through aliases
and overlap chains. All known TRAIN/development passages and the metadata of
the old256 short and64 long final sources are excluded. The full public release
is used without labels, model outputs or quality filtering. Deterministic quota
matching selects one case per component. Long cohorts retain16 sources in each
512–767,768–1023,1024–1535 and1536–2048 cl100k band; short sources span192–511.

The frozen preparation worker checked complete literal public spans and exact
text disjointness against known sources and across all448 selected cases before
sealing final exports. The CPU auditor replayed component/identity metadata,
quotas, deterministic selections, complete spans, development export bytes and
final creation/stat receipts. It did not reopen final source files or claim an
independent post-seal full-text replay. Final content hashes and text guards
must be checked when final evaluation becomes eligible.

This prospective replacement preserves the full128 development and320 final
sizes and the95% complete-summary target after the previous contamination
finding. Original closed final files and all failed results remain unchanged.
Future development requires61/64 same cases passing both qualified judges in
each cohort; final requires244/256 short and61/64 long, all judgments valid.
The newly sealed final cohorts stay closed until a fixed candidate passes
development and original frozen-body integration checks. No labels, student
quality, model training or speed gain is established by source reservation.
""")
    print(json.dumps({"classification": campaign["classification"], "counts": expected_counts,
        "campaign_sha256": file_hash(path)}), flush=True)


def register():
    run.configure()
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Full size component-disjoint NCM source reservation",
        "description": "Fresh128 development and320 sealed final public sources, complete range and stronger family/turn/text component exclusion; no model quality measured.",
        "classification": "clean_full_size_sources_reserved_quality_unmeasured"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.configure()
    finalize() if args.stage == "finalize" else register()
