"""Archive wider unlabeled sources without opening either sealed text cohort."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_application_range_sources as s
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_application_source_pool_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-wide-public-source-pool.md")
DECISION = ROOT / "20261023_application_source_quota_decision_v1.json"


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh append-only result root required")
    producers = []
    for root in (s.AUDIT, s.POOL):
        if s.q.read(root / "worker_terminal_receipt.json")["exit_code"] != 0:
            raise ValueError("both source workers must be observed terminal")
        producer = s.q.read(root / "producer.json")
        verify(producer); producers.append(producer)
    availability, pool = s.q.read(s.AUDIT / "result.json"), s.q.read(s.POOL / "result.json")
    if availability["status"] != "availability_audited" or pool["status"] != "unlabeled_public_pool_prepared":
        raise ValueError("complete public source preparation required")
    quotas = s.q.read(DECISION)["quotas_per_bin"]
    if quotas != pool["per_bin_quotas"]:
        raise ValueError("preselection quotas changed")
    chosen = s.select(s.q.read(s.AUDIT / "eligible_index.json"), quotas)
    flat = [row for rows in chosen.values() for row in rows]
    blocked = s.forbidden()
    if any(len({r[key] for r in flat}) != len(flat) or any(r[key] in blocked[key] for r in flat) for key in s.IDENTITY):
        raise ValueError("group/session/hash/ID overlap")
    if pool["unique_groups"] != len(flat) or pool["counts"] != {name: len(rows) for name, rows in chosen.items()}:
        raise ValueError("prepared counts and deterministic metadata replay differ")
    # Hash the new sealed file; never load or display its source text.
    for path, expected in pool["files"].items():
        if file_hash(path) != expected:
            raise ValueError("prepared source file changed")
    allowed = {*s.IDENTITY, "source", "spans", "source_cl100k_tokens", "bin", "native_prompt_tokens", "pool_split"}
    for name in ("train", "diagnostic"):
        rows = s.q.read(s.POOL / (name + "_sources.json"))
        if [r["id"] for r in rows] != [r["id"] for r in chosen[name]] or any(set(r) != allowed for r in rows):
            raise ValueError("unsealed export identities or source-only schema differ")
    paths = [Path(__file__), Path(s.__file__), s.PLAN, DECISION,
             Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in (s.AUDIT, s.POOL):
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    artifacts = {str(p): file_hash(p) for p in dict.fromkeys(paths)}
    for producer in producers:
        artifacts.update(producer["input_hashes"]); artifacts.update(producer["source_hashes"])
    DEST.mkdir()
    write_json(DEST / "selection_metadata_replay.json", chosen)
    artifacts[str(DEST / "selection_metadata_replay.json")] = file_hash(DEST / "selection_metadata_replay.json")
    aggregate = {"schema": "nal.ncm-wide-source-pool.v1", "classification": "wider_public_sources_prepared_quality_unmeasured",
        "availability": availability, "pool": pool, "metadata_selection_replayed": True,
        "source_group_overlap": False, "old_targets_exported": False, "training_admitted": False,
        "new_sealed_source_texts_read_by_finalizer": False, "original_final_test_opened": False,
        "task_accuracy_measured": False, "NCM_success_established": False, "confirmed": False,
        "speed_tested": False, "artifacts": artifacts}
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    REPORT.write_text(f'''# Wider public source pool prepared; no model quality measured

Date: 2026-10-23. Exploratory data preparation,
`wider_public_sources_prepared_quality_unmeasured`.

The full original training partition contains 204,967 rows. After length,
known-source and exact canonical-public-span checks, 14,638 rows from 3,860
unused groups remain. Availability by cl100k source-length band:

| Tokens per complete source | Eligible rows | Unique groups |
| --- | --- | --- |
| 512–767 | {availability['rows_by_bin'][0]} | {availability['groups_by_bin'][0]} |
| 768–1023 | {availability['rows_by_bin'][1]} | {availability['groups_by_bin'][1]} |
| 1024–1535 | {availability['rows_by_bin'][2]} | {availability['groups_by_bin'][2]} |
| 1536–2048 | {availability['rows_by_bin'][3]} | {availability['groups_by_bin'][3]} |

The rarest band has 69 groups, 18 shared with the longest band. Before any
pool model outputs or judgments, fixed quotas reserved 16 sealed-final, four
diagnostic and 28 training sources per band: **64 sealed, 16 diagnostic and
112 training sources**, from 192 distinct groups. Deterministic metadata replay
verified no ID/session/group/source-hash overlap or known-source reuse.

Complete selected sources contain {pool['source_cl100k_min']}–{pool['source_cl100k_max']}
cl100k tokens; native prompts contain {pool['native_prompt_min']}–{pool['native_prompt_max']}
tokens including unchanged native chat instructions. No truncation or short
native-context filter was used. Stored source lengths and exact public spans
were verified during preparation. Old saved targets were excluded from exports.
The finalizer checked export hashes and metadata without loading sealed texts.

This operational holdout comes from unused groups inside the original training
partition; it is not an untouched original benchmark partition. Both source
workers terminated successfully. Original sealed 256 cases and the new sealed
64 cases have no new outputs or labels from this preparation. The currently
running short-source NCM experiment remains unchanged and does not use this pool.

The pool is unlabeled and admits no new training run. It addresses source
availability, not 95% accuracy, batch/hierarchy quality, native memory fit,
compression or speed. Future labels, long-context lifecycle tests and complete
source-disjoint task evaluation need their own fixed protocols.

Aggregate `{campaign}`, SHA256 `{file_hash(campaign)}`. Confirmed=false.
''')
    print(json.dumps({"status": "finalized", "classification": aggregate["classification"], "counts": pool["counts"]}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {"ncm-wide-public-source-pool-v1": {
        "name": "Disjoint public longer-fragment source pool",
        "description": "Public-span-only data preparation across512-2048cl100k:112train,16diagnostic,64newsealed sources,192uniqueunusedgroups. No oldtargets or sealedtext readback. No modelgeneration, trainingadmission, taskaccuracy or speedclaim.",
        "confirmation_status": "wider_public_sources_prepared_quality_unmeasured", "task_accuracy_measured": False,
        "NCM_success_established": False, "training_admitted": False}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
