"""Apply unchanged semantic gates after explicit first-judgment cache recovery."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_oracle_codec_semantic_recovery as recovery
from experiments.neural_architecture_lab import finalize_ncm_oracle_codec_semantic as base
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json

DEST = recovery.study.ROOT / "20261023_oracle_codec_semantic_recovery_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-oracle-codec-semantic-recovery.md")


def configure():
    recovery.configure()
    base.DEST, base.REPORT = DEST, REPORT


def finalize():
    configure()
    canonical = recovery.study.q.read(recovery.PREP / "judgment_cache.json")
    cache = recovery.study.q.read(recovery.GRADE / "judgment_cache.json")
    if any(cache.get(key) != entry for key, entry in canonical.items()):
        raise ValueError("all canonical and recovered first judgments must remain exact")
    base.finalize()
    prep = recovery.study.q.read(recovery.PREP / "result.json")
    REPORT.write_text(REPORT.read_text() + f"""

## Recovery of the incomplete cache

The preceding semantic pilot/full run was stopped at observed exit 143 when
identical control packets were found in two omitted label-preparation caches.
Its incomplete full audit is invalid. Later duplicate control judgments were
preserved as evidence and excluded regardless of outcome. The recovery retains
all {prep['canonical_first_judgments']} canonical earlier judgments plus
{prep['recovered_unique_judgments']} new first judgments saved by the stopped
run. It ignores {prep['ignored_later_duplicates']} later duplicates. For
{prep['possibly_consumed_transport_attempts']} identities with queued directories
but no persisted response, at most two further transport attempts were allowed.
No earlier valid reply was rerolled, and the original 320 model-output entries
remain exact. The cases, external packets, rubric and gates did not change.
See the [prospective recovery amendment](../07%20-%20Status%20Reports/2026-10-23_ncm-oracle-codec-semantic-cache-recovery.md).
""")
    provenance = {"status": "finalized_recovery_with_unchanged_gates", "parent_status": "invalid_incomplete_prior_first_judgment_cache",
        "runtime_overrides": {"PILOT": str(recovery.PILOT), "GRADE": str(recovery.GRADE),
            "PLAN": str(recovery.PLAN), "HYPOTHESIS": recovery.HYPOTHESIS,
            "DEST": str(DEST), "REPORT": str(REPORT)},
        "source_hashes": {str(p): file_hash(p) for p in (Path(__file__), Path(base.__file__), Path(recovery.__file__), recovery.PLAN)},
        "campaign_sha256": file_hash(DEST / "campaign_results.json"), "report_sha256": file_hash(REPORT),
        "recovered_cache_entries_preserved": len(canonical), "recovery_preparation": prep}
    write_json(DEST / "recovery_finalization_provenance.json", provenance)
    print(json.dumps({"status": provenance["status"], "preserved_entries": len(canonical)}), flush=True)


def register():
    configure()
    provenance = recovery.study.q.read(DEST / "recovery_finalization_provenance.json")
    for path, expected in provenance["source_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("recovery finalization source changed")
    if file_hash(DEST / "campaign_results.json") != provenance["campaign_sha256"] or file_hash(REPORT) != provenance["report_sha256"]:
        raise ValueError("finalized recovery evidence changed")
    base.register()
    register_records(DEST / "campaign_results.json", REPORT, DEST, {"ncm-oracle-codec-semantic-quality-v1": {
        "name": "Stopped semantic codec audit with incomplete prior first-judgment cache",
        "description": "Parent pilot/full audit stopped at exit143 after omitted label-preparation caches caused repeated uncompressed-control judgments. Invalid parent preserved. Canonical earlier decisions and genuinely new saved codec judgments are carried into a separately registered recovery, without selecting by outcome.",
        "confirmation_status": "invalid_incomplete_prior_first_judgment_cache",
        "task_accuracy_measured": False, "oracle_target_codes": True, "source_prediction": False,
        "NCM_success_established": False, "observed_worker_exit": 143,
        "parent_invalidation": str(recovery.OLD_GRADE / "scientific_invalidation.json"),
        "recovery_hypothesis": recovery.HYPOTHESIS}})
    print(json.dumps({"status": "parent_invalidity_and_recovery_registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
