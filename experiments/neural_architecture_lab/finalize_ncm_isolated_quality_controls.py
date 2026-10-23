"""Archive passed pretraining controls and the preserved v1 validation failure."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_isolated_quality_data as data
from experiments.structure_net import ncm_isolated_quality_cache_v2 as cache
from experiments.structure_net import ncm_isolated_quality_train as training
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_isolated_quality_controls_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-isolated-quality-pretraining-controls.md")
FAILED = ROOT / "20261023_isolated_quality_cache_pilot_v1"


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh control finalization required")
    roots = (data.DEST, cache.PILOT, cache.CACHE, training.PILOT)
    for root in roots:
        data.terminal(root)
    results = {root.name: data.run.q.read(root / "result.json") for root in roots}
    metadata, c_pilot, full_cache, t_pilot = [results[r.name] for r in roots]
    if (metadata["cases"] != 252 or metadata["old_short"] != 143 or metadata["new_long"] != 109
            or not metadata["source_disjointness_verified"] or metadata["max_native_prompt"] != 2276
            or c_pilot["status"] != "systems_cache_identity_passed_not_quality_evidence"
            or full_cache["status"] != "mixed_length_source_caches_prepared" or full_cache["cases"] != 252
            or full_cache["reused_old"] != 143 or full_cache["max_source_frame"] != 2304
            or t_pilot["status"] != "systems_clean_quality_training_passed_not_quality_evidence"
            or t_pilot["max_source_frame_seen"] != 2304 or not t_pilot["zero_parent_identity_short_and_long_exact"]
            or not t_pilot["checkpoint_optimizer_RNG_output_reload_exact"]):
        raise ValueError("complete mixed-length/cache/real-gradient controls required")
    failed = data.run.q.read(FAILED / "failure.json")
    if data.run.q.read(FAILED / "worker_terminal_receipt.json")["exit_code"] != 1 or failed["training_started"]:
        raise ValueError("original validation failure must remain explicit")
    cases = data.run.q.read(data.DEST / "cases.json")
    selection = data.run.q.read(data.DEST / "selection.json")
    longest = sorted(cases[143:], key=lambda r: (-r["native_prompt_tokens"], r["id"]))[0]
    if selection["pilot_ids"] != [cases[0]["id"], longest["id"]]:
        raise ValueError("pilot did not cover actual longest selected input")
    manifest = data.run.q.read(cache.CACHE / "manifest.json")
    if file_hash(cache.CACHE / "manifest.json") != full_cache["manifest_sha256"] or manifest["ids"] != [r["id"] for r in cases]:
        raise ValueError("complete252 source-cache ordering differs")
    paths = [Path(__file__), Path(data.__file__), Path(cache.__file__), Path(training.__file__), data.PLAN,
        cache.AMENDMENT, Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in (*roots, FAILED):
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts and p.suffix != ".pt"]
    artifacts = {str(p): file_hash(p) for p in dict.fromkeys(paths)}
    for entry in manifest["cases"].values():
        artifacts[entry["path"]] = entry["sha256"]
    for root in (cache.PILOT, training.PILOT):
        for path in root.rglob("*.pt"):
            artifacts[str(path)] = file_hash(path)
    aggregate = {"schema": "nal.ncm-isolated-quality-controls.v1",
        "classification": "mixed_length_cache_and_isolated_word_training_controls_passed",
        "controls": results, "original_v1_cache_failure": failed, "failed_worker_exit": 1,
        "original_failed_sources_preserved": True, "full_fixed_training_admitted": True,
        "training_config": training.asdict(training.Config()), "selected_model_after_training": "final512only",
        "quality_teacher_control": True, "task_accuracy_measured": False,
        "NCM_success_established": False, "confirmed": False, "final_test_opened": False,
        "speed_tested": False, "artifacts": artifacts}
    DEST.mkdir(); campaign = DEST / "campaign_results.json"; write_json(campaign, aggregate)
    REPORT.write_text(f'''# Isolated quality training: data and maximum-frame controls passed

Date: 2026-10-23. Classification
`mixed_length_cache_and_isolated_word_training_controls_passed`.
Aggregate `{campaign}`, SHA256 `{file_hash(campaign)}`.

All 252 vetted complete-passage training summaries passed source identity,
provenance, target round-trip and128-token capacity checks:143old short labels
and109longer-source labels. Saved atom snippets were not silently treated as
complete-passage summaries. Longer training sources span512–2048cl100k tokens;
their longest native chat prompt contains2,276tokens and uses a2,304-token frame.

The split audit found no ID/session/group/exact-source-hash overlap with
development or diagnostic/new-final metadata. All252identities match the
original training partition. Original256final ID/hash metadata was derived
from the earlier public proof, inheriting session/group exclusion from the
original partition; its source texts were not opened. Neither were the newer
64sealed source texts. Full selected sources and targets were not clipped.

The first cache pilot failed a systems check: token IDs were passed where
the native query method requires embeddings. Its exit-one receipt, source,
failure record and partial files remain intact. A tiny actual Llama CPU test
caught the corrected contract. The v2 pilot recomputed the longest cache and
passed exact native-query save/reload checks. The full v2 cache covers all252
cases, retaining143old cache files and computing109new ones. Original source
backbone weights remained unchanged.

The separate real gradient pilot performed two clean word updates, one on
the first short case and one on the absolute longest selected source. Before
training, the zero word bank matched cached parent logits exactly on both.
Only4,587,520new word Q/K/V/O parameters received gradients. Original MLPs,
embeddings, word head and trained concept bank stayed frozen. Parent/body,
concept predictions, books and CPU cache passed integrity checks, along with
exact parameter/optimizer/RNG/output reload. Peak allocated GPU memory was
{t_pilot['peak_allocated_bytes']:,}bytes.

These controls admit the fixed512-update clean quality-teacher experiment;
they do not establish that it finished, reached95%, or trained a working NCM
replacement. The last measured converged hybrid remains41/64complete-summary
passes; unchanged3B scored52/64. Full unseen-summary grading after training
is required. Acceleration distillation and all final evaluation remain later
gates. No speed or compression claim follows from these systems results.
''')
    print(json.dumps({"status": "finalized", "classification": aggregate["classification"]}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {"ncm-isolated-quality-pretraining-controls-v1": {
        "name": "Vetted mixed-length source cache and isolated word gradient controls",
        "description": "252accepted whole-summary labels (143short+109long), no heldout source overlap; max2276native prompt in2304frame. Preserved rank2-ID/rank3-embedding cachevalidation failure, correctedactualbackbone regressiontest, exactv2reload, andrealshort/longwordgradient/optimizer/RNG checks. Admits fixed512qualityteachercontrol, not modelaccuracy or NCMsuccess.",
        "confirmation_status": "mixed_length_cache_and_isolated_word_training_controls_passed",
        "task_accuracy_measured": False, "full_fixed_training_admitted": True,
        "NCM_success_established": False}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
