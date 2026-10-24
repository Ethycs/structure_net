"""Independently replay complete clean DEV metadata and frozen source tensors."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_clean_development_features as run
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def check_feature(row, entry, path):
    if (entry["path"] != str(path) or entry["sha256"] != file_hash(path)
            or entry["source_tokens"] != len(row["source_ids"])):
        raise ValueError("fixed feature path, bytes or complete source length differs")
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (set(saved) != {"id", "source_sha256", "alignment_sha256", "native_prompt_sha256", "features"}
            or any(saved[k] != row[k] for k in ("id", "source_sha256", "alignment_sha256", "native_prompt_sha256"))):
        raise ValueError("saved feature source/prompt/alignment identity differs")
    tensor = saved["features"]
    if (tensor.shape != (len(row["source_ids"]), 3072) or tensor.dtype != torch.float32
            or tensor.requires_grad or tensor.grad_fn is not None or not torch.isfinite(tensor).all()):
        raise ValueError("finite detached FP32 full-source features required")


def finalize():
    run.configure()
    parts, source_inputs = run.admitted_sources()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh complete DEV feature aggregate/report required")
    for root in (run.DATA, run.PILOT, run.FEATURES):
        run.base.terminal(root)
    receipts = [run.WORKFLOW / "stage_receipts" / (s + ".json")
        for s in ("prepare", "feature-pilot", "features")]
    if any(run.q.read(p)["exit_code"] or run.q.read(p)["observer"] != "subprocess.wait" for p in receipts):
        raise ValueError("three observed successful numerical children required")
    native, gpt, storage = run.tokenizers()
    replayed = {split: [run.prepare_row(r, split, native, gpt, storage) for r in values]
        for split, values in parts.items()}
    prepared = run.q.read(run.DATA / "cases.json")
    result = run.q.read(run.DATA / "result.json")
    if (prepared != replayed or result["status"] != "clean128_DEV_metadata_prepared"
            or result["counts"] != run.EXPECTED or result["cases_sha256"] != file_hash(run.DATA / "cases.json")
            or not result["all_sources_complete"] or not result["source_only"]
            or result["task_accuracy"] is not None or result["final_test_opened"]):
        raise ValueError("all128 canonical prompt/source/offset/frame records required")
    proof = run.base.verify_public([r for values in parts.values() for r in values])
    if run.q.read(run.DATA / "public_proof.json") != {
            "public_file_sha256": run.base.PUBLIC_SHA256, "exact_sources": proof, "matched": 128}:
        raise ValueError("complete exact public source proof differs")
    selected = run.select_pilot(replayed)
    if run.q.read(run.DATA / "selection.json") != {"feature_pilot_ids": selected}:
        raise ValueError("fixed first and longest native-prompt pilot selection differs")
    rows = [r for values in prepared.values() for r in values]
    if (result["max_native_prompt"] != max(len(r["context_ids"]) for r in rows)
            or result["max_GPT2_source"] != max(len(r["source_ids"]) for r in rows)
            or result["max_source_frame"] != max(r["source_frame"] for r in rows)):
        raise ValueError("complete prompt/source geometry maxima differ")
    pilot = run.q.read(run.PILOT / "manifest.json")
    manifest = run.q.read(run.FEATURES / "manifest.json")
    if (set(pilot) != set(selected) or set(manifest) != {r["id"] for r in rows}
            or any(manifest[key] != entry for key, entry in pilot.items())):
        raise ValueError("complete128 manifest with both exact retained pilot files required")
    for root, expected, status in ((run.PILOT, 2, "systems_feature_identity_passed_not_quality"),
            (run.FEATURES, 128, "frozen_source_features_prepared")):
        value = run.q.read(root / "result.json")
        if (value["status"] != status or value["cases"] != expected
                or value["parent_body_hash_unchanged"] != run.base.EXPECTED_BODY
                or not value["all_source_tokens_retained"] or value["peak_allocated_bytes"] > 8_600_000_000
                or value["task_accuracy"] is not None or value["manifest_sha256"] != file_hash(root / "manifest.json")):
            raise ValueError("original frozen28-layer body, full features and actual memory controls required")
    for row in rows:
        root = run.PILOT if row["id"] in pilot else run.FEATURES
        check_feature(row, manifest[row["id"]], root / "features" / (row["id"] + ".pt"))
    if not run.forecaster_ready():
        raise ValueError("actual successful serialized upstream forecaster lifecycle required")
    workflow = run.q.read(run.WORKFLOW / "producer.json")
    run.base.verify(workflow)
    upstream_path = run.forecaster.RESULTS / "campaign_results.json"
    observed_path = run.WORKFLOW / "upstream_terminal_observation.json"
    observed = run.q.read(observed_path)
    if (observed["upstream_pid"] != workflow["config"]["upstream_pid"]
            or observed["captured_kernel_identity"] != workflow["config"]["upstream_kernel_identity"]
            or observed["same_kernel_still_live"] or not observed["complete_registered_lifecycle"]
            or observed["campaign_sha256"] != file_hash(upstream_path)
            or observed["observer"] != "proc_stat_PID_and_start_ticks"):
        raise ValueError("actual upstream kernel termination observation required")
    artifacts = {str(p): file_hash(p) for p in [Path(__file__), Path(run.__file__), run.PLAN,
        run.WORKFLOW / "producer.json", observed_path, *receipts, *source_inputs, upstream_path,
        run.forecaster.RESULTS / (run.forecaster.HYPOTHESIS + "_registry_readback.json")]}
    artifacts.update(workflow["input_hashes"])
    for root in (run.DATA, run.PILOT, run.FEATURES):
        producer = run.q.read(root / "producer.json")
        artifacts.update(producer["input_hashes"])
        artifacts.update({str(p): file_hash(p) for p in root.rglob("*")
            if p.is_file() and "producer_source" not in p.parts})
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": "clean128_DEV_frozen_features_prepared",
        "counts": run.EXPECTED, "parent_body_sha256": run.base.EXPECTED_BODY,
        "metadata": result, "feature_cases": len(manifest), "retained_pilot_cases": len(pilot),
        "all_sources_complete": True, "source_only": True, "target_labels_used": False,
        "original_parent_and_MLPs_frozen": True, "task_accuracy": None, "confirmed": False,
        "NCM_success_established": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": artifacts, "producer_fingerprint": workflow["fingerprint"]}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    run.REPORT.write_text(f"""# Frozen Llama source features for fresh NCM development

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **{campaign['classification']}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`.

All64 short and64 long development passages retain their complete public source
text, component IDs and spans. Canonical native prompts, GPT2 round trips,
character overlap alignment and source frames replay independently. No labels,
generated summaries or sealed final passages enter preparation.

The original28-layer Llama3.2-3B body and MLPs remain frozen. BF16 source forwards
produce detached aligned FP32 tensors for every GPT2 source token. The first
short passage and longest native prompt pass exact repeated extraction, then
their saved files are reused unchanged in the complete128-case cache after a
fresh parent load. Both workers verify original body identity and the8.6GB peak
allocation limit. Feature extraction follows the observed successful terminal
forecaster workflow, with no simultaneous body/head GPU allocation.

The CPU auditor checks every actual saved tensor's identity, complete shape,
precision, detached state, finiteness and bytes, and every observed child exit.
This establishes evaluation feature readiness. Summary quality remains
unmeasured here. The95% complete-summary goal and required semantic evaluations
remain unchanged; no speed gain is established.
""")
    print(json.dumps({"classification": campaign["classification"], "feature_cases": len(manifest),
        "campaign_sha256": file_hash(path)}), flush=True)


def register():
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Complete fresh DEV frozen Llama source features",
        "description": "All128 clean source-component-disjoint development passages, complete canonical prompts and detached FP32 original frozen28-layer Llama features; no summary quality measured.",
        "classification": "clean128_DEV_frozen_features_prepared"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    finalize() if args.stage == "finalize" else register()
