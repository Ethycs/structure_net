"""Audit full690 concept continuation and the complete source-only history cache."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_dense_matched_forecaster as run
from experiments.neural_architecture_lab import finalize_ncm_matched_forecaster as reference
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def finalize():
    run.configure()
    reference.run = run.old
    rows, _, selected = run.data_inputs()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh dense-matched forecaster aggregate/report required")
    study = run.old.clean.recovery.study
    for root in (run.PILOT, run.FIT_PILOT, run.TRAIN, run.FIT, run.CACHE):
        study.data.terminal(root)
    receipts = [run.WORKFLOW / "stage_receipts" / (s + ".json")
        for s in ("train_pilot", "fit_pilot", "train", "fit", "cache")]
    if any(study.q.read(p)["exit_code"] for p in receipts):
        raise ValueError("five observed successful numerical stages required")
    trained = study.q.read(run.TRAIN / "result.json")
    if (trained["updates"] != run.Config().updates or trained["training_pool_cases"] != 690
            or not trained["reload_exact"] or not trained["source_only_reload_probe_exact"]
            or trained["peak_allocated_bytes"] > 8_000_000_000
            or trained["checkpoint_sha256"] != file_hash(run.TRAIN / "model.pt")):
        raise ValueError("fixed full-population training and exact reload required")
    for root, steps, cases in ((run.PILOT, 2, 2), (run.TRAIN, run.Config().updates, 690)):
        value = study.q.read(root / "result.json")
        if (value["updates"] != steps or value["cases"] != cases or value["training_pool_cases"] != 690
                or value["initial_parameter_sha256"] != run.loader.FORECASTER_PARAMETERS
                or value["frozen_codec_sha256_unchanged"] != run.loader.ORIGINAL_CODEC
                or not value["word_modules_finest_head_and_codec_frozen"]
                or not value["reload_exact"] or not value["source_only_reload_probe_exact"]
                or value["peak_allocated_bytes"] > 8_000_000_000 or value["task_accuracy"] is not None
                or value["NCM_success_established"] or value["final_test_opened"] or value["speed_tested"]):
            raise ValueError("complete learned initializer, frozen scope or actual lifecycle controls differ")
    state = torch.load(run.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    if state["producer_fingerprint"] != trained["producer_fingerprint"]:
        raise ValueError("checkpoint producing identity differs")
    trace = [json.loads(line) for line in (run.TRAIN / "trace.jsonl").open()]
    replay = reference.replay_training(rows, trace, state)
    expected_snapshots = {f"step_{i:06d}.pt" for i in range(run.Config().snapshot_interval,
        run.Config().updates, run.Config().snapshot_interval)}
    if {p.name for p in (run.TRAIN / "checkpoints").glob("*.pt")} != expected_snapshots:
        raise ValueError("all fixed intermediate recovery states must remain")
    initial = torch.load(study.bootstrap.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)["model"]
    names = state["frozen_parameter_names"]
    frozen = {name: state["model"][name] for name in names}
    expected_frozen = study.q.read(study.bootstrap.TRAIN / "result.json")["frozen_parameter_sha256_unchanged"]
    if (not names or len(names) != len(set(names)) or any(not torch.equal(value, initial[name]) for name, value in frozen.items())
            or study.base.tensor_digest(frozen) != expected_frozen
            or trained["frozen_parameter_sha256_unchanged"] != expected_frozen):
        raise ValueError("original word, finest-head or other frozen parameters changed")
    fitted = reference.replay_fit(selected)
    if (fitted["checkpoint_sha256"] != trained["checkpoint_sha256"]
            or not fitted["parameters_and_codec_unchanged"] or fitted["peak_allocated_bytes"] > 8_000_000_000):
        raise ValueError("TRAIN fitting used a different final forecaster")
    cached = study.q.read(run.CACHE / "result.json")
    histories = study.q.read(run.CACHE / "histories.json")
    sources = study.q.read(run.CACHE / "source_cases.json")
    if (cached["status"] != "matched_dense_TRAIN_histories_prepared" or cached["cases"] != 690
            or set(histories) != {r["id"] for r in rows} or sources != [study.source_view(r) for r in rows]
            or cached["histories_sha256"] != file_hash(run.CACHE / "histories.json")
            or cached["source_cases_sha256"] != file_hash(run.CACHE / "source_cases.json")
            or cached["checkpoint_sha256"] != trained["checkpoint_sha256"]
            or not cached["source_only"] or cached["target_labels_in_prediction"] or not cached["codec_unchanged"]
            or cached["peak_allocated_bytes"] > 8_000_000_000):
        raise ValueError("complete fixed-checkpoint source-only history cache required")
    for row in rows:
        study.validate_history(row, histories[row["id"]])
        if {"word_ids", "concept_ids", "target_text", "target", "text"} & histories[row["id"]].keys():
            raise ValueError("target/candidate labels cannot enter the source-only cache")
    values = study.q.read(run.FIT / "case_measurements.json")
    if any(histories[v["id"]]["coarse_codes"] != v["coarse_codes"] for v in values):
        raise ValueError("fresh full-cache histories differ from registered fitting path")
    selected_model, selected_codec, _ = run.model(run.TRAIN / "model.pt", run.Config().updates, device="cpu")
    if (cached["forecaster_parameter_sha256"] != study.parameter_hash(selected_model)
            or run.loader.module_hash(selected_codec) != run.loader.ORIGINAL_CODEC):
        raise ValueError("full actual selected model/codec differs from cache identity")
    artifacts = [Path(__file__), Path(run.__file__), run.PLAN, run.WORKFLOW / "producer.json", *receipts,
        run.targets.RESULTS / "campaign_results.json", run.targets.RESULTS / (run.targets.HYPOTHESIS + "_registry_readback.json")]
    for root in (run.PILOT, run.FIT_PILOT, run.TRAIN, run.FIT, run.CACHE):
        artifacts += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    classification = ("matched_dense_forecaster_coarse_fit_passed_word_quality_unmeasured" if fitted["forecaster_fit_gate"]
        else "matched_dense_forecaster_coarse_fit_failed_word_quality_unmeasured")
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": classification, "config": asdict(run.Config()),
        "training": trained, "replay": replay, "fitting": fitted, "source_only_cache": cached,
        "forecaster_fit_gate": fitted["forecaster_fit_gate"], "TRAIN_cases": 690,
        "admission": "registered_full690_dense_latent_targets", "prior_declined_forecaster_unchanged": True,
        "dense_word_adaptation_still_required": True, "semantic_quality_measured": False, "task_accuracy": None,
        "NCM_success_established": False, "confirmed": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(artifacts)}}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    table = "\n".join(f"| {group} | {v['level']} | {v['active_correct']}/{v['active_total']} |"
        for group, values in fitted["components"].items() for v in values["levels"])
    run.REPORT.write_text(f"""# Concept forecasting trained on the complete dense target population

Date: 2026-10-24. Exploratory hypothesis `{run.HYPOTHESIS}`.
Classification **{classification}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`.

| Fixed TRAIN probes | Coarse level | Correct active codes |
| --- | --- | --- |
{table}

The new forecaster continued the exact learned252-case initializer for32768
fixed updates on all690 approved TRAIN summaries, with a fresh optimizer.
The earlier conditional forecaster remains declined. This exploratory study
instead followed qualified full690 dense target construction. The original
Llama body/MLPs, codec, word modules and finest head remained frozen. Source
features retained every token. FP32 masters/optimizer/loss and reconstruction,
BF16 forward, original dropout and disabled TF32 remained fixed. Uniform history
corruption and weighted active/padded concept losses follow the existing recipe.
Only the fixed final checkpoint was selected.

Maximum-frame/max-LR gradients, exact model/optimizer/RNG/native reload and
source-only native-generation pilots preceded full training. The CPU audit
replayed all samples, histories, target hashes, masks, rates, weighted losses,
optimizer steps and sampler/noise states, preserved snapshots and frozen bytes.
All690 fresh native-frame source-only concept histories were cached; the fixed
TRAIN probes agree exactly with separate fitting outputs.

The99% active-code gate per level and TRAIN group is a component diagnostic.
It establishes no summary accuracy and does not automatically admit downstream
training. Dense word adaptation needs a separate fixed protocol using these
actual histories and qualified targets, followed by fresh source-disjoint
complete-summary evaluation. The last student's raw development scores were
0/64 short and1/64 long, with the lone pass from a TRAIN-overlapping passage.
The95% task goal remains unmet; sealed final sources stay closed. No speed
improvement is established by this component.
""")
    print(json.dumps({"classification": classification, "campaign_sha256": file_hash(path)}), flush=True)


def register():
    run.configure()
    path = run.RESULTS / "campaign_results.json"
    campaign = run.old.clean.recovery.study.q.read(path)
    register_records(path, run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Full690 dense-target population concept continuation",
        "description": "Fixed32768 updates from learned252-case forecaster on all690 approved TRAIN cases, frozen original body/codec/word modules; exact source-only histories for dense adaptation. No semantic quality measured.",
        "classification": campaign["classification"], "forecaster_fit_gate": campaign["forecaster_fit_gate"],
        "task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.configure()
    finalize() if args.stage == "finalize" else register()
