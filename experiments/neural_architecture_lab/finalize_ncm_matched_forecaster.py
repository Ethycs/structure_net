"""Audit the complete matched-population source-concept training component."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_matched_forecaster as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def replay_training(rows, trace, state):
    study = run.clean.recovery.study
    config = run.Config()
    if (len(trace) != config.updates or state["step"] != config.updates
            or state["config"] != asdict(config)
            or state["initial_forecaster_checkpoint_sha256"] != run.clean.INITIAL_FORECASTER):
        raise ValueError("fixed complete matched-forecaster checkpoint required")
    labels = {row["id"]: torch.tensor([row["concept_ids"]], dtype=torch.long) for row in rows}
    label_hash = {key: study.base.tensor_digest({"codes": value}) for key, value in labels.items()}
    word_hash = {row["id"]: study.base.tensor_digest({"words":
        torch.tensor([row["word_ids"]], dtype=torch.long)}) for row in rows}
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    noise = torch.Generator().manual_seed(config.noise_seed)
    counts = Counter()
    corrupted_count = 0
    for step, item in enumerate(trace, 1):
        row = rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        corrupted = bool(torch.rand((), generator=noise) < config.corrupt_probability)
        history = run.core.uniform_history(labels[row["id"]], run.WIDTHS + (192,), noise,
            config.maximum_uniform_rate) if corrupted else labels[row["id"]]
        history = run.prediction_history(history)
        history_hash = study.base.tensor_digest({"codes": history})
        if (item["step"] != step or item["id"] != row["id"]
                or item["source_frame"] != row["source_frame"] or item["corrupted"] != corrupted
                or item["history_sha256"] != history_hash
                or item["target_concept_sha256"] != label_hash[row["id"]]
                or item["target_word_sha256"] != word_hash[row["id"]]
                or item["lr"] != run.clean.rate(step, config)):
            raise ValueError("source, full frame, actual/target history, word mask or rate replay differs")
        if (not math.isfinite(item["loss"]) or item["loss"] < 0
                or not math.isfinite(item["gradient_norm"]) or item["gradient_norm"] < 0
                or len(item["levels"]) != 8):
            raise ValueError("finite full hierarchy loss and gradients required")
        weighted = 0.
        for level, values in enumerate(item["levels"]):
            if (values["level"] != level or any(not math.isfinite(values[key]) or values[key] < 0
                    for key in ("active_CE", "padded_CE"))):
                raise ValueError("finite per-level active and padding loss required")
            weighted += (config.root_weight if level == 0 else 1.) * (
                values["active_CE"] + config.padded_weight * values["padded_CE"])
        if not math.isclose(item["loss"], weighted / (config.root_weight + 7), rel_tol=2e-6, abs_tol=1e-7):
            raise ValueError("weighted concept objective differs")
        counts[row["id"]] += 1
        corrupted_count += corrupted
    optimizer = state["optimizer"]
    if (set(counts) != {row["id"] for row in rows}
            or not torch.equal(sampler.get_state(), state["sampler_rng"])
            or not torch.equal(noise.get_state(), state["noise_rng"])
            or not optimizer["state"]
            or any(int(value["step"]) != config.updates for value in optimizer["state"].values())):
        raise ValueError("full coverage, sampler/noise state or optimizer steps differ")
    for group in optimizer["param_groups"]:
        if (tuple(group["betas"]) != (config.beta1, config.beta2)
                or group["weight_decay"] != config.weight_decay or group["eps"] != config.optimizer_eps):
            raise ValueError("serialized optimizer configuration differs")
    return {"updates": len(trace), "sample_counts": dict(sorted(counts.items())),
        "corrupted_updates": corrupted_count, "every_TRAIN_case_sampled": True,
        "all_sources_frames_actual_and_target_histories_word_masks_rates_replayed": True,
        "weighted_objectives_replayed": True, "sampler_noise_and_optimizer_steps_exact": True}


def replay_fit(selected):
    study = run.clean.recovery.study
    result = study.q.read(run.FIT / "result.json")
    measurements = study.q.read(run.FIT / "case_measurements.json")
    if len(measurements) != 32 or [row["id"] for row in measurements] != [row["id"] for row in selected]:
        raise ValueError("complete fixed32 source-only TRAIN predictions required")
    for actual, row in zip(measurements, selected):
        if (actual["source_id"] != row["source_id"] or actual["source_sha256"] != row["source_sha256"]
                or actual["source_frame"] != row["source_frame"]
                or actual["diagnostic_group"] != row["diagnostic_group"]
                or actual["levels"] != run.count_codes(actual["coarse_codes"], row)):
            raise ValueError("source-only concept predictions or active code counts differ")
    summary = run.aggregate(measurements)
    if (summary != result["components"] or result["forecaster_fit_gate"] != run.fitting_gate(summary)
            or file_hash(run.FIT / "case_measurements.json") != result["measurements_sha256"]):
        raise ValueError("matched forecaster aggregation, raw hash or fixed gate differs")
    return result


def finalize():
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh matched forecaster archive and report required")
    rows, _, selected = run.data_inputs()
    study = run.clean.recovery.study
    for root in (run.PILOT, run.FIT_PILOT, run.TRAIN, run.FIT):
        study.data.terminal(root)
    receipts = [run.WORKFLOW / "stage_receipts" / (stage + ".json")
        for stage in ("train_pilot", "fit_pilot", "train", "fit")]
    if any(study.q.read(path)["exit_code"] != 0 for path in receipts):
        raise ValueError("every numerical worker needs an observed zero exit")
    trained = study.q.read(run.TRAIN / "result.json")
    if (trained["updates"] != run.Config().updates or trained["training_pool_cases"] != 690
            or not trained["reload_exact"] or not trained["source_only_reload_probe_exact"]
            or trained["peak_allocated_bytes"] > 8_000_000_000
            or file_hash(run.TRAIN / "model.pt") != trained["checkpoint_sha256"]):
        raise ValueError("complete fixed matched forecaster and reload evidence required")
    state = torch.load(run.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    if state["producer_fingerprint"] != trained["producer_fingerprint"]:
        raise ValueError("checkpoint producing identity differs")
    trace = [json.loads(line) for line in (run.TRAIN / "trace.jsonl").open()]
    replay = replay_training(rows, trace, state)
    snapshots = {f"step_{step:06d}.pt" for step in range(
        run.Config().snapshot_interval, run.Config().updates, run.Config().snapshot_interval)}
    if {path.name for path in (run.TRAIN / "checkpoints").glob("*.pt")} != snapshots:
        raise ValueError("all fixed recovery snapshots must be preserved")
    initial = torch.load(study.bootstrap.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)["model"]
    names = state["frozen_parameter_names"]
    if not names or len(names) != len(set(names)):
        raise ValueError("complete unique frozen parameter names required")
    frozen = {name: state["model"][name] for name in names}
    expected = study.q.read(study.bootstrap.TRAIN / "result.json")["frozen_parameter_sha256_unchanged"]
    if (any(not torch.equal(value, initial[name]) for name, value in frozen.items())
            or study.base.tensor_digest(frozen) != expected
            or trained["frozen_parameter_sha256_unchanged"] != expected):
        raise ValueError("word modules or finest head changed from the registered initializer")
    probes = study.q.read(run.TRAIN / "source_only_reload_probe.json")
    if [row["id"] for row in probes] != [row["id"] for row in run.clean.training_probes(rows)]:
        raise ValueError("complete native reload probe identities required")
    for probe in probes:
        if (len(probe["coarse_codes"]) != 8
                or any(len(codes) != 1 or len(codes[0]) != width for codes, width in zip(probe["coarse_codes"], run.WIDTHS))
                or len(probe["raw_word_ids"]) != 1 or len(probe["raw_word_ids"][0]) != 192):
            raise ValueError("complete native reload code and word frames required")
    fitted = replay_fit(selected)
    if fitted["checkpoint_sha256"] != trained["checkpoint_sha256"]:
        raise ValueError("fitting used another concept checkpoint")
    classification = ("matched_forecaster_fit_passed_word_adaptation_still_required"
        if fitted["forecaster_fit_gate"] else "matched_forecaster_fit_gate_failed")
    artifacts = [Path(__file__), Path(run.__file__), run.PLAN,
        run.WORKFLOW / "producer.json", *receipts,
        run.clean.RESULTS / "campaign_results.json",
        run.clean.RESULTS / (run.clean.HYPOTHESIS + "_registry_readback.json")]
    for root in (run.PILOT, run.FIT_PILOT, run.TRAIN, run.FIT):
        artifacts += [path for path in root.rglob("*")
            if path.is_file() and "producer_source" not in path.parts]
    campaign = {"schema": "nal.ncm-matched-forecaster.v1", "hypothesis_id": run.HYPOTHESIS,
        "classification": classification, "config": asdict(run.Config()), "training": trained,
        "training_cases": 690, "replay": replay, "fitting": fitted,
        "forecaster_fit_gate": fitted["forecaster_fit_gate"], "semantic_quality_measured": False,
        "task_accuracy": None, "predicted_history_word_adaptation_still_required": True,
        "NCM_success_established": False, "confirmed": False, "final_test_opened": False,
        "speed_tested": False, "artifacts": {str(path): file_hash(path) for path in dict.fromkeys(artifacts)}}
    run.RESULTS.mkdir()
    write_json(run.RESULTS / "campaign_results.json", campaign)
    lines = []
    for group in ("original_16", "additional_16", "all_32"):
        for values in fitted["components"][group]["levels"]:
            lines.append(f"| {group} | {values['level']} | {values['active_correct']}/{values['active_total']} | {values['active_correct']/values['active_total']:.2%} |")
    path = run.RESULTS / "campaign_results.json"
    run.REPORT.write_text(f"""# Matched-population NCM concept forecasting

Date: 2026-10-23. Component hypothesis {run.HYPOTHESIS}.
Classification **{classification}**. Campaign {path}.
SHA256 {file_hash(path)}.

| Fixed TRAIN probes | Coarse level | Correct active codes | Code agreement |
| --- | --- | --- | --- |
""" + "\n".join(lines) + f"""

The component gate requires at least99/100 active code agreement at each of
eight coarse levels in each16-case group and their32-case union. This is TRAIN
concept fitting. It establishes no complete-summary semantic accuracy.

The unchanged upstream clean-word component qualified before any model work.
The concept model then started from registered checkpoint
{run.clean.INITIAL_FORECASTER} with learned weights and a fresh optimizer.
It trained exactly49,152updates on the same690 immutable TRAIN summaries and
frozen full-source features. Half the training histories are uniformly
corrupted, with rates descending0.35to0 across the eight coarse levels, as in
the existing objective. This is a level-dependent schedule, not a time schedule.

Word modules, finest head, original codec and original Llama body/MLPs stayed
frozen. No word reconstruction loss or target word entered concept prediction.
The target word mask determines active-block loss weights only. FP32 masters,
optimizer/loss/reconstruction, BF16 forward, original dropout and disabledTF32
were preserved. Only the fixed final checkpoint was selected.

Maximum-frame/max-LR gradients and optimizer/sampler/noise/CPU/CUDA RNG/native
reload checks passed before full training. A fresh32-case pilot verified
finite, repeatable source-only rollouts against native generation. All sampled
sources, complete frames, actual and target histories, word masks, learning
rates, weighted losses, optimizer steps and sampler/noise states replayed.
Frozen parameter values, raw code counts, all recovery snapshots and the
fixed fitting gate were audited. Parent progress and future receipts were
excluded from aggregate hashing.

Predicted-history word adaptation and new semantic evaluation remain required.
The previous completed summary model scored0/64short and0/64long with128 valid
Terra judgments. Both sealed final source cohorts remain unopened. The original
95% complete-summary semantic target and full evaluation scope are unchanged;
no speed or compression benefit has been measured.
""")
    print(json.dumps({"status": "finalized", "classification": classification,
        "campaign_sha256": file_hash(path), "task_accuracy": None}), flush=True)


def register():
    run.data_inputs()
    study = run.clean.recovery.study
    campaign = study.q.read(run.RESULTS / "campaign_results.json")
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {
        run.HYPOTHESIS: {"name": "NCM concept forecasting on the matched690 TRAIN population",
            "description": "Fixed49152-update concept training on all690immutable TRAIN cases after registered clean-word fitting. Original body, codec and word modules frozen. Native source-only32TRAIN code fitting with sampler/noise/optimizer/raw-output replay. Word adaptation, semantic final quality and speed remain unproven.",
            "confirmation_status": campaign["classification"], "task_accuracy_measured": False,
            "TRAIN_fitting_measured": True, "forecaster_fit_gate": campaign["forecaster_fit_gate"],
            "NCM_success_established": False, "predicted_history_word_adaptation_still_required": True}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["finalize", "register"])
    globals()[parser.parse_args().stage]()
