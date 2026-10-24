"""Replay all690 dense word training without claiming semantic accuracy."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_full_population_dense_word as run
from experiments.neural_architecture_lab import finalize_ncm_dense_word_fit as reference
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def objective_receipts(trace, config):
    for item in trace:
        expected = config.latent_weight * item["latent_MSE"] + config.word_weight * item["word_CE"]
        if (not math.isfinite(item["objective_loss"]) or item["objective_loss"] < 0
                or not math.isclose(item["objective_loss"], expected, rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError("actual weighted dense/word objective differs")


def finalize():
    run.configure()
    reference.run = run
    rows, _, histories, labels = run.inputs()
    _, _, selected = run.forecaster.data_inputs()
    study = run.clean.recovery.study
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh full-population word aggregate/report required")
    initial_state = run.loader.checked_checkpoint(run.INITIAL_PATH, run.INITIAL_WORD, 2048)
    initial = initial_state["model"]
    artifacts = [Path(run.__file__), Path(__file__), run.PLAN, run.WORKFLOW / "producer.json", run.WORKFLOW / "admission.json"]
    replays = {}
    for root, pilot in ((run.PILOT, True), (run.TRAIN, False)):
        study.data.terminal(root)
        trained = run.q.read(root / "result.json")
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        trace = [json.loads(line) for line in (root / "trace.jsonl").open()]
        replay = reference.replay_trace(rows, histories, labels, trace, state, pilot=pilot)
        objective_receipts(trace, run.Config())
        names, trainable = state["parameter_names"], state["trainable_parameter_names"]
        if set(names) != set(initial_state["parameter_names"]) or set(state["model"]) != set(initial):
            raise ValueError("complete selected learned word initializer namespace required")
        frozen = {n: state["model"][n] for n in names if n not in trainable}
        if (study.base.tensor_digest({n: initial[n] for n in names}) != trained["initial_parameter_sha256"]
                or study.base.tensor_digest(frozen) != run.loader.DENSE_FROZEN_PARAMETERS
                or trained["frozen_parameter_sha256"] != run.loader.DENSE_FROZEN_PARAMETERS
                or any(not torch.equal(value, initial[n]) for n, value in frozen.items())
                or state["producer_fingerprint"] != trained["producer_fingerprint"]
                or trained["checkpoint_sha256"] != file_hash(root / "model.pt")
                or trained["updates"] != (2 if pilot else run.Config().updates)
                or trained["training_population"] != 690 or trained["cases"] != (2 if pilot else 690)
                or not trained["reload_exact"] or not trained["source_only_composed_reload_exact"]
                or not trained["original_decoder_frozen"] or not trained["dropout_map"] or any(trained["dropout_map"].values())
                or trained["forecaster_parameter_sha256"] != run.q.read(run.WORKFLOW / "admission.json")["forecaster_parameter_sha256"]
                or trained["codec_hash"] != run.loader.ORIGINAL_CODEC or trained["peak_allocated_bytes"] > 8_000_000_000
                or trained["task_accuracy"] is not None or trained["semantic_quality_measured"]
                or trained["final_test_opened"] or trained["speed_tested"]):
            raise ValueError("selected initializer, full population, frozen bytes or actual lifecycle controls differ")
        outputs = run.q.read(root / "source_only_reload_probe.json")
        if ([r["id"] for r in outputs] != [r["id"] for r in run.clean.training_probes(rows)]
                or any(r["coarse_codes"] != [histories[r["id"]]["coarse_codes"]] for r in outputs)):
            raise ValueError("composed native reload probes differ from actual matched cache")
        expected_snapshots = set() if pilot else {f"step_{step:06d}.pt" for step in
            range(run.Config().snapshot_interval, run.Config().updates, run.Config().snapshot_interval)}
        if {p.name for p in (root / "checkpoints").glob("*.pt")} != expected_snapshots:
            raise ValueError("all fixed recovery checkpoints must be retained")
        replays[str(root)] = {"training": trained, "replay": replay}
        artifacts += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    if replays[str(run.PILOT)]["training"]["initial_parameter_sha256"] != replays[str(run.TRAIN)]["training"]["initial_parameter_sha256"]:
        raise ValueError("full training must reload the untouched learned initializer")
    study.data.terminal(run.FIT)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.data.GPT2, local_files_only=True)
    measured = run.q.read(run.FIT / "result.json")
    values = run.q.read(run.FIT / "case_measurements.json")
    components = reference.replay_fit(selected, histories, values, tokenizer)
    gate = run.legacy.fitting_gate(components)
    if (measured["status"] != "full_population_dense_TRAIN_fitting_measured" or measured["training_population"] != 690
            or measured["cases"] != 32 or measured["components"] != components or measured["actual_history_fit_gate"] != gate
            or measured["checkpoint_sha256"] != replays[str(run.TRAIN)]["training"]["checkpoint_sha256"]
            or measured["measurements_sha256"] != file_hash(run.FIT / "case_measurements.json")
            or not measured["parameters_unchanged"] or not measured["fresh_native_cache_parity"]):
        raise ValueError("same32 raw fitting counts and fixed final candidate required")
    receipts = [run.WORKFLOW / "stage_receipts" / (s + ".json") for s in ("pilot", "train", "fit")]
    if any(run.q.read(p)["exit_code"] or run.q.read(p)["observer"] != "subprocess.wait" for p in receipts):
        raise ValueError("three actual successful numerical children required")
    artifacts += [*receipts, *[p for p in run.FIT.rglob("*") if p.is_file() and "producer_source" not in p.parts]]
    workflow = run.q.read(run.WORKFLOW / "producer.json")
    study.data.verify(workflow)
    artifact_hashes = {str(p): file_hash(p) for p in artifacts}
    artifact_hashes.update(workflow["input_hashes"])
    for root in (run.PILOT, run.TRAIN, run.FIT):
        artifact_hashes.update(run.q.read(root / "producer.json")["input_hashes"])
    classification = "full690_dense_word_TRAIN_fit_passed_semantics_unmeasured" if gate else "full690_dense_word_TRAIN_fit_failed_semantics_unmeasured"
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": classification, "config": asdict(run.Config()),
        "training_cases": 690, "TRAIN_probe_cases": 32, "components": components,
        "training": replays, "measurement": measured, "actual_history_fit_gate": gate,
        "source_only_prediction": True, "teacher_targets_used_for_loss_only": True,
        "initial_word_checkpoint_sha256": run.INITIAL_WORD, "upstream_admission": run.q.read(run.WORKFLOW / "admission.json"),
        "task_accuracy": None, "heldout_accuracy_measured": False, "NCM_success_established": False,
        "confirmed": False, "final_test_opened": False, "speed_tested": False, "artifacts": artifact_hashes}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    table = "\n".join(f"| {name} | {r['correct']}/{r['total']} | {r['exact_target_texts']}/{r['cases']} |" for name, r in components.items())
    run.REPORT.write_text(f"""# Dense NCM word training on all approved summaries

Date: 2026-10-24. Exploratory hypothesis `{run.HYPOTHESIS}`.
Classification **{classification}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`.

| Fixed TRAIN probes | Correct active words | Exact target texts |
| --- | --- | --- |
{table}

The word branch continued the exact learned2048-update dense initializer for
32768 fixed updates on all690 approved TRAIN cases. Source-only histories came
from the new final matched forecaster; target words and optimized latent labels
entered only latent MSE plus0.1 nonpadding word CE. The original Llama body and
MLPs, forecaster, codec, decoder and unused heads stayed frozen. FP32 masters,
optimizer, decoder and losses combined with BF16 source/transformer computation,
zero dropout and disabled TF32. Only the fixed final checkpoint was selected.

Maximum-frame/maximum-LR gradients, actual native cache parity and exact model,
optimizer, sampler and RNG reload preceded full training. Frozen bytes and the
8GB allocation limit held. CPU replay checked every selected source, actual
history, target word and teacher latent hash, frame, learning rate, weighted
objective, optimizer step and complete sampler coverage. All fixed recovery
checkpoints and raw192-word diagnostic predictions remain available.

The original32-case parent fitting and semantic failures remain unchanged.
This continuation changes both training population and forecaster; it does not
isolate either cause. The99.5% token gate is a TRAIN diagnostic, with no unseen
accuracy interpretation. Fresh complete-summary semantic evaluation on all128
development cases remains required for this fixed candidate. All sealed final
sources stay closed until eligible. The95% task goal remains unmet, and no speed
or compression gain is established by this study.
""")
    print(json.dumps({"classification": classification, "components": components, "campaign_sha256": file_hash(path)}), flush=True)


def register():
    campaign = run.q.read(run.RESULTS / "campaign_results.json")
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Full690 source only dense NCM word continuation",
        "description": "All690 approved TRAIN cases and qualified dense labels, matched fixed source-only forecaster histories, frozen original body/decoder; fresh semantic evaluation still required.",
        "classification": campaign["classification"], "actual_history_fit_gate": campaign["actual_history_fit_gate"],
        "task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.configure()
    finalize() if args.stage == "finalize" else register()
