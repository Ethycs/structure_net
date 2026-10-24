"""Replay dense source-word fitting without converting TRAIN fit into accuracy."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_dense_word_fit as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

TRAINABLE_PREFIXES = ("source_head.source_norm.", "source_head.source_projection.",
    "source_head.generator.transformer.", "word_projection.")


def replay_trace(rows, histories, labels, trace, state, *, pilot=False):
    config = run.Config()
    study = run.clean.recovery.study
    updates = 2 if pilot else config.updates
    if state["config"] != asdict(config) or state["step"] != updates or len(trace) != updates:
        raise ValueError("fixed complete dense training budget required")
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    probes = run.clean.training_probes(rows)
    expected = {r["id"]: {
        "actual_history_sha256": study.base.tensor_digest({"codes": torch.tensor([histories[r["id"]]["coarse_codes"]])}),
        "target_word_sha256": study.base.tensor_digest({"words": torch.tensor([r["word_ids"]])}),
        "teacher_latent_sha256": study.base.tensor_digest({"latent": labels[r["id"]]["refined"]})} for r in rows}
    counts = Counter()
    for step, item in enumerate(trace, 1):
        row = probes[step - 1] if pilot else rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        frame = config.max_source_frame if pilot and step == 2 else row["source_frame"]
        if (item["step"] != step or item["id"] != row["id"] or item["source_frame"] != frame
                or item["lr"] != (config.max_lr if pilot else run.rate(step))
                or any(item[k] != v for k, v in expected[row["id"]].items())
                or any(not math.isfinite(item[k]) or item[k] < 0 for k in ("latent_MSE", "word_CE", "gradient_norm"))):
            raise ValueError("actual source/history/word/teacher/frame/rate or finite objective differs")
        counts[row["id"]] += 1
    if not torch.equal(sampler.get_state(), state["sampler_rng"]) or (not pilot and set(counts) != {r["id"] for r in rows}):
        raise ValueError("complete selected-population coverage or sampler state differs")
    names, trainable = state["parameter_names"], state["trainable_parameter_names"]
    if (len(names) != len(set(names)) or not set(names) <= set(state["model"])
            or trainable != [name for name in names if name.startswith(TRAINABLE_PREFIXES)]):
        raise ValueError("explicit dense source/transformer/projection parameter scope differs")
    opt = state["optimizer"]
    if len(opt["param_groups"]) != 1 or set(opt["state"]) != set(range(len(trainable))):
        raise ValueError("complete selected optimizer parameter set required")
    group = opt["param_groups"][0]
    if (group["params"] != list(range(len(trainable))) or group["betas"] != (config.beta1, config.beta2)
            or group["eps"] != config.optimizer_eps or group["weight_decay"] != config.weight_decay
            or group["lr"] != (config.max_lr if pilot else run.rate(config.updates))):
        raise ValueError("fixed dense AdamW configuration differs")
    for i, name in enumerate(trainable):
        values = opt["state"][i]
        if int(values["step"]) != updates:
            raise ValueError("every selected optimizer step must be complete")
        for key in ("exp_avg", "exp_avg_sq"):
            if values[key].shape != state["model"][name].shape or not torch.isfinite(values[key]).all():
                raise ValueError("complete finite optimizer moments required")
    return {"updates": updates, "sample_counts": dict(sorted(counts.items())),
        "actual_targets_histories_and_schedule_replayed": True, "final_sampler_optimizer_exact": True}


def replay_fit(rows, histories, values, tokenizer):
    if [r["id"] for r in values] != [r["id"] for r in rows] or len(values) != 32:
        raise ValueError("same complete32 dense fitting identities required")
    for row, item in zip(rows, values):
        if (any(item[k] != row[k] for k in ("id", "source_sha256", "diagnostic_group", "target_text"))
                or item["target_word_ids"] != row["word_ids"]
                or item["actual_coarse_codes"] != histories[row["id"]]["coarse_codes"]):
            raise ValueError("raw complete source, words or actual history differs")
        raw = item["raw_word_ids"]
        if len(raw) != 192 or any(type(word) is not int or not 0 <= word < 50257 for word in raw):
            raise ValueError("complete bounded192-word output required")
        counts = run.clean.diagnostic.counts(torch.tensor([raw]), torch.tensor([row["word_ids"]]))
        text, end = run.clean.recovery.evaluation.reference.decode_words(raw, tokenizer)
        if (any(item[k] != counts[k] for k in ("correct", "total")) or item["text"] != text
                or item["global_EOS_position"] != (end if end < 192 else None)
                or item["exact_target_text"] != (text == row["target_text"]) or len(item["latent_sha256"]) != 64):
            raise ValueError("raw full decoding, count or latent binding differs")
    return run.aggregate(values)


def finalize():
    rows, manifest, histories, labels = run.inputs()
    study = run.clean.recovery.study
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh dense aggregate/report required")
    initial = torch.load(run.clean.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)["model"]
    codec = torch.load(run.teacher.prior.continuous.released.INITIAL / "tokenizer.pt",
        map_location="cpu", weights_only=True, mmap=True)["model"]
    initial = {name: codec["decoder." + name[len("word_decoder."):]] if name.startswith("word_decoder.") else value
        for name, value in initial.items()}
    artifacts = [Path(run.__file__), Path(__file__), run.PLAN, run.WORKFLOW / "producer.json"]
    replays = {}
    for root, pilot in ((run.PILOT, True), (run.TRAIN, False)):
        study.data.terminal(root)
        trained = study.q.read(root / "result.json")
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        trace = [json.loads(line) for line in (root / "trace.jsonl").open()]
        replay = replay_trace(rows, histories, labels, trace, state, pilot=pilot)
        names = state["parameter_names"]
        buffers = {"source_head.generator." + name for name in
            ("lvl_1L", "attn_mask_for_masking", "wp_prefix", "wp_1L")}
        if set(names) != set(initial) - buffers or set(state["model"]) != set(initial):
            raise ValueError("complete original parameter namespace required")
        frozen = {n: state["model"][n] for n in names if n not in state["trainable_parameter_names"]}
        expected_initial = study.base.tensor_digest({n: initial[n] for n in names})
        if (expected_initial != trained["initial_parameter_sha256"]
                or study.base.tensor_digest(frozen) != trained["frozen_parameter_sha256"]
                or any(not torch.equal(value, initial[n]) for n, value in frozen.items())
                or state["producer_fingerprint"] != trained["producer_fingerprint"]
                or trained["checkpoint_sha256"] != file_hash(root / "model.pt")
                or trained["updates"] != (2 if pilot else run.Config().updates)
                or trained["training_population"] != 32 or not trained["reload_exact"]
                or not trained["source_only_composed_reload_exact"] or not trained["original_decoder_frozen"]
                or not trained["dropout_map"] or any(trained["dropout_map"].values())
                or trained["forecaster_parameter_sha256"] != study.q.read(study.CACHE / "result.json")["forecaster_parameter_sha256"]
                or trained["codec_hash"] != run.teacher.prior.continuous.EXPECTED_CODEC
                or trained["peak_allocated_bytes"] > 8_000_000_000):
            raise ValueError("exact dense initializer, frozen parameters, reload or resource proof differs")
        probes = study.q.read(root / "source_only_reload_probe.json")
        if ([r["id"] for r in probes] != [r["id"] for r in run.clean.training_probes(rows)]
                or any(r["coarse_codes"] != [histories[r["id"]]["coarse_codes"]] for r in probes)):
            raise ValueError("native source-only reload/cache probe differs")
        expected_snapshots = set() if pilot else {"step_001024.pt"}
        if {p.name for p in (root / "checkpoints").glob("*.pt")} != expected_snapshots:
            raise ValueError("fixed recovery snapshot set differs")
        replays[str(root)] = {"training": trained, "replay": replay}
        artifacts += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    if replays[str(run.PILOT)]["training"]["initial_parameter_sha256"] != replays[str(run.TRAIN)]["training"]["initial_parameter_sha256"]:
        raise ValueError("pilot updates must not initialize full training")
    study.data.terminal(run.FIT)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.data.GPT2, local_files_only=True)
    measured = study.q.read(run.FIT / "result.json")
    values = study.q.read(run.FIT / "case_measurements.json")
    components = replay_fit(rows, histories, values, tokenizer)
    gate = run.fitting_gate(components)
    if (measured["components"] != components or measured["actual_history_fit_gate"] != gate
            or measured["checkpoint_sha256"] != replays[str(run.TRAIN)]["training"]["checkpoint_sha256"]
            or measured["measurements_sha256"] != file_hash(run.FIT / "case_measurements.json")
            or not measured["parameters_unchanged"] or not measured["fresh_native_cache_parity"]):
        raise ValueError("raw dense fitting aggregate differs")
    receipts = [run.WORKFLOW / "stage_receipts" / (stage + ".json") for stage in ("pilot", "train", "fit")]
    if any(study.q.read(p)["exit_code"] != 0 for p in receipts):
        raise ValueError("three actual numerical child zero exits required")
    artifacts += [*receipts, *[p for p in run.FIT.rglob("*") if p.is_file() and "producer_source" not in p.parts]]
    classification = "dense_source_TRAIN_fit_passed" if gate else "dense_source_TRAIN_fit_failed"
    campaign = {"schema": "nal.ncm-dense-source-word-fit.v1", "hypothesis_id": run.HYPOTHESIS,
        "classification": classification, "config": asdict(run.Config()), "training": replays,
        "measurement": measured, "components": components, "actual_history_fit_gate": gate,
        "training_cases": 32, "source_only_prediction": True, "teacher_targets_used_for_loss_only": True,
        "task_accuracy": None, "heldout_accuracy_measured": False, "NCM_success_established": False,
        "confirmed": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(artifacts)}}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    lines = [f"| {name} | {r['correct']}/{r['total']} = {r['token_agreement']:.2%} | {r['exact_target_texts']}/{r['cases']} |"
        for name, r in components.items()]
    run.REPORT.write_text(f"""# Source only NCM fitting with dense latent targets

Date: 2026-10-23. Hypothesis `{run.HYPOTHESIS}`.
Classification **{classification}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`.

| TRAIN group | Actual history token agreement | Exact target texts |
| --- | --- | --- |
""" + "\n".join(lines) + """

The student trained on these same32 TRAIN cases for2048 fixed updates. Predictions
receive complete frozen-Llama source features and source-only concept histories
from the unchanged forecaster. Correct target concepts, target words and refined
teacher latents never enter prediction inputs. Labels supervise full-frame
latent MSE plus0.1 nonpadding word CE. The original decoder was restored and
frozen, source/transformer/projection dropout was zero, and only the existing
source norm/projection, NCM word transformer and word projection trained.

The initializer retains learned clean-word weights outside the restored decoder.
The two-update maximum-LR/maximum-frame CUDA pilot was not retained in full
training. FP32 masters, optimizer, decoder and loss combine with BF16 source and
transformer computation; both TF32 modes are off. Body/MLPs, codec, forecaster,
original decoder and unused coarse heads stay frozen. Native cache parity,
exact save/reload, all2048 sample/target/teacher/rate/optimizer records, original
frozen parameter bytes and raw192-word outputs replayed.

This is a one-seed training-fit diagnostic. The unchanged99.5% token gate applies
separately to each16-case group and their32-case union, using actual predicted
histories. Passing cannot establish unseen summary accuracy or generalization.
Any full690-case latent construction/training needs a separate prospective plan.
The95% complete-summary goal remains unmet; both final cohorts remain unopened.
No speed or compression improvement is established by these measurements.
""")


def register():
    campaign = run.clean.recovery.study.q.read(run.RESULTS / "campaign_results.json")
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Source only NCM word fitting with refined dense latent labels",
        "description": "32TRAIN fitting control, source-only actual concepts, frozen original decoder; no unseen accuracy or speed claim.",
        "confirmation_status": campaign["classification"], "task_accuracy_measured": False,
        "TRAIN_fitting_measured": True, "actual_history_fit_gate": campaign["actual_history_fit_gate"],
        "NCM_success_established": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.clean.dependencies()
    run.clean.recovery.configure()
    finalize() if args.stage == "finalize" else register()
