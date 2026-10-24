"""Replay and conservatively register the paired whole-TRAIN dropout diagnostic."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_clean_word_dropout as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def classify(gates):
    if set(gates) != set(run.ARMS) or any(type(v) is not bool for v in gates.values()):
        raise ValueError("both valid arm fitting decisions required")
    original, zero = (gates[name] for name in run.ARMS)
    if original and zero:
        return "both_continuations_fit_passed"
    if zero:
        return "zero_dropout_fit_passed_control_failed"
    if original:
        return "original_dropout_only_fit_passed"
    return "both_dropout_fit_gates_failed"


def replay_trace(rows, trace, state, arm, initial_sha):
    config = run.Config()
    study = run.clean.recovery.study
    if (len(trace) != config.updates or state["step"] != config.updates
            or state["arm"] != arm or state["config"] != asdict(config)
            or state["initial_word_checkpoint_sha256"] != initial_sha
            or state["forecaster_checkpoint_sha256"] != run.clean.INITIAL_FORECASTER):
        raise ValueError("exact fixed final checkpoint required")
    maps = state["dropout_maps"]
    if (not maps["original"] or set(maps["original"].values()) != {.1, .3}
            or maps["applied"] != {key: value if arm == run.ARMS[0] else 0.
                                  for key, value in maps["original"].items()}):
        raise ValueError("complete original and applied dropout maps differ")
    expected = {row["id"]: (
        study.base.tensor_digest({"codes": torch.tensor([row["concept_ids"][:608]], dtype=torch.long)}),
        study.base.tensor_digest({"words": torch.tensor([row["word_ids"]], dtype=torch.long)}))
        for row in rows}
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    counts = Counter()
    for step, item in enumerate(trace, 1):
        row = rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        codes, words = expected[row["id"]]
        if (item["step"] != step or item["id"] != row["id"]
                or item["source_frame"] != row["source_frame"]
                or item["correct_history_sha256"] != codes or item["target_word_sha256"] != words
                or item["lr"] != run.rate(step, config)
                or any(not math.isfinite(item[key]) or item[key] < 0
                       for key in ("clean_word_CE", "gradient_norm"))):
            raise ValueError("actual source, frame, correct history, word target, rate or finite loss differs")
        counts[row["id"]] += 1
    optimizer = state["optimizer"]
    if (set(counts) != {row["id"] for row in rows}
            or not torch.equal(sampler.get_state(), state["sampler_rng"])
            or not optimizer["state"]
            or any(int(value["step"]) != config.updates for value in optimizer["state"].values())):
        raise ValueError("whole-population coverage or final sampler/optimizer steps differ")
    for group in optimizer["param_groups"]:
        if (group["betas"] != (config.beta1, config.beta2) or group["eps"] != config.optimizer_eps
                or group["weight_decay"] != config.weight_decay or group["lr"] != config.max_lr):
            raise ValueError("fixed AdamW configuration differs")
    return {"updates": len(trace), "sample_counts": dict(sorted(counts.items())),
        "every_TRAIN_case_sampled": True, "source_frame_history_target_rate_replayed": True,
        "final_optimizer_and_sampler_exact": True}


def replay_fit(arm, selected, histories, tokenizer):
    study = run.clean.recovery.study
    destination = run.root(arm, "fit")
    result = study.q.read(destination / "result.json")
    measured = study.q.read(destination / "case_measurements.json")
    if (result["arm"] != arm or len(measured) != 32
            or [item["id"] for item in measured] != [row["id"] for row in selected]):
        raise ValueError("same fixed complete32 TRAIN probe identities required")
    for item, row in zip(measured, selected):
        if (item["target_word_ids"] != row["word_ids"] or item["target_text"] != row["target_text"]
                or item["source_sha256"] != row["source_sha256"]
                or item["diagnostic_group"] != row["diagnostic_group"]
                or item["actual_coarse_codes"] != histories[row["id"]]["coarse_codes"]):
            raise ValueError("target, source, probe group or frozen actual history differs")
        target = torch.tensor([row["word_ids"]], dtype=torch.long)
        for name in run.clean.diagnostic.ARMS:
            values = item["arms"][name]
            raw = values["raw_word_ids"]
            if len(raw) != 192 or any(type(word) is not int or not 0 <= word < 50257 for word in raw):
                raise ValueError("complete192 valid raw word IDs required")
            counted = run.clean.diagnostic.counts(torch.tensor([raw]), target)
            text, end = run.clean.recovery.evaluation.reference.decode_words(raw, tokenizer)
            if (any(values[key] != counted[key] for key in ("correct", "total"))
                    or values["text"] != text or values["global_EOS_position"] != (end if end < 192 else None)
                    or values["exact_target_text"] != (text == row["target_text"])
                    or not math.isfinite(values["CE"]) or values["CE"] < 0):
                raise ValueError("raw counts, complete decoding or finite loss differs")
        own = torch.tensor([item["arms"]["actual_history"]["raw_word_ids"]])
        gold = torch.tensor([item["arms"]["correct_history"]["raw_word_ids"]]).masked_fill(target.eq(50257), 50257)
        if item["own_gold_agreement"] != run.clean.diagnostic.counts(own, gold):
            raise ValueError("actual/correct history word agreement differs")
        active = target.reshape(1, 4, 48).ne(50257).any(-1)
        position = 0
        for level, width in enumerate(run.forecaster.WIDTHS):
            mask = active.repeat_interleave(width // 4, dim=1)
            actual = torch.tensor([item["actual_coarse_codes"][position:position + width]])
            expected = torch.tensor([row["concept_ids"][position:position + width]])
            if item["concepts"][level] != {"level": level,
                    "active_correct": int(((actual == expected) & mask).sum()), "active_total": int(mask.sum())}:
                raise ValueError("raw active concept counts differ")
            position += width
    components = run.clean.diagnostic.aggregate(measured)
    if (result["components"] != components or result["clean_fit_gate"] != run.clean.fitting_gate(components)
            or result["measurements_sha256"] != file_hash(destination / "case_measurements.json")):
        raise ValueError("complete fitting aggregation or inclusive gate differs")
    return result


def replay_pilot(rows, arm, initial_sha):
    study = run.clean.recovery.study
    destination = run.root(arm, "pilot")
    pilot = study.q.read(destination / "result.json")
    state = torch.load(destination / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    run.validate_checkpoint(state, arm, pilot["dropout_maps"], 2, initial_sha)
    trace = [json.loads(line) for line in (destination / "trace.jsonl").open()]
    selected = run.clean.training_probes(rows)
    if len(trace) != 2 or state["producer_fingerprint"] != pilot["producer_fingerprint"]:
        raise ValueError("complete two-update pilot and checkpoint producing identity required")
    for step, (row, item) in enumerate(zip(selected, trace), 1):
        expected_frame = run.Config().max_source_frame if step == 2 else row["source_frame"]
        if (item["step"] != step or item["id"] != row["id"] or item["source_frame"] != expected_frame
                or item["lr"] != run.Config().max_lr
                or item["correct_history_sha256"] != study.base.tensor_digest({"codes":
                    torch.tensor([row["concept_ids"][:608]], dtype=torch.long)})
                or item["target_word_sha256"] != study.base.tensor_digest({"words":
                    torch.tensor([row["word_ids"]], dtype=torch.long)})
                or any(not math.isfinite(item[key]) or item[key] <= 0
                       for key in ("clean_word_CE", "gradient_norm"))):
            raise ValueError("actual maximum-frame/max-LR pilot trace differs")
    if (pilot["checkpoint_sha256"] != file_hash(destination / "model.pt")
            or not torch.equal(state["sampler_rng"], torch.Generator().manual_seed(run.Config().sampler_seed).get_state())
            or not state["optimizer"]["state"]
            or any(int(value["step"]) != 2 for value in state["optimizer"]["state"].values())):
        raise ValueError("pilot checkpoint, sampler or two optimizer steps differ")
    return pilot


def finalize():
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh archive and report required")
    rows, manifest, histories, selected = run.data_inputs()
    study = run.clean.recovery.study
    initial = run.initial_result()
    initial_state = torch.load(run.clean.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)["model"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.data.GPT2, local_files_only=True)
    results, artifacts, sequences, initial_hashes, original_maps = {}, [], [], [], []
    receipts = [run.WORKFLOW / "stage_receipts" / (name + ".json") for name in
        ("original_dropout_pilot", "zero_dropout_pilot", "original_dropout_train", "original_dropout_fit",
         "zero_dropout_train", "zero_dropout_fit")]
    if any(study.q.read(path)["exit_code"] != 0 for path in receipts):
        raise ValueError("six actual successful numerical child exits required")
    for arm in run.ARMS:
        for stage in ("pilot", "train", "fit"):
            destination = run.root(arm, stage)
            study.data.terminal(destination)
            artifacts += [path for path in destination.rglob("*")
                          if path.is_file() and "producer_source" not in path.parts]
        destination = run.root(arm, "train")
        trained = study.q.read(destination / "result.json")
        state = torch.load(destination / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        trace = [json.loads(line) for line in (destination / "trace.jsonl").open()]
        if (trained["status"] != "matched_dropout_word_trained" or trained["arm"] != arm
                or trained["updates"] != run.Config().updates or trained["training_population"] != 690
                or not trained["reload_exact"] or not trained["source_only_composed_reload_exact"]
                or trained["peak_allocated_bytes"] > 8_000_000_000
                or state["producer_fingerprint"] != trained["producer_fingerprint"]
                or trained["checkpoint_sha256"] != file_hash(destination / "model.pt")
                or trained["dropout_maps"] != state["dropout_maps"]):
            raise ValueError("fixed actual training, reload, dropout and identity evidence required")
        replay = replay_trace(rows, trace, state, arm, initial["checkpoint_sha256"])
        names = state["parameter_names"]
        if not names or len(names) != len(set(names)) or not set(names) <= set(initial_state):
            raise ValueError("complete unique model parameter names required")
        initial_hash = study.base.tensor_digest({name: initial_state[name] for name in names})
        if initial_hash != trained["initial_parameter_sha256"]:
            raise ValueError("raw registered initializer parameter hash differs")
        frozen = {name: value for name, value in state["model"].items()
            if name == "source_head.generator.l0_queries" or name.startswith("source_head.generator.lm_heads.")}
        if (not frozen or study.base.tensor_digest(frozen) != run.clean.FROZEN_WORD
                or any(not torch.equal(value, initial_state[name]) for name, value in frozen.items())):
            raise ValueError("unused branch roots or heads changed")
        expected = {f"step_{step:06d}.pt" for step in
                    range(run.Config().snapshot_interval, run.Config().updates, run.Config().snapshot_interval)}
        if {path.name for path in (destination / "checkpoints").glob("*.pt")} != expected:
            raise ValueError("declared recovery snapshots must remain present")
        pilot = replay_pilot(rows, arm, initial["checkpoint_sha256"])
        probes = study.q.read(destination / "source_only_reload_probe.json")
        if (pilot["status"] != "systems_dropout_gradient_reload_passed" or pilot["updates"] != 2
                or pilot["initial_parameter_sha256"] != initial_hash or pilot["dropout_maps"] != state["dropout_maps"]
                or not pilot["reload_exact"] or not pilot["source_only_composed_reload_exact"]
                or pilot["peak_allocated_bytes"] > 8_000_000_000
                or [row["id"] for row in probes] != [row["id"] for row in run.clean.training_probes(rows)]
                or any(row["coarse_codes"] != [histories[row["id"]]["coarse_codes"]] for row in probes)):
            raise ValueError("actual maximum-frame pilot, initializer and native reload cache parity required")
        fitted = replay_fit(arm, selected, histories, tokenizer)
        if (fitted["checkpoint_sha256"] != trained["checkpoint_sha256"]
                or fitted["dropout_maps"] != state["dropout_maps"]
                or fitted["peak_allocated_bytes"] > 8_000_000_000
                or not fitted["parameters_and_codec_unchanged"] or not fitted["fresh_histories_match_training_cache"]):
            raise ValueError("fresh read-only fitting evidence differs")
        results[arm] = {"training": trained, "replay": replay, "fitting": fitted}
        sequences.append([(item["id"], item["source_frame"], item["correct_history_sha256"],
                           item["target_word_sha256"], item["lr"]) for item in trace])
        initial_hashes.append(initial_hash)
        original_maps.append(state["dropout_maps"]["original"])
    if sequences[0] != sequences[1] or initial_hashes[0] != initial_hashes[1] or original_maps[0] != original_maps[1]:
        raise ValueError("paired complete training identities, initialization and original dropout differ")
    gates = {arm: results[arm]["fitting"]["clean_fit_gate"] for arm in run.ARMS}
    classification = classify(gates)
    upstream = study.q.read(run.clean.FIT / "result.json")
    artifacts += [run.PLAN, run.AUDITOR_FILE, Path(run.__file__), run.WORKFLOW / "producer.json", *receipts,
        run.clean.RESULTS / "campaign_results.json", run.clean.RESULTS / (run.clean.HYPOTHESIS + "_registry_readback.json"),
        run.clean.FIT / "result.json", run.clean.FIT / "case_measurements.json"]
    campaign = {"schema": "nal.ncm-clean-word-dropout.v1", "hypothesis_id": run.HYPOTHESIS,
        "classification": classification, "config": asdict(run.Config()), "arms": results,
        "upstream_fitting": upstream, "paired_sampling_and_initialization_exact": True,
        "fit_gates": gates, "training_cases": 690, "task_accuracy": None,
        "semantic_quality_measured": False, "NCM_success_established": False, "confirmed": False,
        "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(path): file_hash(path) for path in dict.fromkeys(artifacts)}}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    lines = []
    for group in ("original_16", "additional_16", "all_32"):
        values = [upstream["components"][group]["arms"]["correct_history"]]
        values += [results[arm]["fitting"]["components"][group]["arms"]["correct_history"] for arm in run.ARMS]
        cells = [f"{v['correct']}/{v['total']} = {v['token_agreement']:.2%}" for v in values]
        lines.append("| " + group + " | " + " | ".join(cells) + " |")
    run.REPORT.write_text(f"""# Matched whole-TRAIN training-dropout diagnostic

Date:2026-10-23. Hypothesis {run.HYPOTHESIS}.
Classification **{classification}**. Campaign {path}; SHA256 {file_hash(path)}.

| Fixed TRAIN probes | Upstream final | Continued original dropout | Continued zero dropout |
| --- | --- | --- | --- |
""" + "\n".join(lines) + f"""

These are correct-history word-token fitting measurements including EOS and
excluding padding, not semantic summary accuracy. Both arms used the same690
immutable TRAIN examples, the exact registered final initializer
{initial['checkpoint_sha256']}, identical sampled IDs/frames/history/word-target
hashes and rates,8192updates each, and fresh matched AdamW optimizers. Only
training dropout differed: original .3 transformer/.1 decoder versus zero,
covering both nn.Dropout and numeric functional attention dropout. Both were
evaluated in eval mode. One model/dropout seed and one independent sampler seed
were used; changed dropout RNG consumption is part of the intervention.

The unchanged199/200 fitting gate applies separately to original16, added16
and all32 probes. If both continuations pass, extra training cannot be confused
with an isolated benefit from removing dropout. No arm is automatically promoted
to the old queued forecaster, and none establishes unseen generation quality.

Both CUDA pilots, frozen-state/native-cache/reload controls, full paired sampler
and optimizer audits, exact raw fitting replay and conservative registry checks
are retained. Original Llama body/MLPs, codec, old252-case concept forecaster and
unused heads stayed frozen. Predictions received no finest labels or words.

The last completed semantic candidate remains0/64short+0/64long,128valid Terra
judgments. The full95% complete-summary source-disjoint semantic goal remains
unmet. Both sealed final source cohorts stayed unopened. No speed/compression
benefit is measured by this diagnostic.
""")
    print(json.dumps({"status": "finalized", "classification": classification,
        "campaign_sha256": file_hash(path), "task_accuracy": None}), flush=True)


def register():
    campaign = run.clean.recovery.study.q.read(run.RESULTS / "campaign_results.json")
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {
        run.HYPOTHESIS: {"name": "Matched original versus zero training dropout for whole-TRAIN NCM words",
            "description": "Paired8192-update continuations on the same690TRAIN cases diagnose word fitting; no semantic accuracy or speed claim.",
            "confirmation_status": campaign["classification"], "task_accuracy_measured": False,
            "TRAIN_fitting_measured": True, "NCM_success_established": False,
            "original_dropout_fit_gate": campaign["fit_gates"][run.ARMS[0]],
            "zero_dropout_fit_gate": campaign["fit_gates"][run.ARMS[1]]}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.dependencies()
    run.clean.recovery.configure()
    finalize() if args.stage == "finalize" else register()
