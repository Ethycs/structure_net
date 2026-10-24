"""Audit and register the complete clean-history TRAIN fitting component."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_matched_clean_word as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def replay_training(rows, trace, state):
    study = run.recovery.study
    config = run.Config()
    if (len(trace) != config.updates or state["step"] != config.updates
            or state["config"] != asdict(config)
            or state["initial_word_checkpoint_sha256"] != run.INITIAL_WORD
            or state["forecaster_checkpoint_sha256"] != run.INITIAL_FORECASTER):
        raise ValueError("only the fixed complete clean-word model can be audited")
    hashes = {row["id"]: study.base.tensor_digest({"codes":
        torch.tensor([row["concept_ids"][:608]], dtype=torch.long)}) for row in rows}
    word_hashes = {row["id"]: study.base.tensor_digest({"words":
        torch.tensor([row["word_ids"]], dtype=torch.long)}) for row in rows}
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    counts = Counter()
    for step, item in enumerate(trace, 1):
        row = rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        if (item["step"] != step or item["id"] != row["id"]
                or item["source_frame"] != row["source_frame"]
                or item["correct_history_sha256"] != hashes[row["id"]]
                or item["target_word_sha256"] != word_hashes[row["id"]]
                or item["lr"] != run.rate(step, config)):
            raise ValueError("clean training source, full frame, code history or rate differs")
        if any(not math.isfinite(item[key]) or item[key] < 0
               for key in ("clean_word_CE", "gradient_norm")):
            raise ValueError("finite clean loss and gradient norms required")
        counts[row["id"]] += 1
    if (set(counts) != {row["id"] for row in rows}
            or not torch.equal(sampler.get_state(), state["sampler_rng"])
            or not state["optimizer"]["state"]
            or any(int(value["step"]) != config.updates for value in state["optimizer"]["state"].values())):
        raise ValueError("complete population coverage, sampler and optimizer steps required")
    return {"updates": len(trace), "sample_counts": dict(sorted(counts.items())),
        "every_TRAIN_case_sampled": True, "all_sources_frames_correct_histories_rates_replayed": True,
        "final_sampler_and_optimizer_steps_exact": True}


def replay_fit(selected, histories, manifest, tokenizer):
    study = run.recovery.study
    result = study.q.read(run.FIT / "result.json")
    measurements = study.q.read(run.FIT / "case_measurements.json")
    if len(measurements) != 32 or [row["id"] for row in measurements] != [row["id"] for row in selected]:
        raise ValueError("complete fixed32 TRAIN probe identities required")
    for item, row in zip(measurements, selected):
        if (item["target_word_ids"] != row["word_ids"] or item["target_text"] != row["target_text"]
                or item["source_sha256"] != row["source_sha256"]
                or item["diagnostic_group"] != row["diagnostic_group"]
                or item["actual_coarse_codes"] != histories[row["id"]]["coarse_codes"]):
            raise ValueError("complete target, source, group or actual history differs")
        target = torch.tensor([row["word_ids"]])
        for arm in run.diagnostic.ARMS:
            values = item["arms"][arm]
            raw = values["raw_word_ids"]
            if len(raw) != 192 or any(not 0 <= word < 50257 for word in raw):
                raise ValueError("complete192 raw word IDs required")
            text, end = run.recovery.evaluation.reference.decode_words(raw, tokenizer)
            counted = run.diagnostic.counts(torch.tensor([raw]), target)
            if (any(values[key] != counted[key] for key in ("correct", "total"))
                    or values["text"] != text or values["global_EOS_position"] != (end if end < 192 else None)
                    or values["exact_target_text"] != (text == row["target_text"])
                    or not math.isfinite(values["CE"]) or values["CE"] < 0):
                raise ValueError("raw word fitting counts or global EOS decoding differs")
        own = torch.tensor([item["arms"]["actual_history"]["raw_word_ids"]])
        gold = torch.tensor([item["arms"]["correct_history"]["raw_word_ids"]]).masked_fill(target.eq(50257), 50257)
        if item["own_gold_agreement"] != run.diagnostic.counts(own, gold):
            raise ValueError("own versus correct word agreement differs")
        position = 0
        active = target.reshape(1, 4, 48).ne(50257).any(-1)
        for level, width in enumerate((16, 24, 40, 56, 80, 104, 128, 160)):
            mask = active.repeat_interleave(width // 4, dim=1)
            predicted = torch.tensor([item["actual_coarse_codes"][position:position + width]])
            expected = torch.tensor([row["concept_ids"][position:position + width]])
            if item["concepts"][level] != {"level": level,
                    "active_correct": int(((predicted == expected) & mask).sum()),
                    "active_total": int(mask.sum())}:
                raise ValueError("raw active concept fitting counts differ")
            position += width
    actual = run.diagnostic.aggregate(measurements)
    if actual != result["components"] or result["clean_fit_gate"] != run.fitting_gate(actual):
        raise ValueError("fixed TRAIN component aggregation or inclusive fitting gate differs")
    if file_hash(run.FIT / "case_measurements.json") != result["measurements_sha256"]:
        raise ValueError("fit measurements changed")
    return result


def finalize():
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh component archive and report required")
    rows, manifest, histories, selected = run.require_inputs()
    study = run.recovery.study
    for root in (run.PILOT, run.FIT_PILOT, run.TRAIN, run.FIT):
        study.data.terminal(root)
    receipts = [run.WORKFLOW / "stage_receipts" / (stage + ".json")
        for stage in ("train_pilot", "fit_pilot", "train", "fit")]
    if any(study.q.read(path)["exit_code"] != 0 for path in receipts):
        raise ValueError("all numerical children must have actual zero exits")
    trained = study.q.read(run.TRAIN / "result.json")
    if (trained["updates"] != run.Config().updates or trained["training_pool_cases"] != 690
            or not trained["reload_exact"] or not trained["source_only_composed_reload_exact"]
            or trained["peak_allocated_bytes"] > 8_000_000_000
            or file_hash(run.TRAIN / "model.pt") != trained["checkpoint_sha256"]):
        raise ValueError("unchanged final clean-word model and actual reload evidence required")
    state = torch.load(run.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    if state["producer_fingerprint"] != trained["producer_fingerprint"]:
        raise ValueError("checkpoint producing fingerprint differs")
    trace = [json.loads(line) for line in (run.TRAIN / "trace.jsonl").open()]
    replay = replay_training(rows, trace, state)
    snapshots = {f"step_{step:06d}.pt" for step in range(4096, run.Config().updates, 4096)}
    if {path.name for path in (run.TRAIN / "checkpoints").glob("*.pt")} != snapshots:
        raise ValueError("all seven recovery snapshots must remain present")
    initial = torch.load(study.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)["model"]
    frozen = {name: value for name, value in state["model"].items()
        if name == "source_head.generator.l0_queries" or name.startswith("source_head.generator.lm_heads.")}
    if (not frozen or any(not torch.equal(value, initial[name]) for name, value in frozen.items())
            or study.base.tensor_digest(frozen) != run.FROZEN_WORD):
        raise ValueError("unused word branch parameters changed from the registered initializer")
    probes = study.q.read(run.TRAIN / "source_only_reload_probe.json")
    if ([row["id"] for row in probes] != [row["id"] for row in run.training_probes(rows)]
            or any(row["coarse_codes"] != [histories[row["id"]]["coarse_codes"]] for row in probes)):
        raise ValueError("native source-only reload probes differ from the original cache")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.data.GPT2, local_files_only=True)
    fitted = replay_fit(selected, histories, manifest, tokenizer)
    if fitted["checkpoint_sha256"] != trained["checkpoint_sha256"]:
        raise ValueError("TRAIN fitting evaluated a different checkpoint")
    classification = ("clean_word_fit_passed_matched_forecaster_still_required"
        if fitted["clean_fit_gate"] else "clean_word_fit_gate_failed")
    artifacts = [Path(__file__), Path(run.__file__), run.PLAN,
        run.WORKFLOW / "producer.json", *receipts]
    for root in (run.PILOT, run.FIT_PILOT, run.TRAIN, run.FIT):
        artifacts += [path for path in root.rglob("*")
            if path.is_file() and "producer_source" not in path.parts]
    campaign = {"schema": "nal.ncm-matched-clean-word.v1", "hypothesis_id": run.HYPOTHESIS,
        "classification": classification, "config": asdict(run.Config()), "training": trained,
        "training_cases": 690, "replay": replay, "fitting": fitted,
        "clean_fit_gate": fitted["clean_fit_gate"], "semantic_quality_measured": False,
        "task_accuracy": None, "forecaster_matched_population_training_still_required": True,
        "NCM_success_established": False, "confirmed": False, "final_test_opened": False,
        "speed_tested": False, "artifacts": {str(path): file_hash(path) for path in dict.fromkeys(artifacts)}}
    run.RESULTS.mkdir()
    write_json(run.RESULTS / "campaign_results.json", campaign)
    summary = fitted["components"]
    lines = []
    for group in ("original_16", "additional_16", "all_32"):
        own = summary[group]["arms"]["actual_history"]
        correct = summary[group]["arms"]["correct_history"]
        lines.append(f"| {group} | {correct['correct']}/{correct['total']} = {correct['token_agreement']:.2%} | {own['correct']}/{own['total']} = {own['token_agreement']:.2%} |")
    path = run.RESULTS / "campaign_results.json"
    run.REPORT.write_text(f"""# Correct-history word fitting on the complete TRAIN population

Date: 2026-10-23. Component hypothesis {run.HYPOTHESIS}.
Classification: **{classification}**. Campaign {path}.
SHA256: {file_hash(path)}.

| Fixed TRAIN probes | Correct-history token agreement | Actual-history token agreement |
| --- | --- | --- |
""" + "\n".join(lines) + f"""

The component gate requires at least 199/200 active-token agreement in each
sixteen-case probe group and in their complete union. This is TRAIN fitting,
including EOS and excluding padding. It is not semantic summary accuracy.
The preceding model's measured semantic development score remains 0/64 short
and 0/64 long, with 128 valid Terra judgments. This new word branch has not
received a semantic evaluation.

The word branch started from registered checkpoint {run.INITIAL_WORD}; its
parameters were learned previously, while AdamW and RNG streams started fresh.
It trained exactly32,768updates on all690 immutable TRAIN examples, with
correct coarse histories only and no finest target codes or target words in
prediction inputs. FP32 parameters and loss, BF16 forward computation, disabled
TF32, original dropout, complete native source frames and all192 output
positions were preserved. Only the fixed final checkpoint was selected.

Maximum-frame gradient/optimizer/RNG/composed-reload and fresh-model fit-path
pilots completed before full training. All sampled sources, complete frames,
correct-history hashes, learning rates, optimizer steps, sampler state, unused
word-branch parameters and raw fitting counts replayed. The original Llama
body/MLPs, original codec and 252-case forecaster stayed frozen.

A qualifying component result admits further matched-population forecaster
training and later predicted-history adaptation; neither has been established
by this component. Both sealed final source cohorts remain unopened. The goal
still requires at least95% complete-summary semantic quality on the original
source-disjoint final evaluation, after qualifying development results.
No speed or compression benefit is measured here.
""")
    print(json.dumps({"status": "finalized", "classification": classification,
        "campaign_sha256": file_hash(path), "task_accuracy": None}), flush=True)


def register():
    run.require_inputs()
    study = run.recovery.study
    campaign = study.q.read(run.RESULTS / "campaign_results.json")
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {
        run.HYPOTHESIS: {"name": "Correct-history word fitting before matched NCM forecasting",
            "description": "Fixed32768-update clean-code word fitting on all690 immutable TRAIN examples from a registered initializer; fixed16original+16additional TRAIN probes, raw sampler/optimizer/output replay. Semantic quality, matched forecaster training, final95percent target and speed remain unproven.",
            "confirmation_status": campaign["classification"], "task_accuracy_measured": False,
            "TRAIN_fitting_measured": True, "clean_fit_gate": campaign["clean_fit_gate"],
            "NCM_success_established": False,
            "forecaster_matched_population_training_still_required": True}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["finalize", "register"])
    globals()[parser.parse_args().stage]()
