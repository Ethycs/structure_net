"""Finalize the matched EMA-unit control without treating reconstruction as task accuracy."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_sample_scaled_codec_ema as trial
from experiments.structure_net import ncm_broad_codec_reconstruction as measurement
from experiments.neural_architecture_lab.finalize_ncm_broad_learned_codec import replay
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json

DEST = trial.core.ROOT / "20261023_sample_scaled_codec_ema_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-sample-scaled-codec-ema.md")
HYPOTHESIS = "ncm-sample-scaled-codec-ema-v1"


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("append-only finalization requires a fresh destination")
    trial.configure()
    roots = (trial.core.corpus.DEST, trial.PILOT, trial.TRAIN, trial.RECON)
    for root in roots:
        trial.core.terminal(root)
    load = lambda path: json.loads(path.read_text())
    trained = load(trial.TRAIN / "result.json")
    pilot = load(trial.PILOT / "result.json")
    measured = load(trial.RECON / "result.json")
    previous = load(trial.FAILED_RECON / "result.json")
    if (trained["updates"] != 2048 or not trained["reload_exact"] or not pilot["reload_exact"]
            or not trained["initial_inference_matches_original_exactly"]
            or not trained["all_sampled_windows_and_LRs_match_unscaled_run"]
            or trained["initial_model_sha256"] != pilot["initial_model_sha256"]
            or file_hash(trial.TRAIN / "model.pt") != trained["checkpoint_sha256"]
            or not load(trial.RECON / "original_reference_equivalence.json")["all252_raw_outputs_codes_and_text_exact"]):
        raise ValueError("matched initial state, sampling, reload and original-reference controls required")
    before = [json.loads(line) for line in (trial.FAILED_TRAIN / "trace.jsonl").read_text().splitlines()]
    after = [json.loads(line) for line in (trial.TRAIN / "trace.jsonl").read_text().splitlines()]
    if len(before) != 2048 or len(after) != 2048 or any(
            (a["window_indices"], a["lr"]) != (b["window_indices"], b["lr"])
            for a, b in zip(before, after)):
        raise ValueError("sample lists and learning-rate schedules changed")
    cases = load(trial.core.corpus.DEST / "summary_diagnostic.json")
    origins = load(trial.core.corpus.DEST / "fit_origins.json")
    for key in ("source_id", "group_id", "source_sha256"):
        if {c[key] for c in cases} & {o[key] for row in origins for o in row["origins"]}:
            raise ValueError("codec fit/diagnostic source overlap")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(trial.core.corpus.GPT2, local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer = trial.FixedVocabularySize(tokenizer)
    components = {}
    for arm in ("original", "trained"):
        path = trial.RECON / (arm + "_outputs.json")
        if file_hash(path) != measured["output_hashes"][arm]:
            raise ValueError("raw outputs changed")
        components[arm] = replay(cases, load(path), tokenizer)
        if components[arm] != measured[arm]:
            raise ValueError("replayed counts differ")
    if load(trial.RECON / "original_outputs.json") != load(trial.FAILED_RECON / "original_outputs.json"):
        raise ValueError("full reference arm changed")
    passed = measurement.gate(components["original"], components["trained"])
    classification = "codec_reconstruction_gate_passed_semantic_audit_required" if passed else "codec_reconstruction_fidelity_failed"
    if (measured["classification"] != classification or measured["representation_gate"] != passed
            or measured["task_accuracy"] is not None or measured["source_prediction"]
            or not measured["oracle_target_codes"] or measured["NCM_success_established"]):
        raise ValueError("gate or task-accuracy scope changed")
    paths = [Path(__file__), Path(trial.__file__), trial.PLAN,
             Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    paths += [trial.FAILED_RECON / "result.json", trial.FAILED_TRAIN / "trace.jsonl",
              trial.FAILED_RESULTS / "campaign_results.json",
              trial.FAILED_RESULTS / "ncm-broad-learned-codec-reconstruction-v1_registry_readback.json"]
    aggregate = {"schema": "nal.ncm-sample-scaled-codec-ema.v1", "classification": classification,
        "training": trained, "pilot": pilot, "reconstruction": measured, "components": components,
        "previous_unscaled_components": {k: previous[k] for k in ("original", "trained")},
        "recovery_change_from_unscaled_pp": 100 * (components["trained"]["token_recovery"] - previous["trained"]["token_recovery"]),
        "all252_raw_codes_words_counts_replayed": True, "all252_original_outputs_identical": True,
        "all2048_sample_lists_LRs_identical": True, "fit_diagnostic_source_disjoint": True,
        "oracle_target_codes": True, "source_prediction": False, "task_accuracy_measured": False,
        "confirmed": False, "NCM_success_established": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    DEST.mkdir()
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    table = []
    for label, m in (("Released reference", components["original"]),
                     ("Unscaled EMA transfer", previous["trained"]),
                     ("Sample-scaled EMA transfer", components["trained"])):
        table.append(f"| {label} | {m['correct']}/{m['total']} ({100*m['token_recovery']:.3f}%) | {m['exact_texts']}/252 | {m['first_EOS_correct']}/252 | {m['changed_number_cases']} |")
    REPORT.write_text(f"""# Sample-scaled EMA for learned NCM codec transfer

Date: 2026-10-23. Hypothesis `{HYPOTHESIS}`.
Classification **{classification}**. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

| Codec | Correct non-PAD tokens | Exact texts | Correct first EOS | Changed number sequences |
| --- | --- | --- | --- | --- |
""" + "\n".join(table) + f"""

The matched change recovered {aggregate['recovery_change_from_unscaled_pp']:.3f} percentage points
of token recovery, but still trails the released reference and fails the
99.5% representation gate. All arms have zero active invalid vocabulary IDs.
The other gates require 99% first-EOS agreement and no increase in changed-number
cases. Changed number sequences are diagnostics, not independently judged factual
errors. The original failure remains recorded.

Both evaluations supply correct target-derived codes to each codec. These are
oracle reconstruction measurements, **not complete-summary model accuracy**.
The latest measured clean native teacher is 56/64 (87.5%); the previous NCM
hybrid is 41/64. Neither establishes the 95% goal. A new semantic reconstruction
audit can assess preserved meaning without retrospectively passing this study.

The released checkpoint uses EMA statistics consistent with 4,096-window updates;
our microbatch is four. The follow-up scales counts, vector sums and epsilon by
4/4096 and uses decay .99**(4/4096), leaving initial embedding vectors unchanged.
All 2,048 update sample lists and learning rates match the failed transfer.
Initial probe inference and all 252 original-reference outputs match exactly.
The two-update CUDA pilot, full checkpoint/optimizer/RNG/dropout reload, source
exclusion and replay of raw codes, words, texts and counts passed.

The controls hold data, seeds, gradient accumulation, precision, dropout and
optimizer schedule fixed. This isolates the EMA-unit intervention in this run;
it does not establish the missing upstream training recipe or eliminate other
transfer mismatches. Parent weights are absent from codec training and remain
frozen. No source-conditioned generator, final-test result or speedup is claimed.
Both final cohorts remain sealed. Cached tokenizer length reduces validation
overhead while retaining identical outputs; it is not model throughput evidence.
""")
    print(json.dumps({"status": "finalized", "classification": classification, "components": components}), flush=True)


def register():
    x = json.loads((DEST / "campaign_results.json").read_text())
    spec = {"name": "Matched sample-scaled EMA control for learned NCM codec transfer",
        "description": "2048 matched FP32 updates; sample-scaled EMA restores oracle token recovery from 67.17% to 94.26%, below released 95.18% and 99.5% representation gate. All252 raw outputs replayed. No source-only generation or task accuracy.",
        "confirmation_status": x["classification"], "task_accuracy_measured": False,
        "oracle_target_codes": True, "source_prediction": False, "NCM_success_established": False,
        "representation_gate": x["reconstruction"]["representation_gate"]}
    spec.update({arm + "_" + key: value for arm, metrics in x["components"].items() for key, value in metrics.items()})
    register_records(DEST / "campaign_results.json", REPORT, DEST, {HYPOTHESIS: spec})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
